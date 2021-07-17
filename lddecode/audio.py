#!/usr/bin/python3

# Draft of new-audio code

import argparse
import copy
import itertools
import sys
import threading
import time

from functools import partial
from multiprocessing import Process, Queue, JoinableQueue, Pipe

# standard numeric/scientific libraries
import numpy as np
from numpy.lib.function_base import _parse_gufunc_signature
import scipy.signal as sps
import scipy.interpolate as spi

# Use PyFFTW's faster FFT implementation if available
try:
    import pyfftw.interfaces.numpy_fft as npfft
    import pyfftw.interfaces

    pyfftw.interfaces.cache.enable()
    pyfftw.interfaces.cache.set_keepalive_time(10)
except ImportError:
    import numpy.fft as npfft

# internal libraries

# XXX: figure out how to handle these module imports better for vscode imports
try:
    import efm_pll
except ImportError:
    from lddecode import efm_pll

try:
    import core
except ImportError:
    from lddecode import core

try:
    import utils
except ImportError:
    from lddecode import utils

try:
    import utils_logging 
except ImportError:
    from lddecode import utils_logging

try:
    # If Anaconda's numpy is installed, mkl will use all threads for fft etc
    # which doesn't work when we do more threads, do disable that...
    import mkl

    mkl.set_num_threads(1)
except ImportError:
    # If not running Anaconda, we don't care that mkl doesn't exist.
    pass

# globals

blocklen = 32768
drop_begin = 4096-512
drop_end = 512
blockskip = drop_begin + drop_end

def audio_bandpass_butter(center, closerange = 125000, longrange = 180000):
    ''' Returns filter coefficients for first stage per-channel filtering '''
    freqs_inner = [(center - closerange) / freq_hz_half, (center + closerange) / freq_hz_half]
    freqs_outer = [(center - longrange) / freq_hz_half, (center + longrange) / freq_hz_half]
    
    N, Wn = sps.buttord(freqs_inner, freqs_outer, 1, 15)

    return sps.butter(N, Wn, btype='bandpass')

class AudioRF:
    def __init__(self, freq, center_freq):
        self.center_freq = center_freq

        # Set up SysParams to hand off to needed sub-tasks
        SysParams = core.SysParams_PAL if args.vid_standard == 'PAL' else core.SysParams_NTSC

        self.freq = freq
        self.freq_hz = self.freq * 1.0e6
        self.freq_hz_half = self.freq_hz / 2

        SysParams['audio_filterwidth'] = 150000

        apass = SysParams['audio_filterwidth']
        afilt_len = 512

        # Compute stage 1 filters

        audio1_fir = utils.filtfft([sps.firwin(afilt_len, [(self.center_freq-apass)/self.freq_hz_half, (self.center_freq+apass)/self.freq_hz_half], pass_zero=False), 1.0], blocklen)
        lowbin, nbins, a1_freq = utils.fft_determine_slices(self.center_freq, 200000, self.freq_hz, blocklen)
        hilbert = utils.build_hilbert(nbins)

        self.a1_freq = a1_freq

        # Add the demodulated output to this to get actual freq
        self.low_freq = self.freq_hz * (lowbin / blocklen)
        
        self.slicer = lambda x: utils.fft_do_slice(x, lowbin, nbins, blocklen)
        self.filt1 = self.slicer(audio1_fir) * hilbert
        self.filt1f = audio1_fir * utils.build_hilbert(blocklen)
        
        self.audio1_buffer = utils.StridedCollector(blocklen, blockskip)
        self.audio1_clip = blockskip // (blocklen // nbins)

        # Compute stage 2 audio filters: 20k-ish LPF and deemp (center_freq independent)

        N, Wn = sps.buttord(20000 / (a1_freq / 2), 24000 / (a1_freq / 2), 1, 9)
        audio2_lpf = utils.filtfft(sps.butter(N, Wn), blocklen)
        # 75e-6 is 75usec/2133khz (matching American FM emphasis) and 5.3e-6 is approx
        # a 30khz break frequency
        audio2_deemp = utils.filtfft(utils.emphasis_iir(5.3e-6, 75e-6, a1_freq), blocklen)

        self.audio2_decimation = 4
        self.audio2_filter = audio2_lpf * audio2_deemp

    def process_stage1(self, fft_in):
        ''' Apply first state audio filters '''
        a1 = npfft.ifft(self.slicer(fft_in) * self.filt1)
        a1u = utils.unwrap_hilbert(a1, self.a1_freq)
        a1u = a1u[self.audio1_clip:]

        return self.audio1_buffer.add(a1u)

    def process_stage2(self):
        ''' Applies second stage audio filters '''

        if not self.audio1_buffer.have_block():
            return None

        a2_in = self.audio1_buffer.get_block()
        a2_fft = npfft.fft(a2_in)
        a2 = npfft.ifft(a2_fft * self.audio2_filter)

        filtered = utils.sqsum(a2[drop_begin:-drop_end])

        return filtered[::self.audio2_decimation]

class AudioDecoder:

    def __init__(self, args):
        global blocklen, blockskip

        self.out_fd = None
        self.efm_fd = None

        # Have input_buffer store 8-bit bytes, then convert afterwards
        self.input_buffer = utils.StridedCollector(blocklen*2, blockskip*2)

        # Set up SysParams to hand off to needed sub-tasks
        SysParams = core.SysParams_PAL if args.vid_standard == 'PAL' else core.SysParams_NTSC

        self.freq = args.freq
        self.freq_hz = self.freq * 1.0e6
        self.efm_filter = efm_pll.computeefmfilter(self.freq_hz, blocklen)
        #print(self.freq_hz, len(self.efm_filter), self.efm_filter, file=sys.stderr)

        self.aa_channels = []

        if True:
            self.aa_channels.append(AudioRF(self.freq, SysParams['audio_lfreq']))
            
        if True:
            self.aa_channels.append(AudioRF(self.freq, SysParams['audio_rfreq']))


    def process_input_buffer(self):
        while self.input_buffer.have_block():
            buf = self.input_buffer.get_block()
            b = buf.tobytes()
            s16 = np.frombuffer(b, 'int16', len(b)//2)

            fft_in = npfft.fft(s16)

            if not args.daa:

                outputs = []

                for channel in self.aa_channels:
                    if channel.process_stage1(fft_in):
                        # if True, we have enough data to process stage2
                        outputs.append(channel.process_stage2())

                if len(outputs) == 0:
                    pass
                elif (len(outputs) != len(self.aa_channels)):
                    print("ERROR: mismatch in # of processed channels")
                    sys.exit(-1)
                else:
                    ofloat = []

                    for output, channel in zip(outputs, self.aa_channels):
                        o = output + channel.low_freq - channel.center_freq
                        #print(np.mean(o), np.std(o), channel.low_freq, channel.center_freq)
                        ofloat.append(np.clip((o / 150000), -16, 16).astype(np.float32))
                    
                    if len(outputs) == 2:
                        #print(len(outputs), np.mean(o32[0]), np.std(o32[0]), np.std(o32[1]))
                        
                        outdata = np.zeros(len(ofloat[0]) * 2, dtype=np.float32)
                        outdata[0::2] = ofloat[0]
                        outdata[1::2] = ofloat[1]
                    else:
                        outdata = np.array(ofloat[0], dtype=np.float32)
                        
                    if self.out_fd is not None:
                        self.out_fd.write(outdata)

            if efm_decode:
                filtered_efm = npfft.ifft(fft_in * self.efm_filter)[drop_begin:-drop_end]
                filtered_efm2 = np.int16(np.clip(filtered_efm.real, -32768, 32767))

                if args.prefm:
                    rawefm_fd.write(filtered_efm2.tobytes())

                efm_out = efm_pll_object.process(filtered_efm2)
                #print(efm_out.shape, max(filtered_efm.imag))
                
                if self.efm_fd is not None:
                    #print(len(buf), len(filtered_efm2), len(efm_out), file=sys.stderr)
                    efm_fd.write(efm_out.tobytes())

# Command line front end code

def handle_options(argstring = sys.argv):
    options_epilog = """foof"""

    parser = argparse.ArgumentParser(
        description="Extracts audio from raw/TBC'd RF laserdisc captures",
    #    epilog=options_epilog,
    )

    # This is -i instead of a positional, since the first pos can't be overridden
    parser.add_argument("-i", dest="infile", default='-', type=str, help="source file (must be signed 16-bit)")

    parser.add_argument("-o", dest="outfile", default='test', type=str, help="base name for destination files")
    
    parser.add_argument("-f", "--freq", dest='freq', default = "40.0mhz", type=utils.parse_frequency, help="Input frequency")

    parser.add_argument(
        "--disable_analog_audio",
        "--disable_analogue_audio",
        "--daa",
        dest="daa",
        action="store_true",
        default=False,
        help="Disable analog(ue) audio decoding",
    )

    parser.add_argument(
        "--PAL",
        "-p",
        "--pal",
        dest="pal",
        action="store_true",
        help="source is in PAL format",
    )

    parser.add_argument(
        "--NTSC", "-n", "--ntsc", dest="ntsc", action="store_true", help="source is in NTSC format"
    )

    parser.add_argument(
        "--noEFM",
        dest="noefm",
        action="store_true",
        default=False,
        help="Disable EFM front end",
    )

    parser.add_argument(
        "--preEFM",
        dest="prefm",
        action="store_true",
        default=False,
        help="Write filtered but otherwise pre-processed EFM data",
    )
    
    args = parser.parse_args(argstring[1:])

    if args.pal and args.ntsc:
        print("ERROR: Can only be PAL or NTSC")
        exit(1)

    args.vid_standard = "PAL" if args.pal else "NTSC"
    
    return args

testmode = False
if testmode:
    args = handle_options(['--NTSC', '-i', '../ggvsweep1.tbc.r16'])
else:
    args = handle_options(sys.argv)

if args.infile == '-':
    in_fd = None
else:
    in_fd = open(args.infile, 'rb')

if args.outfile == '-':
    out_fd = sys.stdout
    args.prefm = False
    efm_decode = False
else:
    if not args.daa:
        out_fd = open(args.outfile + '.pcmf32', 'wb')

    if args.prefm:
        rawefm_fd = open(args.outfile + '.prefm', 'wb')
    else:
        args.prefm = False

    efm_decode = not args.noefm
    if efm_decode:
        efm_pll_object = efm_pll.EFM_PLL()
        efm_fd = open(args.outfile + '.efm', 'wb')

ad = AudioDecoder(args)

# FIXME
ad.out_fd = out_fd
ad.efm_fd = efm_fd

# Common top-level code
logger = utils_logging.init_logging(None)

inputs = []
output = []
allout = []

while True:
    if args.infile == '-':
        inbuf = sys.stdin.buffer.read(65536)
    else:
        inbuf = in_fd.read(65536)

    if len(inbuf) == 0:
        break
        
    # store the input buffer as a raw 8-bit data, then repack into 16-bit
    # (this allows reading of an odd # of bytes)
    ad.input_buffer.add(np.frombuffer(inbuf, 'int8', len(inbuf)))
    ad.process_input_buffer()


