#!/usr/bin/python3

import io
from io import BytesIO
import re
import subprocess
import sys

from multiprocessing import Process, Pool, Queue, JoinableQueue, Pipe
import threading
import queue

import numpy as np
import scipy as sp
import scipy.signal as sps

import matplotlib
import matplotlib.pyplot as plt

# To support image displays
from PIL import Image
import IPython.display
from IPython.display import HTML


def todb(y, zero=False):
    db = 20 * np.log10(np.abs(y))
    if zero:
        return db - np.max(db)
    else:
        return db


def print_crossings(db, w):
    above_m3 = None
    for i in range(1, len(w) // 2):
        if (db[i] >= -10) and (db[i - 1] < -10):
            print(">-10db crossing at ", w[i])
        if (db[i] >= -3) and (db[i - 1] < -3):
            print(">-3db crossing at ", w[i])
            above_m3 = i
        if (db[i] < -3) and (db[i - 1] >= -3):
            if above_m3 is not None:
                peak_index = np.argmax(db[above_m3:i]) + above_m3
                print("peak at ", w[peak_index], db[peak_index])
            print("<-3db crossing at ", w[i])
        if (db[i] >= 3) and (db[i - 1] < 3):
            print(">3db crossing at ", w[i])


def plotfilter_wh(w, h, freq, zero_base=False):
    db = todb(h, zero_base)

    print_crossings(db, w)

    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.set_title("Digital filter frequency response")

    ax1.plot(w, db, "b")
    ax1.set_ylabel("Amplitude [dB]", color="b")
    ax1.set_xlabel("Frequency [rad/sample]")

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    ax2.plot(w, angles, "g")
    ax2.set_ylabel("Angle (radians)", color="g")

    plt.grid()
    plt.axis("tight")
    plt.show()

    return None


def plotfilter(B, A, dfreq=None, freq=40, zero_base=False):
    if dfreq is None:
        dfreq = freq / 2

    w, h = sps.freqz(B, A, whole=True, worN=4096)
    w = np.arange(0, freq, freq / len(h))

    keep = int((dfreq / freq) * len(h))

    return plotfilter_wh(w[1:keep], h[1:keep], freq, zero_base)


def plotfilter2(F1, F2, dfreq=None, freq=40, zero_base=False):
    if dfreq is None:
        dfreq = freq / 2

    w, h1 = sps.freqz(F1[0], F1[1], whole=True, worN=4096)
    w = np.arange(0, freq, freq / len(h1))
    db1 = todb(h1, zero_base)

    print_crossings(db1, w)

    w, h2 = sps.freqz(F2[0], F2[1], whole=True, worN=4096)
    w = np.arange(0, freq, freq / len(h1))
    db2 = todb(h2, zero_base)

    print("second filter:")
    print_crossings(db2, w)

    keep = int((dfreq / freq) * len(h1))

    fig, ax1 = plt.subplots(1, 1, sharex=True)
    ax1.set_title("Digital filter frequency response")

    ax1.plot(w[:keep], db1[:keep], "b")
    ax1.plot(w[:keep], db2[:keep], "g")
    ax1.set_ylabel("Amplitude [dB]", color="b")
    ax1.set_xlabel("Frequency [rad/sample]")

    plt.grid()
    plt.axis("tight")
    plt.show()

    return None


pi = np.pi
tau = np.pi * 2

# Plotting routines


def dosplot(B, A, freq=(315.0 / 88.0) * 8.0):
    w, h = sps.freqz(B, A)

    fig = plt.figure()
    plt.title("Digital filter frequency response")

    ax1 = fig.add_subplot(111)

    db = 20 * np.log10(abs(h))

    for i in range(1, len(w)):
        if (db[i] >= -10) and (db[i - 1] < -10):
            print(">-10db crossing at ", w[i] * (freq / pi) / 2.0)
        if (db[i] >= -3) and (db[i - 1] < -3):
            print("-3db crossing at ", w[i] * (freq / pi) / 2.0)
        if (db[i] < -3) and (db[i - 1] >= -3):
            print("<-3db crossing at ", w[i] * (freq / pi) / 2.0)
        if (db[i] < -10) and (db[i - 1] >= -10):
            print("<-10db crossing at ", w[i] * (freq / pi) / 2.0)
        if (db[i] < -20) and (db[i - 1] >= -20):
            print("<-20db crossing at ", w[i] * (freq / pi) / 2.0)

    plt.plot(w * (freq / pi) / 2.0, 20 * np.log10(abs(h)), "b")
    plt.ylabel("Amplitude [dB]", color="b")
    plt.xlabel("Frequency [rad/sample]")

    plt.show()


def doplot(B, A, freq=(315.0 / 88.0) * 8.0):
    w, h = sps.freqz(B, A)

    fig = plt.figure()
    plt.title("Digital filter frequency response")

    db = 20 * np.log10(abs(h))
    for i in range(1, len(w)):
        if (db[i] >= -10) and (db[i - 1] < -10):
            print(">-10db crossing at ", w[i] * (freq / pi) / 2.0)
        if (db[i] >= -3) and (db[i - 1] < -3):
            print(">-3db crossing at ", w[i] * (freq / pi) / 2.0)
        if (db[i] < -3) and (db[i - 1] >= -3):
            print("<-3db crossing at ", w[i] * (freq / pi) / 2.0)

    ax1 = fig.add_subplot(111)

    plt.plot(w * (freq / pi) / 2.0, 20 * np.log10(abs(h)), "b")
    plt.ylabel("Amplitude [dB]", color="b")
    plt.xlabel("Frequency [rad/sample]")

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    plt.plot(w * (freq / pi) / 2.0, angles, "g")
    plt.ylabel("Angle (radians)", color="g")

    plt.grid()
    plt.axis("tight")
    plt.show()


def doplot2(B, A, B2, A2, freq=(315.0 / 88.0) * 8.0):
    w, h = sps.freqz(B, A)
    w2, h2 = sps.freqz(B2, A2)

    # 	h.real /= C
    # 	h2.real /= C2

    begin = 0
    end = len(w)
    # 	end = int(len(w) * (12 / freq))

    # 	chop = len(w) / 20
    chop = 0
    w = w[begin:end]
    w2 = w2[begin:end]
    h = h[begin:end]
    h2 = h2[begin:end]

    v = np.empty(len(w))

    # 	print len(w)

    hm = np.absolute(h)
    hm2 = np.absolute(h2)

    v0 = hm[0] / hm2[0]
    for i in range(0, len(w)):
        # 		print i, freq / 2 * (w[i] / pi), hm[i], hm2[i], hm[i] / hm2[i], (hm[i] / hm2[i]) / v0
        v[i] = (hm[i] / hm2[i]) / v0

    fig = plt.figure()
    plt.title("Digital filter frequency response")

    ax1 = fig.add_subplot(111)

    v = 20 * np.log10(v)

    # 	plt.plot(w * (freq/pi) / 2.0, v)
    # 	plt.show()
    # 	exit()

    plt.plot(w * (freq / pi) / 2.0, 20 * np.log10(abs(h)), "r")
    plt.plot(w * (freq / pi) / 2.0, 20 * np.log10(abs(h2)), "b")
    plt.ylabel("Amplitude [dB]", color="b")
    plt.xlabel("Frequency [rad/sample]")

    ax2 = ax1.twinx()
    angles = np.unwrap(np.angle(h))
    angles2 = np.unwrap(np.angle(h2))
    plt.plot(w * (freq / pi) / 2.0, angles, "g")
    plt.plot(w * (freq / pi) / 2.0, angles2, "y")
    plt.ylabel("Angle (radians)", color="g")

    plt.grid()
    plt.axis("tight")
    plt.show()


# This matches FDLS-based conversion surprisingly well (i.e. FDLS is more accurate than I thought ;) )
def BA_to_FFT(B, A, blocklen):
    return np.complex64(sps.freqz(B, A, blocklen, whole=True)[1])


# Notebook support functions follow

# Draws a uint16 image, downscaled to uint8
def draw_raw_bwimage(bm, x=2800, y=525, hscale=1, vscale=2, outsize=None):
    if y is None:
        y = len(bm) // x

    if outsize is None:
        outsize = (x * hscale, y * vscale)

    bmf = np.uint8(bm[0 : x * y] / 256.0)
    #    print(bmf.shape)
    if x is not None:
        bms = bmf.reshape(len(bmf) // x, -1)
    else:
        bms = bmf

    #    print(bms.dtype, bms.shape, bms[:][0:y].shape)
    im = Image.fromarray(bms[0:y])
    im = im.resize(outsize)
    b = BytesIO()
    im.save(b, format="png")
    return IPython.display.Image(b.getvalue())


def draw_field(field):
    return draw_raw_bwimage(field.dspicture, field.outlinelen, field.outlinecount)


def draw_raw_field(self, channel="demod"):
    # Draws the pre-TBC field.  Useful for determining if there's a skip (i.e. issue #509)
    cooked = self.hz_to_output(
        self.data["video"][channel][int(self.linelocs[0]) : int(self.linelocs[-1])]
    )

    return draw_raw_bwimage(cooked, self.inlinelen, len(self.linelocs) - 2, vscale=4)


def draw_really_raw_field(self, channel="demod"):
    # Draws the pre-TBC field.  Useful for determining if there's a skip (i.e. issue #509)
    cooked = self.hz_to_output(self.data["video"][channel])
    cooked = cooked[: (len(cooked) // self.inlinelen) * self.inlinelen]

    return draw_raw_bwimage(
        cooked, self.inlinelen, (len(cooked) // self.inlinelen), vscale=4
    )


def plotline(field, line, offset=0, usecs=63.5, linelocs=None, data="demod"):
    ls = field.lineslice(line, offset, usecs - offset, linelocs)
    plt.plot(field.data["video"][data][ls])


class RGBoutput:
    def __init__(self, outname):
        try:
            rv = subprocess.run(
                "ld-chroma-decoder {0}.tbc {0}.rgb".format(outname),
                capture_output=True,
                shell=True,
            )
        except:

            return None

        if rv.returncode != 0:
            print("Failed to run ld-chroma-decoder: ", rv.returncode)
            print(rv.stderr.split("\n"))

        stderr = rv.stderr.decode("utf-8")

        outres = re.search("trimmed to ([0-9]{3}) x ([0-9]{3})", stderr)
        outframes = re.search("complete - ([0-9]*) frames", stderr)

        if outres is None or outframes is None:
            print("Error, did not decode correctly")

        self.x, self.y = [int(v) for v in outres.groups()]
        self.numframes = int(outframes.groups()[0])

        with open("{0}.rgb".format(outname), "rb") as fd:
            # 3 colors, 2 bytes/color
            raw = fd.read((self.x * self.y * 3 * 2 * self.numframes))

        self.rgb = np.frombuffer(raw, "uint16", len(raw) // 2)

    def lineslice(self, frame, line):
        if line >= self.y or frame > self.numframes:
            return None

        def getslice(offset):
            return slice(
                ((self.y * frame) + line) * (self.x * 3) + offset,
                ((self.y * frame) + line + 1) * (self.x * 3) + offset,
                3,
            )

        return [getslice(offset) for offset in range(3)]

    def plotline(self, frame, line):
        sl = self.lineslice(frame, line)

        if sl is None:
            return None

        plt.plot(self.rgb[sl[0]], "r")
        plt.plot(self.rgb[sl[1]], "g")
        plt.plot(self.rgb[sl[2]], "b")

    def display(self, framenr, scale=1):
        begin = self.lineslice(framenr, 0)[0].start
        end = self.lineslice(framenr, self.y - 1)[0].stop

        rawimg = self.rgb[begin:end].reshape((self.y, self.x, 3))
        rawimg_u8 = np.uint8(rawimg // 256)
        im = Image.fromarray(rawimg_u8)
        b = BytesIO()
        im.save(b, format="png")

        return IPython.display.Image(
            b.getvalue(), width=int(self.x * scale), height=int(self.y * scale)
        )
