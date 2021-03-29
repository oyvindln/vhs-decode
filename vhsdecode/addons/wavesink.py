import numpy as np
import wave
from samplerate import resample


class WaveSink:
    def __init__(self, fs=40e6, audio_rate=192000, name="wavesink_log.wav"):
        self.samp_rate = fs
        self.audio_rate = int(audio_rate)
        self.filename = name
        self.file = None
        self.init_file()
        self.scales = (1, 1)
        self.offsets = (0, 0)

    def init_file(self):
        self.file = wave.open(self.filename, 'wb')
        self.file.setsampwidth(2)
        self.file.setnchannels(2)
        self.file.setframerate(self.audio_rate)

    def __del__(self):
        self.file.close()

    def set_name(self, newname):
        if self.file is not None:
            self.file.close()
        self.filename = newname
        self.init_file()

    def set_rate(self, audio_rate):
        if self.file is not None:
            self.file.close()
        self.audio_rate = audio_rate
        self.init_file()

    def set_scale(self, scale):
        self.scales = scale

    def set_offset(self, offset):
        self.offsets = offset

    def to_wave(self, ch0, ch1):
        if max(ch0) > 1 or max(ch1) > 1:
            print('Wave signal clipping')

        ratio = self.audio_rate / self.samp_rate
        left = resample(np.multiply(ch0, 0x7FFF), ratio, converter_type='linear')
        right = resample(np.multiply(ch1, 0x7FFF), ratio, converter_type='linear')
        interleaved = np.zeros(len(left)+len(right))
        interleaved[::2] = left
        interleaved[1::2] = right
        return np.asarray(interleaved, dtype=np.int16)


    def write(self, ch0, ch1):
        wave = self.to_wave(
            np.multiply(np.add(ch0, self.offsets[0]), self.scales[0]),
            np.multiply(np.add(ch1, self.offsets[1]), self.scales[1]),
        )
        self.file.writeframes(wave)
