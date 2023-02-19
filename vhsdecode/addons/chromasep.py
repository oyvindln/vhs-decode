from fractions import Fraction
import vhsdecode.utils as utils
import numpy as np
from scipy import signal

try:
    from samplerate import resample

    use_samplerate = True
except ImportError as e:
    # print("[chromasep.py] WARN: Cannot find samplerate, processing will be slower")
    # print("exec:\n\tsudo pip3 install samplerate")
    # print("(to fix this inconvenience) %s" % e)
    use_samplerate = False


def signal_resample(data, n, d, converter_type="linear"):
    upsize = len(data) * n
    downsize = round(upsize / d)

    if converter_type == "linear":
        upscaled = np.interp(
            np.linspace(0.0, 1.0, upsize, endpoint=False),
            np.linspace(0.0, 1.0, len(data), endpoint=False),
            data,
        )
        return np.interp(
            np.linspace(0.0, 1.0, downsize, endpoint=False),
            np.linspace(0.0, 1.0, len(upscaled), endpoint=False),
            upscaled,
        )
    else:
        upscaled = signal.resample(data, upsize)
        return signal.resample(upscaled, downsize)


# The converter params are the same as in libsamplerate:
# In ascending quality/complexity order:
#   zero_order_hold, linear, sinc_fastest, sinc_best
def samplerate_resample(data, n, d, converter_type="linear"):
    ratio = n / d
    return resample(data, ratio=ratio, converter_type=converter_type)


class ChromaSepClass:
    def __init__(self, fs, fsc, converter_type="linear"):
        self.fs = fs
        self.fsc = fsc
        self.converter_type = converter_type
        self.multiplier = 8
        self.delay = int(self.multiplier / 2)

        fscx = int(self.fsc * self.multiplier * 1e6)
        self.ratio = Fraction(fscx / self.fs).limit_denominator(1000)

        if use_samplerate:
            self.method = samplerate_resample
        else:
            self.method = signal_resample

    # It resamples the luminance data to self.multiplier * fsc
    # Applies the comb filter, then resamples it back
    def work(self, luminance):
        downsampled = self.method(
            luminance,
            self.ratio.numerator,
            self.ratio.denominator,
            converter_type=self.converter_type,
        )

        delayed = np.roll(downsampled, self.delay)
        combed = np.multiply(np.add(downsampled, delayed), 0.5)
        del downsampled
        del delayed

        result = self.method(
            combed,
            self.ratio.denominator,
            self.ratio.numerator,
            converter_type=self.converter_type,
        )

        result = utils.pad_or_truncate(result, luminance)
        assert len(luminance) == len(result), (
            "Something wrong happened during the comb filtering stage. Expected samples %d, got %d"
            % (len(luminance), len(result))
        )
        return result
