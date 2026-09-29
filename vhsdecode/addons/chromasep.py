from fractions import Fraction
import numpy as np
from scipy import signal
from soxr import resample


def signal_resample(data, output_rate, input_rate, converter_type="linear"):
    upsize = len(data) * output_rate
    downsize = round(upsize / input_rate)

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


# In ascending quality/complexity order:
#   https://sourceforge.net/p/soxr/code/ci/master/tree/src/soxr.h#l283
#   * QQ   'Quick' cubic interpolation.
#   * LQ   'Low' 16-bit with larger rolloff.
#   * MQ   'Medium' 16-bit with medium rolloff.
#   * HQ   'High quality'.
#   * VHQ  'Very high quality'.
def soxr_resample(data, input_rate, output_rate, quality="LQ"):
    return resample(data, input_rate, output_rate, quality)


class ChromaSepClass:
    def __init__(self, fs, fsc, logger, quality="QQ"):
        self.fs = fs
        self.fsc = fsc
        self.quality = quality
        self.multiplier = 8
        self.delay = int(self.multiplier / 2)

        fscx = int(self.fsc * self.multiplier * 1e6)
        self.ratio = Fraction(fscx / self.fs).limit_denominator(1000)

    # It resamples the luminance data to self.multiplier * fsc
    # Applies the comb filter, then resamples it back
    def work(self, luminance):
        downsampled = soxr_resample(
            luminance,
            self.ratio.denominator,
            self.ratio.numerator,
            self.quality,
        )

        delayed = np.roll(downsampled, self.delay)
        combed = np.multiply(np.add(downsampled, delayed), 0.5)
        del downsampled
        del delayed

        result = soxr_resample(
            combed,
            self.ratio.numerator,
            self.ratio.denominator,
            self.quality,
        )

        # Pad or truncate to the luminance length.
        if len(luminance) > len(result):
            err = len(luminance) - len(result)
            result = np.append(result, luminance[len(result) - err : len(result)])
        else:
            result = result[len(result) - len(luminance) :]
        assert len(luminance) == len(result), (
            "Something wrong happened during the comb filtering stage. Expected samples %d, got %d"
            % (len(luminance), len(result))
        )
        return result
