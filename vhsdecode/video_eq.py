import numpy as np
import vhsdecode.utils as utils
from vhsdecode.linear_filter import FiltersClass


class VideoEQ:
    """Sharpness control based on format paremers and sharpness setting"""

    def __init__(self, decoder_params, sharpness_level, freq_hz):
        # sharpness filter / video EQ
        iir_eq_loband = utils.firdes_highpass(
            freq_hz,
            decoder_params["video_eq"]["loband"]["corner"],
            decoder_params["video_eq"]["loband"]["transition"],
            decoder_params["video_eq"]["loband"]["order_limit"],
        )

        self._video_eq_filter = {
            0: FiltersClass(iir_eq_loband[0], iir_eq_loband[1], freq_hz),
            # 1: utils.FiltersClass(iir_eq_hiband[0], iir_eq_hiband[1], freq_hz),
        }

        self._gain = decoder_params["video_eq"]["loband"]["order_limit"]
        self._sharpness_level = sharpness_level

    def filter_video(self, demod):
        """It enhances the upper band of the video signal"""
        overlap = 10  # how many samples the edge distortion produces
        ha = self._video_eq_filter[0].filtfilt(demod)
        hb = self._video_eq_filter[0].lfilt(demod[:overlap])
        hc = np.concatenate(
            (hb[:overlap], ha[overlap:])
        )  # edge distortion compensation, needs check
        hf = np.multiply(self._gain, hc)

        gain = self._sharpness_level
        result = np.multiply(np.add(np.roll(np.multiply(gain, hf), 0), demod), 1)

        return result
