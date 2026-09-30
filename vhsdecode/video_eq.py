import numpy as np
import scipy.signal as sps


class VideoEQ:
    """Sharpness control based on format paremers and sharpness setting"""

    def __init__(self, decoder_params, sharpness_level, freq_hz):
        # sharpness filter / video EQ
        corner = decoder_params["video_eq"]["loband"]["corner"]
        transition = decoder_params["video_eq"]["loband"]["transition"]
        self._b, self._a = sps.butter(
            *sps.buttord(corner, corner + transition, 3, 30, fs=freq_hz), "highpass", fs=freq_hz
        )
        # Filter state carried over between calls.
        self._zi = sps.lfilter_zi(self._b, self._a)

        self._gain = decoder_params["video_eq"]["loband"]["order_limit"]
        self._sharpness_level = sharpness_level

    def filter_video(self, demod):
        """It enhances the upper band of the video signal"""
        overlap = 10  # how many samples the edge distortion produces
        ha = sps.filtfilt(self._b, self._a, demod)
        hb, self._zi = sps.lfilter(self._b, self._a, demod[:overlap], zi=self._zi)
        hc = np.concatenate(
            (hb[:overlap], ha[overlap:])
        )  # edge distortion compensation, needs check
        hf = np.multiply(self._gain, hc)

        gain = self._sharpness_level
        result = np.multiply(np.add(np.roll(np.multiply(gain, hf), 0), demod), 1)

        return result
