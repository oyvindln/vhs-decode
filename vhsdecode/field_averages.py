from vhsdecode.utils import StackableMA
from collections import deque


class FieldAverage:
    def __init__(self, chroma_agc_fields):
        self._rf_level = StackableMA()

        self.chroma_level_len = chroma_agc_fields
        self._chroma_level_even = deque(maxlen=self.chroma_level_len)
        self._chroma_level_odd = deque(maxlen=self.chroma_level_len)

        # self.line_length = StackableMA()
        # self.vsync_dist = StackableMA

    @property
    def rf_level(self):
        return self._rf_level

    # even field state for the chroma automatic gain control
    @property
    def chroma_level_even(self):
        return self._chroma_level_even
    
    # odd field state for the chroma automatic gain control
    @property
    def chroma_level_odd(self):
        return self._chroma_level_odd

