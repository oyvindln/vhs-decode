from vhsdecode.utils import StackableMA
from collections import deque


class FieldAverage:
    def __init__(self, chroma_agc_fields, group_delay_len):
        self._rf_level = StackableMA()
        
        # disabled by default, add docs indicating that this should be enable for noisy / home video
        self.chroma_level_len = chroma_agc_fields
        self._chroma_level_even = deque(maxlen=self.chroma_level_len)
        self._chroma_level_odd = deque(maxlen=self.chroma_level_len)
        
        # -1 means this is disabled
        if group_delay_len == -1:
            group_delay_len = 0

        self.group_delay_len = group_delay_len
        self._group_delay = {}
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
    
    # state for the group delay processing that happens in FieldShared.downscale
    @property
    def group_delay(self):
        return self._group_delay
