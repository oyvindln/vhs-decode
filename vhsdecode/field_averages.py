from vhsdecode.utils import StackableMA
from collections import deque


class FieldAverage:
    def __init__(self):
        self._rf_level = StackableMA()
        
        # TODO: parameterize
        # disabled by default, add docs indicating that this should be enable for noisy / home video
        self.chroma_level_len = 0
        self._chroma_level_even = deque(maxlen=self.chroma_level_len)
        self._chroma_level_odd = deque(maxlen=self.chroma_level_len)
        
        # TODO: parameterize
        self.group_delay_len = 8
        self._group_delay = deque(maxlen=self.group_delay_len)
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
