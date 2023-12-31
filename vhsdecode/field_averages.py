from vhsdecode.utils import StackableMA


class FieldAverage:
    def __init__(self):
        self._rf_level = StackableMA()
        self._chroma_level = StackableMA()
        # self.line_length = StackableMA()
        # self.vsync_dist = StackableMA

    @property
    def rf_level(self):
        return self._rf_level

    @property
    def chroma_level(self):
        return self._chroma_level
