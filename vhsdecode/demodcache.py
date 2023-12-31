import time
from lddecode.core import DemodCache


class DemodCacheTape(DemodCache):
    def __init__(self, *args, **kwargs):
        super(DemodCacheTape, self).__init__(*args, **kwargs)

    def worker(self):
        """Override to skip mtf stuff since that's laserdisc specific."""
        blocksrun = 0
        blockstime = 0

        rf = self.rf

        while True:
            item = self.q_in.get()

            if item is None or item[0] == "END":
                return

            if item[0] == "DEMOD":
                blocknum, block, target_MTF, request = item[1:]

                output = {}

                if "fft" not in block:
                    fftdata = None
                else:
                    fftdata = block["fft"]

                st = time.time()
                output["demod"] = rf.demodblock(
                    data=block["rawinput"],
                    fftdata=fftdata,
                    mtf_level=target_MTF,
                    cut=True,
                )
                blockstime += time.time() - st
                blocksrun += 1

                output["request"] = request
                output["MTF"] = 0  # Not used so just set to 0 for time.

                self.q_out.put((blocknum, output))
            elif item[0] == "NEWPARAMS":
                self.apply_newparams(item[1])
