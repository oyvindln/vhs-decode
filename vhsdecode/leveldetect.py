import numpy as np
from numba import njit


@njit(cache=True)
def find_sync_levels(data, def_sync_level, def_blank_level, linefreq):
    """Very crude sync level detection"""
    # Skip a few samples to avoid any possible edge distortion.
    # data = field.data["video"]["demod_05"][10:]

    # Start with finding the minimum value of the input.
    sync_min = np.amin(data)
    max_val = np.amax(data)

    # Use the max for a temporary reference point which may be max ire or not.
    difference = max_val - sync_min

    # Find approximate sync areas.
    on_sync = data < (sync_min + (difference / 15))

    found_porch = False

    offset = 0
    blank_level = None

    while not found_porch:
        # Look for when we leave the approximate sync area next...
        search_start = np.argwhere(on_sync[offset:])[0][0]
        next_cross_raw = np.argwhere(1 - on_sync[search_start:])[0][0] + search_start
        # and a bit past that we ought to be in the back porch for blanking level.
        next_cross = next_cross_raw + int(1.5 * linefreq)
        blank_level = data[next_cross] + offset
        if blank_level > sync_min + (difference / 15):
            found_porch = True
        else:
            # We may be in vsync, try to skip ahead a bit
            # TODO: This may not work yet.
            offset += linefreq * 50
            if offset > len(data) - 10:
                # Give up
                return None, None

    if False:
        import matplotlib.pyplot as plt

        # data = field.data["video"]["demod_05"]

        fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
        # ax1.plot((20 * np.log10(self.Filters["Fdeemp"])))
        #        ax1.plot(hilbert, color='#FF0000')
        # ax1.plot(data, color="#00FF00")
        ax1.axhline(sync_min, color="#0000FF")
        #        ax1.axhline(blank_level, color="#000000")
        ax1.axvline(search_start, color="#FF0000")
        ax1.axvline(next_cross_raw, color="#00FF00")
        ax1.axvline(next_cross, color="#0000FF")
        ax1.axhline(blank_level, color="#000000")
        #            ax1.axhline(self.iretohz(self.SysParams["vsync_ire"]))
        #            ax1.axhline(self.iretohz(7.5))
        #            ax1.axhline(self.iretohz(100))
        # print("Vsync IRE", self.SysParams["vsync_ire"])
        #            ax2 = ax1.twinx()
        #            ax3 = ax1.twinx()
        ax1.plot(data)
        ax2.plot(on_sync)
        #            ax2.plot(luma05[:2048])
        #            ax4.plot(env, color="#00FF00")
        #            ax3.plot(np.angle(hilbert))
        #            ax4.plot(hilbert.imag)
        #            crossings = find_crossings(env, 700)
        #            ax3.plot(crossings, color="#0000FF")
        plt.show()
        #            exit(0)

    return sync_min, blank_level
