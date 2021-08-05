from collections import namedtuple
from vhsdecode.addons.vsyncserration import VsyncSerration
import numpy as np
import vhsdecode.utils as utils
import itertools
from lddecode.utils import inrange
import lddecode.core as ldd
import math
import hashlib
from numba import njit


@njit(cache=True)
def check_levels(data, old_sync, new_sync, new_blank, vsync_hz_ref, hz_ire):
    """Check if adjusted levels give are somewhat sane."""
    # ldd.logger.info("am below new blank %s , amount below half_sync %s", amount_below, amount_below_half_sync)
    # ldd.logger.info("change %s _ %s", old_sync - new_sync, (hz_ire * 15))

    if (vsync_hz_ref - new_sync) > (hz_ire * 15):
        # Too far below format's standard sync to make sense
        return False
    if new_sync - old_sync < (hz_ire * 5):
        # Small change - probably ok
        return True

    amount_below = len(np.argwhere(data < new_sync)) / len(data)
    amount_below_half_sync = len(np.argwhere(data < new_blank)) / len(data)

    # If there is a lot of data below the detected vsync level, or almost no data below the detected
    # 50% of hsync level it's likely the levels are not correct, so avoid adjusting.
    if amount_below > 0.07 or amount_below_half_sync < 0.005:
        return False

    return True


"""Pulse definition for findpulses_n. Needs to be outside the function to work with numba.
Make sure to refresh numba cache if modified.
"""
Pulse = namedtuple("Pulse", "start len")


@njit(cache=True)
def findpulses_numba(sync_ref, low, high, min_synclen, max_synclen):
    """Locate possible pulses by looking at areas within some range."""
    mid_sync = high
    where_all_picture = np.where(sync_ref > mid_sync)[0]
    locs_len = np.diff(where_all_picture)
    # min_synclen = self.eq_pulselen * 1 / 8
    # max_synclen = self.linelen * 5 / 8
    is_sync = np.bitwise_and(locs_len > min_synclen, locs_len < max_synclen)
    where_all_syncs = np.where(is_sync)[0]
    pulses_starts = where_all_picture[where_all_syncs]
    pulses_lengths = locs_len[where_all_syncs]
    return [Pulse(z[0], z[1]) for z in zip(pulses_starts, pulses_lengths)]


# stores the last valid blacklevel, synclevel and vsynclocs state
# preliminary solution to fix spurious decoding halts (numpy error case)
class FieldState:
    def __init__(self, sysparams):
        self.SysParams = sysparams
        self.fv = self.SysParams["FPS"] * 2
        ma_depth = round(self.fv / 5) if self.fv < 60 else round(self.fv / 6)
        ma_min_watermark = int(ma_depth / 2)
        self.blanklevels = utils.StackableMA(
            window_average=ma_depth, min_watermark=ma_min_watermark
        )
        self.synclevels = utils.StackableMA(
            window_average=ma_depth, min_watermark=ma_min_watermark
        )
        self.locs = None

    def setSyncLevel(self, level):
        self.synclevels.push(level)

    def setLevels(self, sync, blank):
        self.blanklevels.push(blank)
        self.setSyncLevel(sync)

    def getSyncLevel(self):
        return self.synclevels.pull()

    def getLevels(self):
        blevels = self.blanklevels.pull()
        if blevels is not None:
            return self.getSyncLevel(), blevels
        else:
            return None, None

    def setLocs(self, locs):
        self.locs = locs

    def getLocs(self):
        return self.locs

    def hasLevels(self):
        return self.blanklevels.has_values() and self.synclevels.has_values()


class Resync:
    def __init__(self, fs, sysparams, debug=False):
        self.debug = debug
        self.samp_rate = fs
        self.SysParams = sysparams.copy()
        self.VsyncSerration = VsyncSerration(fs, sysparams)
        self.FieldState = FieldState(sysparams)
        self.eq_pulselen = self.VsyncSerration.getEQpulselen()
        self.linelen = self.VsyncSerration.getLinelen()

    def debug_field(self, sync_reference):
        ldd.logger.debug(
            "Hashed field sync reference %s"
            % hashlib.md5(sync_reference.tobytes("C")).hexdigest()
        )

    # checks for SysParams consistency
    def sysparams_consistency_checks(self, field):
        reclamp_ire0 = False
        # AGC is allowed to change two sysparams
        if field.rf.useAGC:
            if field.rf.SysParams["ire0"] != self.SysParams["ire0"]:
                ldd.logger.debug(
                    "AGC changed SysParams[ire0]: %.02f Hz", field.rf.SysParams["ire0"]
                )
                self.SysParams["ire0"] = field.rf.SysParams["ire0"].copy()
                reclamp_ire0 = True

            if field.rf.SysParams["hz_ire"] != self.SysParams["hz_ire"]:
                ldd.logger.debug(
                    "AGC changed SysParams[hz_ire]: %.02f Hz",
                    field.rf.SysParams["hz_ire"],
                )
                self.SysParams["hz_ire"] = field.rf.SysParams["hz_ire"].copy()

        if self.SysParams != field.rf.SysParams:
            ldd.logger.error("SysParams changed during runtime!")
            ldd.logger.debug("Original: %s" % self.SysParams)
            ldd.logger.debug("Altered : %s" % field.rf.SysParams)
            assert False, "SysParams changed during runtime!"

        return reclamp_ire0

    def fallback_vsync_loc_means(self, field, pulses):
        # determine sync pulses from vsync
        vsync_locs = []
        vsync_means = []

        for i, p in enumerate(pulses):
            if p.len > field.usectoinpx(10):
                vsync_locs.append(i)
                vsync_means.append(
                    np.mean(
                        field.data["video"]["demod_05"][
                            int(p.start + field.rf.freq) : int(
                                p.start + p.len - field.rf.freq
                            )
                        ]
                    )
                )

        return vsync_locs, vsync_means

    # search for black level on back porch
    def pulses_blacklevel(self, field, pulses, vsync_locs):
        # take the eq pulses before and after vsync
        r1 = range(vsync_locs[0] - 5, vsync_locs[0])
        r2 = range(vsync_locs[-1] + 1, vsync_locs[-1] + 6)

        black_means = []

        for i in itertools.chain(r1, r2):
            if i < 0 or i >= len(pulses):
                continue

            p = pulses[i]
            if inrange(p.len, field.rf.freq * 0.75, field.rf.freq * 3):
                black_means.append(
                    np.mean(
                        field.data["video"]["demod_05"][
                            int(p.start + (field.rf.freq * 5)) : int(
                                p.start + (field.rf.freq * 20)
                            )
                        ]
                    )
                )

        return black_means

    # search for sync and blanking levels from back porch
    def pulses_levels(self, field, pulses):
        vsync_locs, vsync_means = self.fallback_vsync_loc_means(field, pulses)

        if len(vsync_means) == 0:
            synclevel = self.FieldState.getSyncLevel()
            if synclevel is None:
                return None, None
        else:
            synclevel = np.median(vsync_means)
            self.FieldState.setSyncLevel(synclevel)
            self.FieldState.setLocs(vsync_locs)

        if vsync_locs is None or not len(vsync_locs):
            vsync_locs = self.FieldState.getLocs()

        # Now compute black level and try again
        black_means = self.pulses_blacklevel(field, pulses, vsync_locs)

        # Set to nan if empty to avoid warning.
        blacklevel = math.nan if len(black_means) == 0 else np.median(black_means)

        if np.isnan(blacklevel).any() or np.isnan(synclevel).any():
            # utils.plot_scope(field.data["video"]["demod_05"], title='Failed field demod05')
            sl, bl = self.FieldState.getLevels()
            if bl is not None and sl is not None:
                blacklevel, synclevel = bl, sl
            else:
                return None, None
        else:
            self.FieldState.setLevels(synclevel, blacklevel)

        return synclevel, blacklevel

    def findpulses_range(self, field, vsync_hz):
        sync_ire = field.rf.hztoire(vsync_hz)
        pulse_hz_min = field.rf.iretohz(sync_ire - 10)
        pulse_hz_max = field.rf.iretohz(sync_ire + 10)
        return pulse_hz_min, pulse_hz_max

        # lddu.findpulses() equivalent

    def findpulses(self, sync_ref, low, high):
        return findpulses_numba(
            sync_ref, low, high, self.eq_pulselen * 1 / 8, self.linelen * 5 / 8
        )

    def add_pulselevels_to_serration_measures(self, field):
        if self.VsyncSerration.hasSerration():
            sync, blank = self.VsyncSerration.getLevels()
        else:
            # it starts finding the sync from the minima in 5 ire steps
            ire_step = 5
            min_sync = np.min(field.data["video"]["demod_05"])
            retries = 30
            while retries > 0:
                pulse_hz_min, pulse_hz_max = self.findpulses_range(field, min_sync)
                pulses = self.findpulses(
                    field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
                )
                # this number might need calculation
                if len(pulses) > 100:
                    break
                min_sync = field.rf.iretohz(field.rf.hztoire(min_sync) + ire_step)
                retries -= 1

            sync, blank = self.pulses_levels(field, pulses)
            # chewed tape case
            if sync is None or blank is None:
                return

        # the tape chewing test passed, then it should find sync
        pulse_hz_min, pulse_hz_max = self.findpulses_range(field, sync)
        pulses = self.findpulses(
            field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
        )

        f_sync, f_blank = self.pulses_levels(field, pulses)
        if f_sync is not None and f_blank is not None:
            self.VsyncSerration.push_levels((f_sync, f_blank))

    # Do a level check
    def level_check(self, field, sync, blank, sync_reference):
        vsync_hz = field.rf.iretohz(field.rf.SysParams["vsync_ire"])
        return check_levels(
            sync_reference,
            vsync_hz,
            sync,
            blank,
            field.rf.sysparams_const.vsync_hz,
            field.rf.sysparams_const.hz_ire,
        )

    def getpulses_override(self, field):
        """Find sync pulses in the demodulated video signal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """

        sync_reference = field.data["video"]["demod_05"]
        if self.debug:
            self.debug_field(sync_reference)

        # measures the serration levels if possible
        self.VsyncSerration.work(sync_reference)
        # adds the sync and blanking levels from the back porch
        self.add_pulselevels_to_serration_measures(field)

        # safe clips the bottom of the sync pulses but leaves picture area unchanged
        # NOTE: Disabled for now as it doesn't seem to have much purpose at the moment and can
        # cause weird artifacts on the output.
        demod_data = (
            field.data["video"]["demod"]
            if not field.rf.sync_clip
            else self.VsyncSerration.safe_sync_clip(
                sync_reference, field.data["video"]["demod"]
            )
        )

        # if it has levels, then compensate blanking bias
        if self.VsyncSerration.hasLevels() or self.FieldState.hasLevels():
            if self.VsyncSerration.hasLevels():
                sync, blank = self.VsyncSerration.getLevels()
            else:
                sync, blank = self.FieldState.getLevels()

            if self.sysparams_consistency_checks(field):
                field.rf.SysParams["ire0"] = blank

            dc_offset = field.rf.SysParams["ire0"] - blank
            sync_reference += dc_offset
            if not field.rf.disable_dc_offset:
                demod_data += dc_offset
            sync, blank = sync + dc_offset, blank + dc_offset

            field.data["video"]["demod"] = demod_data
            field.data["video"]["demod_05"] = sync_reference
            pulse_hz_min, pulse_hz_max = self.findpulses_range(field, sync)
        else:
            # pass one using standard levels (fallback sync logic)
            # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
            pulse_hz_min, pulse_hz_max = self.findpulses_range(
                field, field.rf.iretohz(field.rf.SysParams["vsync_ire"])
            )

            # checks if the DC offset is abnormal before correcting it
            new_sync = self.VsyncSerration.mean_bias()
            vsync_hz = field.rf.iretohz(field.rf.SysParams["vsync_ire"])
            new_blank = field.rf.iretohz(field.rf.hztoire(new_sync) / 2)
            check = self.level_check(field, new_sync, new_blank, sync_reference)
            if (
                not field.rf.disable_dc_offset
                and not pulse_hz_min < new_sync < vsync_hz
                and check
            ):
                field.data["video"]["demod_05"] = sync_reference - new_sync + vsync_hz
                field.data["video"]["demod"] = demod_data - new_sync + vsync_hz

        # utils.plot_scope(field.data["video"]["demod_05"])
        return self.findpulses(
            field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
        )
