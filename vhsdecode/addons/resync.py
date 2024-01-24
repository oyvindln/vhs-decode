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


def iretohz(sysparams, ire):
    return sysparams.ire0 + (sysparams.hz_ire * ire)


def hztoire(sysparams, hz, ire0=None):
    if not ire0:
        ire0 = sysparams.ire0
    return (hz - ire0) / sysparams.hz_ire


@njit(cache=True, nogil=True)
def check_levels(data, old_sync, new_sync, new_blank, vsync_hz_ref, hz_ire, full=True):
    """Check if adjusted levels are somewhat sane."""
    # ldd.logger.info("am below new blank %s , amount below half_sync %s", amount_below, amount_below_half_sync)
    # ldd.logger.info("change %s _ %s", old_sync - new_sync, (hz_ire * 15))

    blank_sync_ire_diff = (new_blank - new_sync) / hz_ire

    # ldd.logger.info("ire diff: %s", blank_sync_ire_diff)

    # Check if too far below format's standard sync, or the difference between sync and blank is too large
    # to make sense
    if (vsync_hz_ref - new_sync) > (hz_ire * 15) or blank_sync_ire_diff > 47:
        return False
    if new_sync - old_sync < (hz_ire * 5):
        # Small change - probably ok
        return True

    if full:
        amount_below = len(np.argwhere(data < new_sync)) / len(data)
        amount_below_half_sync = len(np.argwhere(data < new_blank)) / len(data)

        # If there is a lot of data below the detected vsync level, or almost no data below the detected
        # 50% of hsync level it's likely the levels are not correct, so avoid adjusting.
        if amount_below > 0.07 or amount_below_half_sync < 0.005:
            return False

    return True


# search for black level on back porch
def _pulses_blacklevel(demod_05, freq_mhz: float, pulses, vsync_locs, synclevel):
    if not vsync_locs or len(vsync_locs) == 0:
        return None

    # take the eq pulses before and after vsync

    # We skip shorter pulses in case
    before_first = vsync_locs[0]
    after_last = vsync_locs[-1]
    last_index = len(pulses) - 1

    if len(vsync_locs) != 12:
        # Skip pulses that are way to close together to be vsync to avoid assuming noise are pulses.
        # TODO: use linelen and system for num pulses
        while (
            before_first > 1
            and pulses[before_first].start - pulses[before_first - 1].start < 600
        ):
            before_first -= 1

        while (
            after_last < last_index
            and pulses[after_last].start - pulses[after_last + 1].start < 600
        ):
            after_last += 1

    # TODO: This needs to be reworked for samples where the levels vary throughout the field
    r1 = range(max(before_first - 5, 1), before_first) if before_first > 1 else range(0)
    r2 = (
        range(after_last + 1, max(after_last + 6, last_index))
        if after_last < last_index - 1
        else range(0)
    )

    black_means = []

    for i in itertools.chain(r1, r2):
        if i < 0 or i >= len(pulses):
            continue

        p = pulses[i]
        if inrange(p.len, freq_mhz * 0.75, freq_mhz * 3):
            mean_value = np.mean(
                demod_05[int(p.start + (freq_mhz * 5)) : int(p.start + (freq_mhz * 20))]
            )
            black_means.append(mean_value)

    return black_means


"""Pulse definition for findpulses_n. Needs to be outside the function to work with numba.
Make sure to refresh numba cache if modified.
"""
Pulse = namedtuple("Pulse", "start len")

# Old impl for reference
# @njit(cache=True, nogil=True, parallel=True)
# def _findpulses_numba_raw_b(sync_ref, high, min_synclen, max_synclen):
#     """Locate possible pulses by looking at areas within some range.
#     Outputs arrays of starts and lengths
#     """
#     mid_sync = high
#     where_all_picture = np.where(sync_ref > mid_sync)[0]
#     locs_len = np.diff(where_all_picture)
#     # min_synclen = self.eq_pulselen * 1 / 8
#     # max_synclen = self.linelen * 5 / 8
#     is_sync = np.bitwise_and(locs_len > min_synclen, locs_len < max_synclen)
#     where_all_syncs = np.where(is_sync)[0]
#     pulses_lengths = locs_len[where_all_syncs]
#     pulses_starts = where_all_picture[where_all_syncs]
#     return pulses_starts, pulses_lengths


@njit(cache=True, nogil=True)
def _findpulses_numba_raw(sync_ref, high, min_synclen, max_synclen):
    """Locate possible pulses by looking at areas within some range.
    Outputs arrays of starts and lengths
    """

    in_pulse = sync_ref[0] <= high

    # Start/lengths lists
    # It's possible this could be optimized further by using a different data structure here.
    starts = []
    lengths = []

    cur_start = 0

    # Basic algorithm here is swapping between two states, going to the other one if we detect the
    # current sample passed the threshold.
    for pos, value in enumerate(sync_ref):
        if in_pulse:
            if value > high:
                length = pos - cur_start
                # If the pulse is in range, and it's not a starting one
                if inrange(length, min_synclen, max_synclen) and cur_start != 0:
                    starts.append(cur_start)
                    lengths.append(length)
                in_pulse = False
        elif value <= high:
            cur_start = pos
            in_pulse = True

    # Not using a possible trailing pulse
    # if in_pulse:
    #     # Handle trailing pulse
    #     length = len(sync_ref) - 1 - cur_start
    #     if inrange(length, min_synclen, max_synclen):
    #         starts.append(cur_start)
    #         lengths.append(length)

    return np.asarray(starts), np.asarray(lengths)


def _to_pulses_list(pulses_starts, pulses_lengths):
    """Make list of Pulse objects from arrays of pulses starts and lengths"""
    # Not using numba for this right now as it seemed to cause random segfault in tests.
    # list(map(Pulse, pulses_starts, pulses_lengths))
    return [Pulse(z[0], z[1]) for z in zip(pulses_starts, pulses_lengths)]


def _findpulses_numba(sync_ref, high, min_synclen, max_synclen):
    """Locate possible pulses by looking at areas within some range.
    .outputs a list of Pulse tuples
    """
    pulses_starts, pulses_lengths = _findpulses_numba_raw(
        sync_ref, high, min_synclen, max_synclen
    )
    return _to_pulses_list(pulses_starts, pulses_lengths)


def _fallback_vsync_loc_means(
    demod_05, pulses, sample_freq_mhz: float, min_len: int, max_len: int
):
    """Get the mean value of the video level inside pulses above a set threshold.

    Args:
        demod_05 ([type]): Video data to get levels from.
        pulses ([type]): List of detected pulses
        sample_freq_mhz (float): Sample frequency of the data in mhz
        min_len (int): only use pulses longer than this threshold.
        max_len (int): don't use pulses longer than this threshold.


    Returns:
        [type]: a list of vsync locations in the list of pulses and a list of mean values
    """
    vsync_locs = []
    vsync_means = []

    mean_pos_offset = sample_freq_mhz

    for i, p in enumerate(pulses):
        if p.len > min_len and p.len < max_len:
            vsync_locs.append(i)
            vsync_means.append(
                np.mean(
                    demod_05[
                        int(p.start + mean_pos_offset) : int(
                            p.start + p.len - mean_pos_offset
                        )
                    ]
                )
            )

    return vsync_locs, vsync_means


def findpulses_range(sp, vsync_hz, blank_hz=None):
    """Calculate half way point between blank and sync based on ire and reference
    uses calculated 0 IRE based on other params if blank_hz is not provided.
    """
    if not blank_hz:
        # Fall back to assume blank is at standard 0 ire.
        blank_hz = iretohz(sp, 0)
    sync_ire = hztoire(sp, vsync_hz)
    pulse_hz_min = iretohz(sp, sync_ire - 10)
    # Look for pulses at the halfway between vsync tip and blanking.
    pulse_hz_max = (iretohz(sp, sync_ire) + blank_hz) / 2
    return pulse_hz_min, pulse_hz_max


# stores the last valid blacklevel, synclevel and vsynclocs state
# preliminary solution to fix spurious decoding halts (numpy error case)
class FieldState:
    def __init__(self, sysparams):
        fv = sysparams["FPS"] * 2
        ma_depth = round(fv / 5) if fv < 60 else round(fv / 6)
        # ma_min_watermark = int(ma_depth / 2)
        # TODO: Set to 0 for now to start using detected levels on the first field
        # May want to alter later to do this more dynamically.
        ma_min_watermark = 0
        self._blanklevels = utils.StackableMA(
            window_average=ma_depth, min_watermark=ma_min_watermark
        )
        self._synclevels = utils.StackableMA(
            window_average=ma_depth, min_watermark=ma_min_watermark
        )
        self._locs = None

    def set_sync_level(self, level):
        self._synclevels.push(level)

    def set_levels(self, sync, blank):
        self._blanklevels.push(blank)
        self.set_sync_level(sync)

    def pull_sync_level(self):
        return self._synclevels.pull()

    def pull_levels(self):
        blevels = self._blanklevels.pull()
        if blevels is not None:
            return self.pull_sync_level(), blevels
        else:
            return None, None

    def set_locs(self, locs):
        self._locs = locs

    def get_locs(self):
        return self._locs

    def has_levels(self):
        return self._blanklevels.has_values() and self._synclevels.has_values()


class Resync:
    def __init__(self, fs, sysparams, sysparams_const, divisor=1, debug=False):
        self.divisor = divisor
        self.debug = debug
        self.samp_rate = fs

        if debug:
            self.SysParams = sysparams.copy()
        self._vsync_serration = VsyncSerration(fs, sysparams, divisor)
        self._field_state = FieldState(sysparams)
        self.eq_pulselen = self._vsync_serration.getEQpulselen()
        self.linelen = self._vsync_serration.get_line_len()
        self.use_serration = True
        # This should be enough to cover all "long" pulses,
        # longest variant being ones where all of vsync is just one long pulse.
        self._long_pulse_max = self.linelen * 5
        # Last half-way point between blank/sync we used when looking for pulses.
        self._last_pulse_threshold = findpulses_range(
            sysparams_const, sysparams_const.vsync_hz
        )

        # self._temp_c = 0

    @property
    def long_pulse_max(self):
        return self._long_pulse_max

    @property
    def last_pulse_threshold(self):
        return self._last_pulse_threshold

    def has_levels(self):
        return self._field_state.has_levels()

    def _debug_field(self, sync_reference):
        ldd.logger.debug(
            "Hashed field sync reference %s"
            % hashlib.md5(sync_reference.tobytes("C")).hexdigest()
        )

    def _pulses_blacklevel(self, field, pulses, vsync_locs, synclevel):
        return _pulses_blacklevel(
            field.data["video"]["demod_05"],
            field.rf.freq,
            pulses,
            vsync_locs,
            synclevel,
        )

    # checks for SysParams consistency
    def _sysparams_consistency_checks(self, field):
        reclamp_ire0 = False
        # AGC is allowed to change two sysparams
        if field.rf.useAGC:
            if not self.debug:
                ldd.logger.warning("Not doing consistency checks with debug disabled!")
                return False
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

        # if self.SysParams != field.rf.SysParams:
        #     ldd.logger.error("SysParams changed during runtime!")
        #     ldd.logger.debug("Original: %s" % self.SysParams)
        #     ldd.logger.debug("Altered : %s" % field.rf.SysParams)
        #     assert False, "SysParams changed during runtime!"

        return reclamp_ire0

    # search for sync and blanking levels from back porch
    def pulses_levels(
        self, field, sp, pulses, pulse_level=0, store_in_field_state=False
    ):
        vsync_len_px = field.usectoinpx(sp.vsync_pulse_us)
        min_len = vsync_len_px * 0.8
        max_len = vsync_len_px * 1.2

        vsync_locs, vsync_means = _fallback_vsync_loc_means(
            field.data["video"]["demod_05"], pulses, field.rf.freq, min_len, max_len
        )

        if len(vsync_means) == 0:
            synclevel = self._field_state.pull_sync_level()
            if synclevel is None:
                return None, None
        else:
            synclevel = np.median(vsync_means)
            self._field_state.set_sync_level(synclevel)
            self._field_state.set_locs(vsync_locs)

        # TODO: Think this was a bug - need to use absolute locs here,
        # not position in pulse list
        # if vsync_locs is None or not len(vsync_locs):
        #     vsync_locs = self._field_state.get_locs()

        # Now compute black level and try again
        black_means = self._pulses_blacklevel(field, pulses, vsync_locs, synclevel)
        # print("black_means", black_means)

        # Set to nan if empty to avoid warning.
        blacklevel = (
            math.nan
            if not black_means or len(black_means) == 0
            else np.median(black_means)
        )
        # If black level is below sync level, something has gone very wrong.
        if blacklevel < synclevel:
            blacklevel = math.nan

        if False:
            # store_in_field_state:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            fig.set_size_inches(40, 10)
            plt.text(1, 1, "black: %s sync: %s" % (blacklevel, synclevel))
            ax1.plot(field.data["video"]["demod_05"])
            ax1.axhline(synclevel)
            ax1.axhline(pulse_level, color="#00FF00")
            if blacklevel is not math.nan:
                ax1.axhline(blacklevel, color="#000000")
            for ps in pulses:
                ax2.axvline(ps.start, color="#00FF00")
                ax2.axvline(ps.start + ps.len, color="#0000FF")
            for loc in vsync_locs:
                ax2.axvline(pulses[loc].start, color="#FF0000")
            # ax2.plot(self.Filters["FVideo05"])
            plt.show()

        if np.isnan(blacklevel).any() or np.isnan(synclevel).any():
            ldd.logger.debug("blacklevel or synclevel had a NaN!")
            # utils.plot_scope(field.data["video"]["demod_05"], title='Failed field demod05')
            sl, bl = self._field_state.pull_levels()
            if bl is not None and sl is not None:
                blacklevel, synclevel = bl, sl
            else:
                return None, None
        else:
            # Make sure these levels are sane before using them.
            # Also don't save if we only found 1 or 2 vsyncs in case
            # they were false positives.
            if (
                self.level_check(
                    field.rf.sysparams_const,
                    synclevel,
                    blacklevel,
                    field.data["video"]["demod_05"],
                    True,
                )
                and len(vsync_means) > 3
            ):
                if store_in_field_state:
                    self._field_state.set_levels(synclevel, blacklevel)
            else:
                ldd.logger.debug("level check failed in pulses_levels!")
                return None, None

        return synclevel, blacklevel

    def findpulses(self, sync_ref, high):
        return _findpulses_numba(
            sync_ref, high, self.eq_pulselen * 1 / 8, self.long_pulse_max
        )

    def _findpulses_arr(self, sync_ref, high):
        return _findpulses_numba_raw(
            sync_ref, high, self.eq_pulselen * 1 / 8, self.long_pulse_max
        )

    def _findpulses_arr_reduced(self, sync_ref, high, divisor, sp):
        """Run findpulses using only every divisor samples"""
        min_len = (self.eq_pulselen * 1 / 8) / divisor
        max_len = (self.long_pulse_max) / divisor

        pulses_starts, pulses_lengths = _findpulses_numba_raw(
            sync_ref[::divisor], high, min_len, max_len
        )

        pulses_starts *= divisor
        pulses_lengths *= divisor

        return pulses_starts, pulses_lengths

    def add_pulselevels_to_serration_measures(
        self, field, demod_05, sp, check_long=False
    ):
        if self._vsync_serration.has_serration():
            sync, blank = self._vsync_serration.pull_levels()
        else:
            # it starts finding the sync from the minima in 5 ire steps
            ire_step = 5
            min_sync = np.min(demod_05)
            retries = 30
            min_vsync_check = field.usectoinpx(sp.vsync_pulse_us) * 0.8
            long_pulse_min = field.usectoinpx(sp.vsync_pulse_us) * 2.6
            long_pulse_max = self.long_pulse_max

            num_assumed_vsyncs_prev = 0
            long_pulses_prev = 0
            prev_min_sync = min_sync
            found_candidate = False
            check_next = True
            while retries > 0:
                pulse_hz_min, pulse_hz_max = findpulses_range(sp, min_sync)
                pulses_starts, pulses_lengths = self._findpulses_arr_reduced(
                    demod_05, pulse_hz_max, self.divisor, sp
                )

                # this number might need calculation
                if len(pulses_lengths) > 200:
                    # Check that at least 2 pulses are long enough to be vsync to avoid noise
                    # being counted as pulses
                    num_assumed_vsyncs = len(
                        pulses_lengths[pulses_lengths > min_vsync_check]
                    )

                    long_pulses = 0

                    if check_long and num_assumed_vsyncs <= 2:
                        # If requested, we also do checks for "long" pulses found in non-standard vsync.
                        long_pulses = len(
                            pulses_lengths[
                                inrange(pulses_lengths, long_pulse_min, long_pulse_max)
                            ]
                        )

                    if num_assumed_vsyncs > 4 or long_pulses >= 1:
                        if (
                            num_assumed_vsyncs == 12 or long_pulses == 2
                        ) and not check_next:
                            # if we have exactly 12 vsyncs meaning we likely found all of them and no more
                            # (or 2 long pulses) we are likely good.
                            break
                        elif (
                            not found_candidate
                            or num_assumed_vsyncs > num_assumed_vsyncs_prev
                            or long_pulses > long_pulses_prev
                        ):
                            # We found a set with at least some vsyncs or long pulses...
                            found_candidate = True
                            num_assumed_vsyncs_prev = num_assumed_vsyncs
                            long_pulses_prev = long_pulses
                            prev_min_sync = min_sync
                            # ...so do one more iteration to see if we get closer to the expected number,
                            # if not use the currently found levels.
                            check_next = True
                        elif (
                            num_assumed_vsyncs < num_assumed_vsyncs_prev
                            or long_pulses < long_pulses_prev
                            or check_next is False
                        ):
                            # We found less so go back to previous levels and end the search.
                            min_sync = prev_min_sync
                            pulse_hz_min, pulse_hz_max = findpulses_range(sp, min_sync)
                            pulses_starts, pulses_lengths = self._findpulses_arr(
                                demod_05, pulse_hz_max
                            )
                            break
                        else:
                            check_next = False

                min_sync = iretohz(sp, hztoire(sp, min_sync) + ire_step)
                retries -= 1

            pulses = _to_pulses_list(pulses_starts, pulses_lengths)

            sync, blank = self.pulses_levels(field, sp, pulses, pulse_hz_max)
            # chewed tape case
            if sync is None or blank is None:
                ldd.logger.debug("Level detection failed - sync or blank is None")
                return

        # the tape chewing test passed, then it should find sync
        pulse_hz_min, pulse_hz_max = findpulses_range(sp, sync, blank_hz=blank)

        pulses = self.findpulses(demod_05, pulse_hz_max)

        if False:
            import matplotlib.pyplot as plt

            # ldd.logger.info("hz to ire %s", hztoire(sp, blank))
            fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
            ax1.plot(field.data["video"]["demod_05"])
            ax1.axhline(sync)
            ax1.axhline(pulse_hz_max, color="#00FF00")
            # ax1.axhline((sync + blank) / 2.0 , color="#FF0000")
            ax1.axhline(blank, color="#000000")
            for p in pulses:
                ax2.axvline(p.start, color="#00FF00")
                ax2.axvline(p.start + p.len, color="#0000FF")
            # ax2.plot(self.Filters["FVideo05"])
            plt.show()

        f_sync, f_blank = self.pulses_levels(
            field, sp, pulses, pulse_hz_max, store_in_field_state=True
        )
        if f_sync is not None and f_blank is not None:
            self._vsync_serration.push_levels((f_sync, f_blank))
        else:
            ldd.logger.debug(
                "Level detection had issues, so don't store anything in VsyncSerration."
            )

    # Do a level check
    def level_check(self, sysparams_const, sync, blank, sync_reference, full=True):
        vsync_hz = (
            sysparams_const.vsync_hz
        )  # field.rf.iretohz(field.rf.SysParams["vsync_ire"])
        # TODO: See if we need to read vsync_ire from sysparams here
        return check_levels(
            sync_reference,
            vsync_hz,
            sync,
            blank,
            sysparams_const.vsync_hz,
            sysparams_const.hz_ire,
            full,
        )

    def get_pulses(self, field, check_levels=True):
        if self.use_serration:
            return self._get_pulses_serration(field, check_levels)
        else:
            import vhsdecode.leveldetect

            sync, blank = None, None
            if self._field_state.has_levels():
                sync, blank = self._field_state.pull_levels()
                pulses = self.findpulses(
                    field.data["video"]["demod_05"], (blank + sync) / 2
                )
                if len(pulses) > 200 and len(pulses) < 800:
                    return pulses
                ldd.logger.info("Re-checking levels..")

            def_sync = field.rf.iretohz(field.rf.SysParams["vsync_ire"])
            def_blank = field.rf.iretohz(field.rf.SysParams["ire0"])
            sync, blank = vhsdecode.leveldetect.find_sync_levels(
                field.data["video"]["demod_05"],
                def_sync,
                def_blank,
                field.get_linefreq(),
            )
            self._field_state.set_levels(sync, blank)

            return self.findpulses(field.data["video"]["demod_05"], (blank + sync) / 2)

    def _get_pulses_serration(self, field, check_levels):
        """Find sync pulses in the demodulated video signal"""

        sp = field.rf.sysparams_const

        sync_reference = field.data["video"]["demod_05"]
        if self.debug:
            self._debug_field(sync_reference)

        if check_levels or not self._field_state.has_levels():
            # measures the serration levels if possible
            self._vsync_serration.work(sync_reference)
            # adds the sync and blanking levels from the back porch
            self.add_pulselevels_to_serration_measures(
                field, sync_reference, sp, field.rf.options.fallback_vsync
            )

        # safe clips the bottom of the sync pulses but leaves picture area unchanged
        # NOTE: Disabled for now as it doesn't seem to have much purpose at the moment and can
        # cause weird artifacts on the output.
        demod_data = (
            field.data["video"]["demod"]
            # if not field.rf.options.sync_clip
            # else self._vsync_serration.safe_sync_clip(
            #    sync_reference, field.data["video"]["demod"]
            # )
        )

        # if self._temp_c == 1:
        #     np.savetxt("PAL_GOOD.txt.gz", sync_reference)
        # self._temp_c += 1

        # if it has levels, then compensate blanking bias
        if self._vsync_serration.has_levels() or self._field_state.has_levels():
            if self._vsync_serration.has_levels():
                new_sync, new_blank = self._vsync_serration.pull_levels()
                if self.level_check(sp, new_sync, new_blank, sync_reference):
                    sync, blank = new_sync, new_blank
                elif self._field_state.has_levels():
                    sync, blank = self._field_state.pull_levels()
                    ldd.logger.debug(
                        "Level check failed on serration measured levels [new_sync: %s, new_blank: %s], falling back to levels from FieldState [sync %s, blank %s].",
                        new_sync,
                        new_blank,
                        sync,
                        blank,
                    )
                else:
                    # Check failed on serration levels and field state does not contain levels
                    # TODO: Handle properly if this occurs
                    ldd.logger.debug(
                        "Level check failed on serration measured levels, using defaults."
                    )

                    sync = sp.ire0
                    blank = sp.vsync_hz
            else:
                sync, blank = self._field_state.pull_levels()

            if self._sysparams_consistency_checks(field):
                field.rf.SysParams["ire0"] = blank

            dc_offset = sp.ire0 - blank
            sync_reference += dc_offset
            if not field.rf.options.disable_dc_offset:
                demod_data += dc_offset
                field.data["video"]["demod"] = demod_data
            sync, blank = sync + dc_offset, blank + dc_offset

            field.data["video"]["demod_05"] = sync_reference
            pulse_hz_min, pulse_hz_max = findpulses_range(sp, sync)
        else:
            # pass one using standard levels (fallback sync logic)
            # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
            pulse_hz_min, pulse_hz_max = findpulses_range(sp, sp.vsync_hz)

            # checks if the DC offset is abnormal before correcting it
            new_sync = self._vsync_serration.mean_bias()
            vsync_hz = sp.vsync_hz
            new_blank = iretohz(sp, hztoire(sp, new_sync) / 2)

            check = self.level_check(sp, new_sync, new_blank, sync_reference)
            if (
                not field.rf.options.disable_dc_offset
                and not pulse_hz_min < new_sync < vsync_hz
                and check
            ):
                field.data["video"]["demod_05"] = sync_reference - new_sync + vsync_hz
                field.data["video"]["demod"] = demod_data - new_sync + vsync_hz

        self._last_pulse_threshold = pulse_hz_max

        return self.findpulses(field.data["video"]["demod_05"], pulse_hz_max)
