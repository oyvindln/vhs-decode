from vhsdecode.utils import (
    FiltersClass,
    firdes_lowpass,
    firdes_highpass,
    plot_scope,
    dualplot_scope,
    zero_cross_det,
    StackableMA,
)
import numpy as np
from scipy.signal import argrelextrema
from os import getpid
import lddecode.core as ldd


def t_to_samples(samp_rate, value):
    return samp_rate / value


class VsyncSerration:
    def __init__(self, fs, sysparams, show_decoded_serration=False):
        self.show_decoded = show_decoded_serration
        self.samp_rate = fs
        self.SysParams = sysparams
        self.fv = self.SysParams["FPS"] * 2
        self.fh = self.SysParams["FPS"] * self.SysParams["frame_lines"]
        self.venv_limit = 5
        self.serration_limit = 3
        iir_vsync_env = firdes_lowpass(self.samp_rate, self.fv * self.venv_limit, 1e3)
        self.vsyncEnvFilter = FiltersClass(
            iir_vsync_env[0], iir_vsync_env[1], self.samp_rate
        )

        iir_serration_base_lo = firdes_highpass(self.samp_rate, self.fh, self.fh)
        iir_serration_base_hi = firdes_lowpass(self.samp_rate, self.fh, self.fh)

        self.serrationFilter_base = {
            FiltersClass(
                iir_serration_base_lo[0], iir_serration_base_lo[1], self.samp_rate
            ),
            FiltersClass(
                iir_serration_base_hi[0], iir_serration_base_hi[1], self.samp_rate
            ),
        }

        iir_serration_envelope_lo = firdes_lowpass(
            self.samp_rate, self.fh / self.serration_limit, self.fh / 2
        )
        self.serrationFilter_envelope = FiltersClass(
            iir_serration_envelope_lo[0], iir_serration_envelope_lo[1], self.samp_rate
        )

        self.eq_pulselen = round(
            t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6))
        )
        self.vsynclen = round(t_to_samples(self.samp_rate, self.fv))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.pid = getpid()
        self.levels = StackableMA(min_watermark=2), StackableMA(min_watermark=2)  # sync, blanking
        self.sync_level_bias = np.array([])
        self.fieldcount = 0

    def get_levels(self):
        sync, blank = self.levels[0].pull(), self.levels[1].pull()
        return sync, blank

    def has_levels(self):
        return (
            self.levels[0].has_values()
            and self.levels[1].has_values()
        )

    def push_levels(self, levels):
        for ix, level in enumerate(levels):
            self.levels[ix].push(level)

    def mutemask(self, raw_locs, blocklen, pulselen):
        mask = np.zeros(blocklen)
        locs = raw_locs[np.where(raw_locs < blocklen - pulselen)[0]]
        for loc in locs:
            mask[loc : loc + pulselen] = [1] * pulselen
        return mask[:blocklen]

    def levels_minima(self, lo_part):
        where_min = argrelextrema(lo_part, np.less)[0]
        return np.mean(lo_part[where_min])

    def vsync_envelope_simple(self, data):
        hi_filtered = self.vsyncEnvFilter.filtfilt(data)
        cut_level = np.min(hi_filtered)
        lo_part = np.clip(data, a_min=0, a_max=cut_level)
        lo_level = self.levels_minima(lo_part)
        return hi_filtered, lo_level

    def vsync_envelope_double(self, data):
        forward = self.vsync_envelope_simple(data)
        reverse_t = self.vsync_envelope_simple(np.flip(data))
        reverse = np.flip(reverse_t[0]), reverse_t[1]
        half = int(len(data) / 2)
        # end of forward + beginning of reverse
        result = np.append(reverse[0][:half], forward[0][half:]), np.ones(len(data)) * forward[1]
        # dualplot_scope(forward[0], forward[1])
        # dualplot_scope(result[0], result[1], title="VBI envelope")
        return result

    def chainfiltfilt(self, data, filters):
        for filter in filters:
            data = filter.filtfilt(data)
        return data

    def power_ratio_search(self, data):
        first_harmonic = np.power(
            self.chainfiltfilt(data, self.serrationFilter_base), 2
        )
        first_harmonic = self.serrationFilter_envelope.filtfilt(first_harmonic)
        return argrelextrema(first_harmonic, np.less)[0]

    def select_serration(self, where_min, serrations):
        selected = np.array([], np.int)
        for id, edge in enumerate(serrations):
            for s_min in where_min:
                next_serration_id = min(id + 1, len(serrations) - 1)
                if edge <= s_min <= serrations[next_serration_id]:
                    selected = np.append(selected, edge)
        return selected

    def vsync_arbitrage(self, where_allmin, serrations, datalen):
        result = np.array([], np.int)
        if len(where_allmin) > 0:
            valid_serrations = self.select_serration(where_allmin, serrations)
            for serration in valid_serrations:
                if (
                    serration - self.vsynclen >= 0
                    or serration + self.vsynclen <= datalen - 1
                ):
                    result = np.append(result, serration)
        elif len(where_allmin) == 1:
            if where_allmin[0] + self.vsynclen < datalen - 1:
                result = np.append(where_allmin[0], where_allmin[0] + self.vsynclen)
            else:
                result = np.append(
                    where_allmin[0], max(where_allmin[0] - self.vsynclen, 0)
                )
        else:
            result = None

        return result

    def get_serration_sync_levels(self, serration):
        half_amp = np.mean(serration)
        peaks = np.where(serration > half_amp)[0]
        valleys = np.where(serration <= half_amp)[0]
        levels = np.median(serration[valleys]), np.median(serration[peaks])
        return levels

    def search_eq_pulses(self, data, pos, linespan=30):
        start, end = max(0, pos - self.linelen * linespan), min(
            len(data) - 1, pos + self.linelen * linespan
        )
        min_block = data[start:end]
        level = (np.median(min_block) - np.min(min_block)) / 2
        level += np.min(min_block)
        zero_block = min_block - level
        sync_pulses = zero_cross_det(zero_block)
        diff_sync = np.diff(sync_pulses)

        where_min_diff = np.where(
            np.logical_and(
                self.eq_pulselen * 0.2 < diff_sync, diff_sync < self.eq_pulselen * 5 / 4
            )
        )[0]

        if 9 <= len(where_min_diff) <= 12:
            eq_s, eq_e = sync_pulses[where_min_diff[0]], min(
                int(sync_pulses[where_min_diff[-1:][0]] + self.eq_pulselen / 2),
                len(data) - 1,
            )
            data_s, data_e = eq_s + start, eq_e + start
            serration = data[data_s:data_e]

            # validates it by time length, range must be calculated
            if 17e3 < len(serration) < 23e3:
                self.push_levels(self.get_serration_sync_levels(serration))
                if self.show_decoded:
                    sync, blank = self.get_levels()
                    marker = np.ones(len(serration)) * blank
                    dualplot_scope(
                        serration,
                        marker,
                        title="VBI EQ serration + measured blanking level",
                    )
                return True, data_s, data_e
            else:
                return False, None, None
        else:
            return False, None, None

    def mean_bias(self):
        return np.mean(self.sync_level_bias)

    def remove_bias(self, data):
        return data - self.sync_level_bias

    def vsync_envelope(self, data, padding=1024):  # 0x10000
        padded = np.append(np.flip(data[:padding]), data)
        forward = self.vsync_envelope_double(padded)
        self.sync_level_bias = forward[1][padding:]
        diff = np.add(forward[0][padding:], -self.sync_level_bias)
        where_allmin = argrelextrema(diff, np.less)[0]
        if len(where_allmin) > 0:
            serrations = self.power_ratio_search(padded)
            where_min = self.vsync_arbitrage(where_allmin, serrations, len(padded))
            serration_locs = list()
            if len(where_min) > 0:
                mask_len = self.linelen * 5
                state = False
                for w_min in where_min:
                    state, serr_loc, serr_len = self.search_eq_pulses(data, w_min)
                    if state:
                        serration_locs.append(serr_loc)
                        mask_len = serr_len - serr_loc

                if self.show_decoded:
                    data_copy = data.copy()  # self.remove_bias(data)
                    if len(serration_locs) > 0:
                        mask = self.mutemask(
                            np.array(serration_locs), len(data_copy), mask_len
                        )
                        dualplot_scope(
                            data_copy,
                            np.clip(
                                mask * max(data_copy),
                                a_max=max(data_copy),
                                a_min=min(data_copy),
                            ),
                            title="VBI position",
                        )
                    else:
                        plot_scope(data_copy, title="Missing serration measure")
                        ldd.logger.warning("A serration measure is missing")
                return state
            else:
                # dualplot_scope(forward[0], forward[1], title='unexpected arbitrage')
                ldd.logger.warning("Unexpected vsync arbitrage")
                return None
        else:
            # dualplot_scope(forward[0], forward[1], title='unexpected, there is no minima')
            ldd.logger.warning("Unexpected video envelope")
            return None

    def work(self, data):
        self.vsync_envelope(data)
        if self.has_levels():
            ldd.logger.debug(
                "VBI serration levels %d - Sync tip: %.02f kHz, Blanking (ire0): %.02f kHz"
                % (
                    self.levels[0].size(),
                    self.get_levels()[0] / 1e3,
                    self.get_levels()[1] / 1e3,
                )
            )
        elif self.fieldcount % 10 == 0:
            ldd.logger.debug(
                "VBI EQ serration pulses search failed (using fallback logic)"
            )

        self.fieldcount += 1

    # safe clips the bottom of the sync pulses, but not the picture area
    def safe_sync_clip(self, sync_ref, data):
        if self.has_levels():
            sync, blank = self.get_levels()
            mid_sync = (sync + blank) / 2
            where_all_picture = np.where(sync_ref > mid_sync)[0]
            locs_len = np.diff(where_all_picture)
            min_synclen = self.eq_pulselen * 3 / 4
            max_synclen = self.eq_pulselen * 3
            is_sync = np.bitwise_and(locs_len > min_synclen, locs_len < max_synclen)
            where_all_syncs = np.where(is_sync)[0]
            clip_from = where_all_picture[where_all_syncs]
            clip_len = locs_len[where_all_syncs]
            for ix, begin in enumerate(clip_from):
                data[begin : begin + clip_len[ix]] = sync

        return data
