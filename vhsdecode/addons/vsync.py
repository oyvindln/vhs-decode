from vhsdecode.utils import FiltersClass, firdes_lowpass, firdes_highpass, firdes_bandpass, plot_scope, dualplot_scope, filter_plot, zero_cross_det, \
    pad_or_truncate, fft_plot, moving_average
import numpy as np
from scipy.signal import argrelextrema
from os import getpid


def t_to_samples(samp_rate, value):
    return samp_rate / value


class Vsync:

    def __init__(self, fs, sysparams):
        self.samp_rate = fs
        self.SysParams = sysparams
        self.fv = self.SysParams["FPS"] * 2
        self.fh = self.SysParams["FPS"] * self.SysParams["frame_lines"]
        self.venv_limit = 5
        self.serration_limit = 3
        iir_vsync_env = firdes_lowpass(self.samp_rate, self.fv * self.venv_limit, 1e3)
        self.vsyncEnvFilter = FiltersClass(iir_vsync_env[0], iir_vsync_env[1], self.samp_rate)

        iir_serration_base_lo = firdes_highpass(self.samp_rate, self.fh, self.fh)
        iir_serration_base_hi = firdes_lowpass(self.samp_rate, self.fh, self.fh)

        self.serrationFilter_base = {
            FiltersClass(iir_serration_base_lo[0], iir_serration_base_lo[1], self.samp_rate),
            FiltersClass(iir_serration_base_hi[0], iir_serration_base_hi[1], self.samp_rate),
        }

        iir_serration_envelope_lo = firdes_lowpass(self.samp_rate, self.fh / self.serration_limit, self.fh / 2)
        self.serrationFilter_envelope = FiltersClass(iir_serration_envelope_lo[0], iir_serration_envelope_lo[1], self.samp_rate)

        self.eq_pulselen = round(t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6)))
        self.vsynclen = round(t_to_samples(self.samp_rate, self.fv))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.pid = getpid()
        self.levels = list(), list() # sync / blanking
        self.level_average = 30
        self.sync_level_bias = np.array([])
        self.fieldcount = 0

    def get_levels(self):
        sync, sync_list = moving_average(self.levels[0], window=self.level_average)
        blank, blank_list = moving_average(self.levels[1], window=self.level_average)
        self.levels = sync_list, blank_list
        return sync, blank

    def has_levels(self):
        return len(self.levels[0]) > 2 and len(self.levels[1]) > 2

    def push_levels(self, levels):
        for ix, level in enumerate(levels):
            self.levels[ix].append(level)

    def mutemask(self, raw_locs, blocklen, pulselen):
        mask = np.zeros(blocklen)
        locs = raw_locs[np.where(raw_locs < blocklen - pulselen)[0]]
        for loc in locs:
            mask[loc:loc+pulselen] = [1] * pulselen
        return mask[:blocklen]

    def vsync_envelope_simple(self, data):
        hi_part = np.clip(data, a_max=np.max(data), a_min=0)
        inv_data = np.multiply(data, -1)
        lo_part_inv = np.clip(inv_data, a_max=np.max(inv_data), a_min=0)
        lo_part = np.multiply(lo_part_inv, -1)
        hi_filtered = self.vsyncEnvFilter.filtfilt(hi_part)
        lo_filtered = self.vsyncEnvFilter.filtfilt(lo_part)
        return hi_filtered, lo_filtered

    def vsync_envelope_double(self, data):
        forward = self.vsync_envelope_simple(data)
        reverse_t = self.vsync_envelope_simple(np.flip(data))
        reverse = np.flip(reverse_t[0]), np.flip(reverse_t[1])
        half = int(len(data) / 2)
        # end of forward + beginning of reverse
        result = np.append(reverse[0][:half], forward[0][half:]), np.append(reverse[1][:half], forward[1][half:])
        #dualplot_scope(forward[0], forward[1])
        #dualplot_scope(result[0], result[1], title="VBI envelope")
        return result

    def chainfiltfilt(self, data, filters):
        for filter in filters:
            data = filter.filtfilt(data)
        return data

    def power_ratio_search(self, data):
        first_harmonic = np.power(self.chainfiltfilt(data, self.serrationFilter_base), 2)
        first_harmonic = self.serrationFilter_envelope.filtfilt(first_harmonic)
        return argrelextrema(first_harmonic, np.less)[0]

    def select_serration(self, where_min, serrations):
        selected = np.array([], np.int)
        for id, edge in enumerate(serrations):
            for s_min in where_min:
                next_serration_id = min(id + 1, len(serrations) -1)
                if edge <= s_min <= serrations[next_serration_id]:
                    selected = np.append(selected, edge)
        return selected

    def vsync_arbitrage(self, where_allmin, serrations, datalen):
        result = np.array([], np.int)
        if len(where_allmin) > 0:
            valid_serrations = self.select_serration(where_allmin, serrations)
            for serration in valid_serrations:
                if serration - self.vsynclen >= 0 or serration + self.vsynclen <= datalen -1:
                    result = np.append(result, serration)
        elif len(where_allmin) == 1:
            if where_allmin[0] + self.vsynclen < datalen - 1:
                result = np.append(where_allmin[0], where_allmin[0] + self.vsynclen)
            else:
                result = np.append(where_allmin[0], max(where_allmin[0] - self.vsynclen, 0))
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
        start, end = max(0, pos - self.linelen * linespan), min(len(data) -1, pos + self.linelen * linespan)
        min_block = data[start:end]
        level = (np.median(min_block) - np.min(min_block)) / 2
        level += np.min(min_block)
        zero_block = min_block - level
        sync_pulses = zero_cross_det(zero_block)
        diff_sync = np.diff(sync_pulses)

        where_min_diff = np.where(np.logical_and(self.eq_pulselen * 0.2 < diff_sync, diff_sync <= self.eq_pulselen))[0]
        if 9 <= len(where_min_diff) <= 12:
            eq_s, eq_e = sync_pulses[where_min_diff[0]], \
                         min(int(sync_pulses[where_min_diff[-1:][0]] + self.eq_pulselen / 2), len(data) - 1)
            data_s, data_e = eq_s + start, eq_e + start
            serration = data[data_s:data_e]
            if 17e3 < len(serration) < 23e3:
                self.push_levels(self.get_serration_sync_levels(serration))
                #sync, blank = self.get_levels()
                #marker = np.ones(len(serration)) * blank
                #dualplot_scope(serration, marker, title='VBI EQ serration + measured blanking level')
                return True, data_s, data_e
            else:
                return False, None, None
        else:
            if not self.has_levels() and self.fieldcount % 10 == 0:
                print('VBI EQ pulses search failed')
            return False, None, None

    def vsync_envelope(self, data, padding=1024):  # 0x10000
        padded = np.append(np.flip(data[:padding]), data)
        forward = self.vsync_envelope_double(padded)
        self.sync_level_bias = forward[1][padding:]
        diff = np.add(forward[0][padding:], forward[1][padding:])
        where_allmin = argrelextrema(diff, np.less)[0]
        if len(where_allmin) > 0:
            serrations = self.power_ratio_search(padded)
            where_min = self.vsync_arbitrage(where_allmin, serrations, len(padded))
            serration_locs = list()
            if len(where_min) > 0:
                mask_len = self.linelen * 5
                for w_min in where_min:
                    state, serr_loc, serr_len = self.search_eq_pulses(data, w_min)
                    if state:
                        serration_locs.append(serr_loc)
                        mask_len = serr_len - serr_loc

                #data_copy = data - self.sync_level_bias
                #if len(serration_locs) > 0:
                    # mask = self.mutemask(np.array(serration_locs), len(data_copy), mask_len)
                    # dualplot_scope(data_copy, np.clip(mask * max(data_copy), a_max=max(data_copy), a_min=min(data_copy)), title="VBI position")
                #else:
                    # plot_scope(data_copy, title="Missing serration measure")
                    #print('A serration measure is missing')
                return None
            else:
                #dualplot_scope(forward[0], forward[1], title='unexpected')
                return None
        else:
            #dualplot_scope(forward[0], forward[1], title='unexpected')
            return None

    def get_syncedgelocs(self, data):
        self.vsync_envelope(data)
        if self.has_levels():
            print('levels:', len(self.levels[0]), self.get_levels())

        return np.array([]), np.array([]), np.array([])

    # computes the internal levels
    def work(self, data):
        hlevels, blevels, hsampling_pos = self.get_syncedgelocs(data)
        self.fieldcount += 1