from vhsdecode.addons.wavesink import WaveSink
from vhsdecode.utils import FiltersClass, firdes_lowpass, firdes_highpass, firdes_bandpass, plot_scope, dualplot_scope, filter_plot, zero_cross_det, \
    pad_or_truncate, fft_plot
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
        self.venv_limit = 3
        self.serration_limit = 3
        iir_vsync_env = firdes_lowpass(self.samp_rate, self.fv * self.venv_limit, 1e3)
        self.vsyncEnvFilter = FiltersClass(iir_vsync_env[0], iir_vsync_env[1], self.samp_rate)

        iir_serration_base_lo = firdes_highpass(self.samp_rate, self.fh, self.fh)
        iir_serration_base_hi = firdes_lowpass(self.samp_rate, self.fh, self.fh)

        self.serrationFilter_base = {
            FiltersClass(iir_serration_base_lo[0], iir_serration_base_lo[1], self.samp_rate),
            FiltersClass(iir_serration_base_hi[0], iir_serration_base_hi[1], self.samp_rate),
        }

        iir_serration_second_lo = firdes_highpass(self.samp_rate, self.fh * 2, self.fh * 2)
        iir_serration_second_hi = firdes_lowpass(self.samp_rate, self.fh * 2, self.fh * 2)

        self.serrationFilter_second = {
            FiltersClass(iir_serration_second_lo[0], iir_serration_second_lo[1], self.samp_rate),
            FiltersClass(iir_serration_second_hi[0], iir_serration_second_hi[1], self.samp_rate),
        }

        iir_serration_envelope_lo = firdes_lowpass(self.samp_rate, self.fh / self.serration_limit, self.fh / 2)
        self.serrationFilter_envelope = FiltersClass(iir_serration_envelope_lo[0], iir_serration_envelope_lo[1], self.samp_rate)

        self.eq_pulselen = round(t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6)))
        self.vsynclen = round(t_to_samples(self.samp_rate, self.fv))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.pid = getpid()
        self.sink = WaveSink(self.samp_rate)
        self.sink.set_scale(1)
        self.sink.set_offset(0)
        self.sink.set_name('vsyncdata.wav')
        self.levels = None, None

    # return true if lo_bound < value < hi_bound
    def hysteresis_checker(self, value, ref, error=0.1):
        lo_bound = (1 - error) * ref
        hi_bound = (1 + error) * ref
        if lo_bound < value < hi_bound:
            return True
        else:
            return False

    def mutemask(self, raw_locs, blocklen, pulselen):
        mask = np.zeros(blocklen)
        locs = raw_locs[np.where(raw_locs < blocklen - pulselen)[0]]
        for loc in locs:
            mask[loc:loc+pulselen] = [1] * pulselen
        return mask[:blocklen]

    # writes the wav file
    def sink_levels(self, hlevels, blevels):
        attenuation = 2e6
        hlevel_bias, blevel_bias = np.mean(hlevels), np.mean(blevels)
        ch0, ch1 = (hlevels - hlevel_bias) / attenuation, (blevels - blevel_bias) / attenuation
        self.sink.write(ch0, ch1)

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
        #dualplot_scope(result[0], result[1])
        return result

    def fft_power_ratio(self, data, f):
        fft = np.fft.fft(data)
        power = np.abs(fft) ** 2
        sample_freq = np.fft.fftfreq(len(data), d=1.0 / self.samp_rate)
        step = sample_freq[1] - sample_freq[0]
        main = int(f / step)
        half = int(f / (2 * step))
        return power[main] / power[half]

    def chunks(self, l, n):
        n = max(1, n)
        split = [l[x:x + n] for x in range(0, len(l), n)]
        return split

    def is_valid_serration(self, chunk):
        half_amp = (np.max(chunk) - np.min(chunk)) / 2
        half_level = half_amp + np.min(chunk)
        dc_chunk = chunk - half_level
        crosses = zero_cross_det(dc_chunk)
        return 6 <= len(crosses) <= 7

    def fft_power_bin_search(self, data, f, bins=16):
        data_chunks = self.chunks(data, int(len(data) / bins))
        bin_power = list()

        for chunk in data_chunks:
            if len(chunk) == int(len(data) / bins):
                bin_power.append(self.fft_power_ratio(chunk, f))

        peak_id = int(np.where(bin_power == max(bin_power))[0])
        if max(bin_power) > 100 and self.is_valid_serration(data_chunks[peak_id]):
            plot_scope(data_chunks[peak_id], title='serration chunk')
            print(max(bin_power), min(bin_power))
            return peak_id
        else:
            print('Missing video serration')
            #print(max(bin_power))
            #plot_scope(data_chunks[peak_id], title='missing chunk')
            return None

    def chainfiltfilt(self, data, filters):
        for filter in filters:
            data = filter.filtfilt(data)
        return data

    def power_ratio_search(self, data):
        first_harmonic = np.power(self.chainfiltfilt(data, self.serrationFilter_base), 2)
        first_harmonic = self.serrationFilter_envelope.filtfilt(first_harmonic)
        return argrelextrema(first_harmonic, np.less)[0]

    def window_plot(self, data, where_min, window=2048):
        for n, point in enumerate(where_min):
            start = max(0, int(point - window / 2))
            end = min(len(data), int(point + window / 2))
            plot_scope(data[start:end], title='Zoom point %d/%d' % (n, len(where_min)))
            #dc_rem = data[start:end] - np.mean(data[start:end])
            #self.fft_power_bin_search(dc_rem, self.fh * 2)

            #fft = np.fft.fft(data[start:end])
            #freq = np.fft.fftfreq(len(data), data[1] - data[0])

    def vsync_arbitrage(self, where_allmin):
        meas_pulse_width = self.samp_rate
        cut = 0
        if len(where_allmin) > 1:
            while self.hysteresis_checker(meas_pulse_width, self.vsynclen, error=0.05):
                meas_pulse_width = where_allmin[len(where_allmin)-cut-1] - where_allmin[0]
                cut += 1
            result = np.append(where_allmin[0], where_allmin[len(where_allmin)-cut-1])
        else:
            result = np.append(where_allmin[0], where_allmin[0] + self.vsynclen)

        return result

    def select_serration(self, where_min, serrations):
        selected = np.array([], np.int)
        for id, edge in enumerate(serrations):
            for s_min in where_min:
                next_serration_id = min(id + 1, len(serrations) -1)
                if edge <= s_min <= serrations[next_serration_id]:
                    selected = np.append(selected, edge)
        return selected

    def vsync_arbitrage2(self, where_allmin, serrations, datalen):
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
            self.levels = self.get_serration_sync_levels(serration)
            #marker = np.ones(len(serration)) * self.levels[1]
            #dualplot_scope(serration, marker, title='serration')
            return True, data_s, data_e
        else:
            if self.levels == (None, None):
                print('VBI EQ pulses search failed', self.levels)
            return False, None, None


    #from where_min search for the min level and repeat
    def vsync_envelope(self, data, padding=1024):  # 0x10000
        padded = np.append(data[:padding], data)
        forward = self.vsync_envelope_double(padded)
        diff = np.add(forward[0][padding:], forward[1][padding:])
        where_allmin = argrelextrema(diff, np.less)[0]
        if len(where_allmin) > 0:
            serrations = self.power_ratio_search(padded)
            where_min = self.vsync_arbitrage2(where_allmin, serrations, len(padded))
            if len(where_min) > 0:
                mask = self.mutemask(where_min, len(data), self.linelen * 5)
                dualplot_scope(data, np.clip(mask * max(data), a_max=max(data), a_min=min(data)))
                #self.window_plot(data, where_min, window=self.linelen * 25)
                for w_min in where_min:
                    self.search_eq_pulses(data, w_min)
                return mask
            else:
                #dualplot_scope(forward[0], forward[1], title='unexpected')
                return None
        else:
            #dualplot_scope(forward[0], forward[1], title='unexpected')
            return None

    # main
    def get_syncedgelocs(self, data):
        mask = self.vsync_envelope(data)

        if mask is not None:
            self.sink.write(mask / 2, mask / 2)

        return np.array([]), np.array([]), np.array([])

    # computes the internal state
    def work(self, data):
        hlevels, blevels, hsampling_pos = self.get_syncedgelocs(data)
