from vhsdecode.addons.wavesink import WaveSink
from vhsdecode.utils import FiltersClass, firdes_lowpass, firdes_highpass, firdes_bandpass, plot_scope, dualplot_scope, filter_plot, zero_cross_det, \
    pad_or_truncate, fft_plot
import numpy as np
from scipy.signal import argrelextrema
from os import getpid


def identity(value):
    return value


def t_to_samples(samp_rate, value):
    return samp_rate / value


class DCrestore:

    def __init__(self, fs, sysparams, scale=identity):
        self.samp_rate = fs
        self.SysParams = sysparams
        self.scale = scale
        self.fv = self.SysParams["FPS"] * 2
        self.fh = self.SysParams["FPS"] * self.SysParams["frame_lines"]
        self.h_limit = 2
        self.v_limit = 3
        self.venv_limit = 3

        iir_hsync = firdes_bandpass(self.samp_rate, self.fh, 1e3, self.fh * self.h_limit, 1e3)
        # filter_plot(iir_hsync[0], iir_hsync[1], self.samp_rate, type='lowpass', title='Hsync')
        self.hsyncFilter = FiltersClass(iir_hsync[0], iir_hsync[1], self.samp_rate)

        iir_vsync_lo = firdes_lowpass(self.samp_rate, self.fv * self.v_limit, 1e3)
        iir_vsync_hi = firdes_highpass(self.samp_rate, self.fv, 1e3)
        iir_vsync_env = firdes_lowpass(self.samp_rate, self.fv * self.venv_limit, 1e3)


        self.vsyncFilter = {
            FiltersClass(iir_vsync_lo[0], iir_vsync_lo[1], self.samp_rate),
            FiltersClass(iir_vsync_hi[0], iir_vsync_hi[1], self.samp_rate)
        }

        self.vsyncEnvFilter = FiltersClass(iir_vsync_env[0], iir_vsync_env[1], self.samp_rate)

        self.eq_pulselen = round(t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6)))
        self.vsynclen = round(t_to_samples(self.samp_rate, self.fv))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.sink = WaveSink(self.samp_rate)
        self.hsync_level_delay = int(self.eq_pulselen * 3 / 2)
        self.blank_level_delay = int(self.eq_pulselen * 7 / 2)
        self.integration_constant = 10
        self.levels = None, None
        self.hpos = np.array([])
        self.bias = self.scale(self.SysParams['vsync_ire'])
        self.pid = getpid()
        print('pid', self.pid)
        self.sink.set_scale(1)
        self.sink.set_offset(0)
        self.field_buffer = np.array([])

    def max_hpulses_per_block(self, blocklen):
        return np.round(blocklen / self.linelen)

    def min_hpulses_per_block(self, blocklen):
        return self.max_hpulses_per_block(blocklen) - 1

    # to get the real min extrema
    def filter_real_min_extrema(self, data, extremas):
        data_min = np.mean(data[extremas])
        data_max = 0
        span = np.abs(data_max - data_min)
        threshold = data_min + span / 2
        real_mins = list()
        for extrema in extremas:
            if data[extrema] < threshold:
                real_mins.append(extrema)
        return np.asarray(real_mins)

    # linear regression thing
    def get_trendloc(self, start_points):
        x = range(0, len(start_points))
        try:
            coef = np.polyfit(x, start_points, 1)
            poly1d_fn = np.poly1d(coef)
            return poly1d_fn
        except (TypeError, ValueError):
            return None

    # return true if lo_bound < value < hi_bound
    def hysteresis_checker(self, value, ref, error=0.1):
        lo_bound = (1 - error) * ref
        hi_bound = (1 + error) * ref
        if lo_bound < value < hi_bound:
            return True
        else:
            return False

    # this returns where is the first 'valid' sync
    def sync_lock_start(self, sync):
        sync_diff = np.diff(sync)
        for ix, diff in enumerate(sync_diff):
            if self.hysteresis_checker(diff, self.linelen):
                return ix
        return 0

    # this returns the data points where the start of the sync should be
    def sync_lock_filter(self, sync):
        assert len(sync) > 0, "got empty sync detection %s" % sync
        sync_diff = np.diff(sync)
        fsync = np.array([])
        plot_scope(sync, title='sync index')
        for ix, diff in enumerate(sync_diff):
            if self.hysteresis_checker(diff, self.linelen):
                fsync = np.append(fsync, sync[ix])
        #print('sync edges:', sync_diff)
        #print('after hyst:', np.diff(fsync), len(fsync))
        #plot_scope(sync, title='sync indexes')
        return np.array(fsync, dtype=np.int)

    # computes the sync trend
    def get_trend(self, start_points):
        x = range(0, len(start_points))
        return self.get_trendloc(start_points)(x)

    # this produces 1 where the signal will be kept, and 0 where it need to be muted
    def get_mutemask(self, start_loc, blocklen):
        zeros = np.zeros(self.eq_pulselen)
        ones = np.ones(self.linelen - self.eq_pulselen)
        mask = np.array([])
        linemask = np.roll(np.append(zeros, ones), start_loc)
        while len(mask) < blocklen:
            mask = np.append(mask, linemask)
        return mask[:blocklen]

    def mutemask(self, raw_locs, blocklen, pulselen):
        mask = np.zeros(blocklen)
        locs = raw_locs[np.where(raw_locs < blocklen - pulselen)[0]]
        for loc in locs:
            mask[loc:loc+pulselen] = [1] * pulselen
        return mask[:blocklen]

    # mask that adds in the signal muting interval (sample & hold)
    def get_addmask(self, start_loc, hlevels, blocklen):
        inv_mask = np.add(1, -self.get_mutemask(start_loc))
        mask = inv_mask[:start_loc] * hlevels[0]
        init_id = start_loc
        for level in hlevels:
            start = init_id
            end = min(init_id + self.linelen, blocklen)
            mask = np.append(mask, inv_mask[start:end] * level)
            init_id += self.linelen

        return mask[:blocklen]

    # return sampling space range
    def get_sampling_pos(self, start_loc, blocklen):
        hspace = range(start_loc, blocklen, self.linelen)
        return np.array(hspace)

    # writes the wav file
    def sink_levels(self, hlevels, blevels):
        attenuation = 2e6
        hlevel_bias, blevel_bias = np.mean(hlevels), np.mean(blevels)
        ch0, ch1 = (hlevels - hlevel_bias) / attenuation, (blevels - blevel_bias) / attenuation
        self.sink.write(ch0, ch1)

    # returns levels at sampling pos in another array
    def get_levels(self, data, sampling_pos, blocklen):
        levels = np.array([])
        for pos in sampling_pos:
            start = pos
            end = min(pos + self.integration_constant, blocklen)
            levels = np.append(levels, np.mean(data[start:end]))

        return levels

    # computes start hsync location and sync levels with blanking levels
    def get_syncblank(self, data, start_points):
        start_pos = start_points #self.sync_lock_filter(start_points)  # self.get_startloc(start_points) % self.linelen
        hsampling_pos = self.hsync_level_delay + start_pos
        bsampling_pos = np.where(self.blank_level_delay + start_pos < len(data))[0]
        hlevels = data[hsampling_pos] #self.get_levels(data, hsampling_pos, len(data))
        blevels = data[bsampling_pos] #self.get_levels(data, bsampling_pos, len(data))

        if len(blevels) < len(hlevels):
            blevels = np.append(blevels, blevels[-1:])

        #dualplot_scope(hlevels, blevels)
        return hlevels, blevels, hsampling_pos

    # compensate sync levels (move it to self.bias) in data based on locs and levels
    def compensate_syncs(self, data, locs, hlevels, blocklen):
        assert len(locs) == len(hlevels), "hsync locs should be the same length as level measures"
        # head
        result = np.add(data[:locs[0]], -hlevels[0] + self.bias)
        # body
        for id, pos in enumerate(locs):
            end = min(pos + self.linelen, blocklen)
            result = np.append(
                result, np.add(data[pos:end], -hlevels[id] + self.bias)
            )
        # tail
        last_level = locs[-1]
        data_copy = np.add(data, -last_level + self.bias)
        result = pad_or_truncate(result, data_copy)

        assert len(data) <= len(result), "len data: %d, len result: %d: %s" % (len(data), len(result), np.diff(locs))

        return result[:blocklen]

    def chainfiltfilt(self, data, filters):
        for filter in filters:
            data = filter.filtfilt(data)
        return data

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

    def fft_power_bin_search(self, data, f, bins=16):
        data_chunks = self.chunks(data, int(len(data) / bins))
        bin_power = list()
        for chunk in data_chunks:
            if len(chunk) == int(len(data) / bins):
                bin_power.append(self.fft_power_ratio(chunk, f))
        peak_id = int(np.where(bin_power == max(bin_power))[0])
        plot_scope(data_chunks[peak_id], title='peak chunk')
        return np.array(bin_power)

    def window_plot(self, data, where_min, window=2048):
        for n, point in enumerate(where_min):
            start = max(0, int(point - window / 2))
            end = min(len(data), int(point + window / 2))
            #plot_scope(data[start:end], title='Zoom point %d/%d' % (n, len(where_min)))
            dc_rem = data[start:end] - np.mean(data[start:end])
            print(self.fft_power_bin_search(dc_rem, self.fh * 2))

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

    #from where_min search for the min level and repeat
    def vsync_envelope(self, data, padding=1024):  # 0x10000
        padded = np.append(data[:padding], data)
        forward = self.vsync_envelope_double(padded)
        diff = np.add(forward[0][padding:], forward[1][padding:])
        where_allmin = argrelextrema(diff, np.less)[0]

        if len(where_allmin) > 0:
            where_min = self.vsync_arbitrage(where_allmin)
            mask = self.mutemask(where_min, len(data), self.linelen * 5)
            dualplot_scope(data, mask * max(data))
            self.window_plot(data, where_min, window=self.linelen * 25)
            #print(np.diff(where_min))
            return mask
        else:
            dualplot_scope(forward[0], forward[1], title='unexpected')
            return None

    # main
    def get_syncedgelocs(self, data):
        mask = self.vsync_envelope(data)
        expanded_data = data

        hsync = self.hsyncFilter.filtfilt(expanded_data)
        hsync_diff = np.sign(np.diff(hsync))
        all_hpoints = zero_cross_det(hsync_diff) #argrelextrema(hsync_diff, np.less)[0]
        #print(np.diff(zero_cross_det(hsync_diff)))
        start_hpoints = all_hpoints #self.filter_real_min_extrema(hsync_diff, all_hpoints)

        vsync = self.chainfiltfilt(expanded_data, self.vsyncFilter)
        vsync_diff = np.sign(np.diff(vsync))
        all_vpoints = zero_cross_det(vsync_diff) #argrelextrema(vsync_diff, np.less)[0]
        start_vpoints = all_vpoints #self.filter_real_min_extrema(vsync_diff, all_vpoints)

        print(len(start_hpoints), len(start_vpoints))
        #dualplot_scope(data[:self.linelen*4], hsync[:self.linelen*4])

        #vmask = self.mutemask(start_vpoints, len(data), self.linelen * 4)
        #hmask = self.mutemask(start_hpoints, len(data), self.eq_pulselen * 2)

        #dualplot_scope(expanded_data, vsync_diff * max(expanded_data))

        #assert len(mask) == len(vsync_diff), "sink sequence mismatch %d != %d" % (len(mask), len(vsync_diff))
        if mask is not None:
            self.sink.write(mask / 2, mask / 2)

        return np.array([]), np.array([]), np.array([]) #self.get_syncblank(data, start_hpoints)

    # compensates sync on data after work()
    def compensate_sync(self, data):
        blocklen = len(data)
        #print(self.hpos)
        if self.min_hpulses_per_block(len(data)) <= len(self.hpos) <= self.max_hpulses_per_block(len(data)):
            mute_mask = self.get_mutemask(self.hpos[0], blocklen)
            add_mask = self.get_addmask(self.hpos[0], self.levels[0], blocklen)
            # resync = np.add(np.multiply(data, mute_mask), add_mask)
            comp = self.compensate_syncs(data, self.hpos, self.levels[0], blocklen)
            assert len(comp) == blocklen, "processing mismatch, %d != %d" % (len(comp), blocklen)
            #dualplot_scope(data, comp)
            #print('COMPENSATED!')
            return mute_mask
        else:
            #print('NON COMPENSATED!')
            return data

    # computes the internal state
    def work(self, data):
        hlevels, blevels, hsampling_pos = self.get_syncedgelocs(data)
        if len(hsampling_pos) > 0:
            self.levels = hlevels, blevels
            self.hpos = hsampling_pos
            #self.sink_levels(hlevels, blevels)

        self.field_buffer = data
        # print(mask)
        # plot_scope(mask)
        # dualplot_scope(data[:self.linelen], mask[:self.linelen])
