from vhsdecode.addons.wavesink import WaveSink
from vhsdecode.utils import FiltersClass, firdes_lowpass, plot_scope, dualplot_scope, filter_plot, zero_cross_det, \
    pad_or_truncate
import numpy as np
from scipy.signal import argrelextrema


def identity(value):
    return value


def t_to_samples(samp_rate, value):
    return samp_rate / value


class DCrestore:

    def __init__(self, fs, sysparams, blocklen, scale=identity):
        self.samp_rate = fs
        self.SysParams = sysparams
        self.blocklen = blocklen
        self.scale = scale
        self.fv = self.SysParams["FPS"] * 2
        self.fh = self.SysParams["FPS"] * self.SysParams["frame_lines"]
        self.harmonic_limit = 3

        iir_hsync = firdes_lowpass(self.samp_rate, self.fh * self.harmonic_limit, 110e3, 3)
        # filter_plot(iir_hsync[0], iir_hsync[1], self.samp_rate, type='lowpass', title='Hsync')
        self.hsyncFilter = FiltersClass(iir_hsync[0], iir_hsync[1], self.samp_rate)

        iir_vsync = firdes_lowpass(self.samp_rate, self.fv * self.harmonic_limit, 220e3, 2)
        self.vsyncFilter = FiltersClass(iir_vsync[0], iir_vsync[1], self.samp_rate)
        # filter_plot(iir_vsync[0], iir_vsync[1], self.samp_rate, type='lowpass', title='Vsync')

        self.eq_pulselen = round(t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6)))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.sink = WaveSink(self.fh, 44100)
        self.max_hpulses_per_block = np.round(self.blocklen / self.linelen)
        self.min_hpulses_per_block = self.max_hpulses_per_block - 1
        self.hsync_level_delay = int(self.eq_pulselen * 3 / 2)
        self.blank_level_delay = int(self.eq_pulselen * 7 / 2)
        self.integration_constant = 10
        self.levels = None, None
        self.hpos = np.array([])
        self.bias = self.scale(self.SysParams['vsync_ire'])

    # to get the real min extrema
    def filter_real_min_extrema(self, data, extremas):
        data_min = np.min(data)
        data_max = np.max(data)
        span = data_max - data_min
        threshold = data_min + span / 4
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
    def get_mutemask(self, start_loc):
        zeros = np.zeros(self.eq_pulselen)
        ones = np.ones(self.linelen - self.eq_pulselen)
        mask = np.array([])
        linemask = np.roll(np.append(zeros, ones), start_loc)
        while len(mask) < self.blocklen:
            mask = np.append(mask, linemask)
        return mask[:self.blocklen]

    # mask that adds in the signal muting interval (sample & hold)
    def get_addmask(self, start_loc, hlevels):
        inv_mask = np.add(1, -self.get_mutemask(start_loc))
        mask = inv_mask[:start_loc] * hlevels[0]
        init_id = start_loc
        for level in hlevels:
            start = init_id
            end = min(init_id + self.linelen, self.blocklen)
            mask = np.append(mask, inv_mask[start:end] * level)
            init_id += self.linelen

        return mask[:self.blocklen]

    # return sampling space range
    def get_sampling_pos(self, start_loc):
        hspace = range(start_loc, self.blocklen, self.linelen)
        return np.array(hspace)

    # writes the wav file
    def sink_levels(self, hlevels, blevels):
        attenuation = 2e6
        hlevel_bias, blevel_bias = np.mean(hlevels), np.mean(blevels)
        ch0, ch1 = (hlevels - hlevel_bias) / attenuation, (blevels - blevel_bias) / attenuation
        self.sink.write(ch0, ch1)

    # returns levels at sampling pos in another array
    def get_levels(self, data, sampling_pos):
        levels = np.array([])
        for pos in sampling_pos:
            start = pos
            end = min(pos + self.integration_constant, self.blocklen)
            levels = np.append(levels, np.mean(data[start:end]))

        return levels

    # computes start hsync location and sync levels with blanking levels
    def get_syncblank(self, data, start_points):
        start_pos = self.sync_lock_filter(start_points)  # self.get_startloc(start_points) % self.linelen
        hsampling_pos = self.hsync_level_delay + start_pos
        bsampling_pos = self.blank_level_delay + start_pos
        #hsampling_pos = self.get_sampling_pos(hstart_pos)
        #bsampling_pos = self.get_sampling_pos(bstart_pos)
        hlevels = self.get_levels(data, hsampling_pos)
        blevels = self.get_levels(data, bsampling_pos)

        if len(blevels) < len(hlevels):
            blevels = np.append(blevels, blevels[-1:])

        return hlevels, blevels, hsampling_pos

    # compensate sync levels (move it to self.bias) in data based on locs and levels
    def compensate_syncs(self, data, locs, hlevels):
        assert len(locs) == len(hlevels), "hsync locs should be the same length as level measures"
        # head
        result = np.add(data[:locs[0]], -hlevels[0] + self.bias)
        # body
        for id, pos in enumerate(locs):
            end = min(pos + self.linelen, self.blocklen)
            result = np.append(
                result, np.add(data[pos:end], -hlevels[id] + self.bias)
            )
        # tail
        last_level = locs[-1]
        data_copy = np.add(data, -last_level + self.bias)
        result = pad_or_truncate(result, data_copy)

        assert len(data) <= len(result), "len data: %d, len result: %d: %s" % (len(data), len(result), np.diff(locs))

        return result[:self.blocklen]

    # main
    def get_syncedgelocs(self, data):
        hsync = self.hsyncFilter.filtfilt(data)
        hsync_diff = np.diff(hsync)
        all_points = argrelextrema(hsync_diff, np.less)[0]
        start_points = self.filter_real_min_extrema(hsync_diff, all_points)

        return self.get_syncblank(data, start_points)

    # compensates sync on data after work()
    def compensate_sync(self, data):
        if self.min_hpulses_per_block <= len(self.hpos) <= self.max_hpulses_per_block:
            # mute_mask = self.get_mutemask(self.hpos[0])
            # add_mask = self.get_addmask(self.hpos[0], self.levels[0])
            # resync = np.add(np.multiply(data, mute_mask), add_mask)
            comp = self.compensate_syncs(data, self.hpos, self.levels[0])
            assert len(comp) == len(data), "processing mismatch, %d != %d" % (len(comp), len(data))
            #dualplot_scope(data, comp)
            print('COMPENSATED!')
            return comp
        else:
            print('NON COMPENSATED!')
            return data

    # computes the internal state
    def work(self, data):
        hlevels, blevels, hsampling_pos = self.get_syncedgelocs(data)
        if len(hsampling_pos) > 0:
            self.levels = hlevels, blevels
            self.hpos = hsampling_pos
            self.sink_levels(hlevels, blevels)

        # print(mask)
        # plot_scope(mask)
        # dualplot_scope(data[:self.linelen], mask[:self.linelen])
