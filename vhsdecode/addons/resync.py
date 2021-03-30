from vhsdecode.addons.wavesink import WaveSink
from vhsdecode.utils import FiltersClass, firdes_lowpass, plot_scope, dualplot_scope, filter_plot, zero_cross_det, pad_or_truncate
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
        #filter_plot(iir_hsync[0], iir_hsync[1], self.samp_rate, type='lowpass', title='Hsync')
        self.hsyncFilter = FiltersClass(iir_hsync[0], iir_hsync[1], self.samp_rate)

        iir_vsync = firdes_lowpass(self.samp_rate, self.fv * self.harmonic_limit, 220e3, 2)
        self.vsyncFilter = FiltersClass(iir_vsync[0], iir_vsync[1], self.samp_rate)
        #filter_plot(iir_vsync[0], iir_vsync[1], self.samp_rate, type='lowpass', title='Vsync')

        self.eq_pulselen = round(t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6)))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.sink = WaveSink(self.fh, 44100)
        self.max_hpulses_per_block = np.round(self.blocklen / self.linelen)
        self.min_hpulses_per_block = self.max_hpulses_per_block - 1
        self.hsync_level_delay = int(self.eq_pulselen * 3 / 2)
        self.blank_level_delay = int(self.eq_pulselen * 7 / 2)
        self.integration_constant = 10
        self.levels = None, None
        self.hpos = 0


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

    def get_trendloc(self, start_points):
        x = range(0, len(start_points))
        try:
            coef = np.polyfit(x, start_points, 1)
            poly1d_fn = np.poly1d(coef)
            return poly1d_fn
        except (TypeError, ValueError):
            return None

    def get_startloc(self, start_points):
        x = range(0, len(start_points))
        trend = self.get_trendloc(start_points)
        if trend is not None:
            error = start_points - trend(x)
            abs_error = np.abs(error)
            #plot_scope(error)
            #ch0 = error / 1000
            #self.sink.write(ch0, ch0)
            try:
                start_loc = int(np.where(error == min(abs_error))[0])
            except TypeError:
                start_loc = 0
        else:
            start_loc = 0

        if len(start_points) > 0:
            return start_points[start_loc]
        else:
            return 0

    def get_trend(self, start_points):
        x = range(0, len(start_points))
        return self.get_trendloc(start_points)(x)

    def get_mutemask(self, start_loc):
        zeros = np.zeros(self.eq_pulselen)
        ones = np.ones(self.linelen - self.eq_pulselen)
        mask = np.array([])
        linemask = np.roll(np.append(zeros, ones), start_loc)
        while len(mask) < self.blocklen:
            mask = np.append(mask, linemask)
        return mask[:self.blocklen]

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

    def get_sampling_pos(self, start_loc):
        hspace = range(start_loc, self.blocklen, self.linelen)
        return np.array(hspace)

    def sink_levels(self, hlevels, blevels):
        attenuation = 2e6
        hlevel_bias, blevel_bias = np.mean(hlevels), np.mean(blevels)
        ch0, ch1 = (hlevels - hlevel_bias) / attenuation, (blevels - blevel_bias) / attenuation
        self.sink.write(ch0, ch1)

    def get_levels(self, data, sampling_pos):
        levels = np.array([])
        for pos in sampling_pos:
            start = pos
            end = min(pos + self.integration_constant, self.blocklen)
            levels = np.append(levels, np.mean(data[start:end]))

        return levels

    def get_syncblank(self, data, start_points):
        start_pos = self.get_startloc(start_points) % self.linelen
        hstart_pos = self.hsync_level_delay + start_pos
        bstart_pos = self.blank_level_delay + start_pos
        hsampling_pos = self.get_sampling_pos(hstart_pos)
        bsampling_pos = self.get_sampling_pos(bstart_pos)
        hlevels = self.get_levels(data, hsampling_pos)
        blevels = self.get_levels(data, bsampling_pos)

        if len(blevels) < len(hlevels):
            blevels = np.append(blevels, blevels[-1:])

        self.sink_levels(hlevels, blevels)
        return hlevels, blevels, hsampling_pos

    def compensate_syncs(self, data, locs, hlevels):
        assert len(locs) == len(hlevels), "hsync locs should be the same length as level measures"
        bias = self.scale(self.SysParams['vsync_ire'])
        result = np.add(data[:locs[0]], -hlevels[0] + bias)
        for id, pos in enumerate(locs):
            end = min(pos + self.linelen, self.blocklen)
            result = np.append(
                result, np.add(data[pos:end], -hlevels[id] + bias)
            )

        return result

    def get_syncedgelocs(self, data):
        hsync = self.hsyncFilter.filtfilt(data)
        hsync_diff = np.diff(hsync)
        all_points = argrelextrema(hsync_diff, np.less)[0]
        start_points = self.filter_real_min_extrema(hsync_diff, all_points)

        hlevels, blevels, hstart_pos = self.get_syncblank(data, start_points)

        self.levels = hlevels, blevels
        self.hpos = hstart_pos

    def compensate_sync(self, data):
        assert len(self.hpos) > 0, 'You should call work() before this'
        #mute_mask = self.get_mutemask(self.hpos[0])
        #add_mask = self.get_addmask(self.hpos[0], self.levels[0])
        #resync = np.add(np.multiply(data, mute_mask), add_mask)
        return self.compensate_syncs(data, self.hpos, self.levels[0])

    def work(self, data):
        self.get_syncedgelocs(data)
        #print(mask)
        #plot_scope(mask)
        #dualplot_scope(data[:self.linelen], mask[:self.linelen])
