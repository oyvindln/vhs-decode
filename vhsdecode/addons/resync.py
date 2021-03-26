from vhsdecode.addons.FMdeemph import FMDeEmphasis
from vhsdecode.utils import FiltersClass, firdes_lowpass, plot_scope, dualplot_scope, filter_plot, zero_cross_det
import numpy as np
from scipy.signal import argrelextrema

def identity(value):
    return value

def t_to_samples(samp_rate, value):
    return samp_rate / value

class DCrestore:

    def __init__(self, fs, tau, sysparams, ignore, scale=identity):
        self.samp_rate = fs
        self.tau = tau
        self.SysParams = sysparams
        self.ignoresamples = ignore
        self.scale = scale
        self.fv = self.SysParams["FPS"] * 2
        self.fh = self.SysParams["FPS"] * self.SysParams["frame_lines"]

        iir_deemph = FMDeEmphasis(self.samp_rate, self.tau).get()
        #filter_plot(iir_deemph[0], iir_deemph[1], self.samp_rate, type='lowpass', title='FM deemph')
        self.deemphFilter = FiltersClass(iir_deemph[0], iir_deemph[1], self.samp_rate)

        self.harmonic_limit = 3
        iir_hsync = firdes_lowpass(self.samp_rate, self.fh * self.harmonic_limit, 110e3, 3)
        #filter_plot(iir_hsync[0], iir_hsync[1], self.samp_rate, type='lowpass', title='Hsync')
        self.hsyncFilter = FiltersClass(iir_hsync[0], iir_hsync[1], self.samp_rate)

        iir_vsync = firdes_lowpass(self.samp_rate, self.fv * self.harmonic_limit, 220e3, 2)
        self.vsyncFilter = FiltersClass(iir_vsync[0], iir_vsync[1], self.samp_rate)
        #filter_plot(iir_vsync[0], iir_vsync[1], self.samp_rate, type='lowpass', title='Vsync')

        self.eq_pulselen = round(t_to_samples(self.samp_rate, 1 / (self.SysParams["eqPulseUS"] * 1e-6)))
        self.linelen = round(t_to_samples(self.samp_rate, self.fh))
        self.last_lineoffset = 0

    def gen_hsyncmask(self):
        eq_pulse = np.zeros(self.eq_pulselen)
        line_pulse = np.ones(self.linelen - self.eq_pulselen)
        return np.append(eq_pulse, line_pulse)

    def get_rawlinelocs(self, data, int_window=10):
        hsync = self.hsyncFilter.filtfilt(data)
        level = np.ones(len(data)) * self.scale(-self.SysParams["vsync_ire"] / 3)
        cross = hsync - level
        where = zero_cross_det(cross)
        if len(where) > 20:
            where_diff = np.diff(where)
            end_points = argrelextrema(where_diff, np.greater)[0]
            start_points = argrelextrema(where_diff, np.less)[0]
            step = int(np.mean(np.diff(end_points)))

            # edge case
            while len(end_points) < len(start_points):
                end_points = np.append(end_points, end_points[len(end_points)-1] + step)

                if end_points[len(end_points)-1] >= len(where):
                    end_points[len(end_points) - 1] = len(where) - 1

            gated_hsync = list()
            sync_min_holds = list()
            sync_starts = list()
            for i, start in enumerate(start_points):
                try:
                    s_p = min(start_points[i], end_points[i])
                    e_p = max(start_points[i], end_points[i])
                    hsync_area = data[where[s_p]:where[e_p]]
                    if len(hsync_area) > 0:

                        gated_hsync.append(hsync_area)
                        #plot_scope(hsync_area)
                        hmin_id = int(np.where(hsync_area == min(hsync_area))[0])
                        hmin_id = max(hmin_id, int_window)
                        hmin_id = min(hmin_id, len(hsync_area) - int_window - 1)
                        sync_starts.append(where[s_p] + hmin_id)
                        hmin = np.mean(hsync_area[hmin_id-int_window:hmin_id+int_window])
                        sync_min_holds.append(hmin)
                        print('Hslice min, id', hmin, where[s_p] + hmin_id)

                except IndexError as e:
                    print(e)
                    print(len(where), start_points, end_points)
                    exit(0)

            plot_scope(sync_min_holds)
            print(len(sync_min_holds), sync_starts, np.diff(sync_starts))
            #plot_scope(result)
            #for id, pos in enumerate(where):
            #dualplot_scope(level[self.ignoresamples:], hsync[self.ignoresamples:])
            return None
        else:
            #plot_scope(self.gen_hsyncmask())
            print('cannot clamp signal')
            return None

    def work(self, data):
        deemph = self.deemphFilter.filtfilt(data)
        locs = self.get_rawlinelocs(deemph)
