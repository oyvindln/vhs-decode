from vhsdecode.addons.vsyncserration import VsyncSerration
import numpy as np
import vhsdecode.utils as utils
import lddecode.utils as lddu
import itertools
from lddecode.utils import inrange
import lddecode.core as ldd
import json
import math
import hashlib

# stores the last valid blacklevel, synclevel and vsynclocs state
# preliminary solution to fix spurious decoding halts (numpy error case)
class FieldState:
    def __init__(self):
        self.blanklevels = np.array([])
        self.synclevels = np.array([])
        self.locs = None
        self.field_average = 30
        self.min_watermark = 3

    def setSyncLevel(self, level):
        self.synclevels = np.append(self.synclevels, level)

    def setLevels(self, blank, sync):
        self.blanklevels = np.append(self.blanklevels, blank)
        self.setSyncLevel(sync)

    def getSyncLevel(self):
        if np.size(self.synclevels) > 0:
            synclevel, self.synclevels = utils.moving_average(
                self.synclevels, window=self.field_average
            )
            return synclevel
        else:
            return None

    def getLevels(self):
        if np.size(self.blanklevels) > 0:
            blacklevel, self.blanklevels = utils.moving_average(
                self.blanklevels, window=self.field_average
            )
            return blacklevel, self.getSyncLevel()
        else:
            return None, None

    def setLocs(self, locs):
        self.locs = locs

    def getLocs(self):
        return self.locs

    def hasLevels(self):
        return (
            np.size(self.blanklevels) > self.min_watermark
            and np.size(self.synclevels) > self.min_watermark
        )


class Resync:
    def __init__(self, fs, sysparams):
        self.samp_rate = fs
        self.SysParams = sysparams
        self.VsyncSerration = VsyncSerration(fs, sysparams)
        self.field_state = FieldState()

    def debug_field(self, sysParams, sync_reference):
        ldd.logger.debug("Hashed field sync reference %s" % hashlib.md5(sync_reference.tobytes('C')).hexdigest())
        sysparams_json = json.dumps(sysParams, indent=4).encode('utf-8')
        ldd.logger.debug("Hashed field SysParams %s" % hashlib.md5(sysparams_json).hexdigest())

    def getpulses_override(self, field):
        """Find sync pulses in the demodulated video signal

        NOTE: TEMPORARY override until an override for the value itself is added upstream.
        """
        # measures the serration levels if possible
        sync_reference = field.data["video"]["demod_05"]
        self.debug_field(field.rf.SysParams, sync_reference)

        self.VsyncSerration.work(sync_reference)
        # safe clips the bottom of the sync pulses but leaves picture area unchanged
        demod_data = self.VsyncSerration.safe_sync_clip(
            sync_reference, field.data["video"]["demod"]
        )

        # if has levels, then compensate blanking bias
        if self.VsyncSerration.has_levels() or self.field_state.hasLevels():
            if self.VsyncSerration.has_levels():
                sync, blank = self.VsyncSerration.get_levels()
            else:
                blank, sync = self.field_state.getLevels()

            if not field.rf.disable_dc_offset:
                dc_offset = field.rf.SysParams["ire0"] - blank
                sync_reference += dc_offset
                demod_data += dc_offset
                sync, blank = sync + dc_offset, blank + dc_offset
                # forced blank
                # field.data["video"]["demod"] = np.clip(field.data["video"]["demod"], a_min=sync, a_max=blank)

            field.data["video"]["demod_05"] = np.clip(
                sync_reference, a_min=sync, a_max=blank
            )
            field.data["video"]["demod"] = demod_data
            sync_ire, blank_ire = field.rf.hztoire(sync), field.rf.hztoire(blank)
            pulse_hz_min = field.rf.iretohz(sync_ire)
            pulse_hz_max = field.rf.iretohz((sync_ire + blank_ire) / 2)
        else:
            # pass one using standard levels (fallback sync logic)
            # pulse_hz range:  vsync_ire - 10, maximum is the 50% crossing point to sync
            pulse_hz_min = field.rf.iretohz(field.rf.SysParams["vsync_ire"] - 10)
            pulse_hz_max = field.rf.iretohz(field.rf.SysParams["vsync_ire"] / 2)

            # checks if the DC offset is abnormal before correcting it
            mean_bias = self.VsyncSerration.mean_bias()
            if (
                not field.rf.disable_dc_offset
                and not pulse_hz_min
                < mean_bias
                < field.rf.iretohz(field.rf.SysParams["vsync_ire"])
            ):
                field.data["video"]["demod_05"] = (
                    sync_reference
                    - mean_bias
                    + field.rf.iretohz(field.rf.SysParams["vsync_ire"])
                )
                field.data["video"]["demod"] = (
                    demod_data
                    - mean_bias
                    + field.rf.iretohz(field.rf.SysParams["vsync_ire"])
                )

        # utils.plot_scope(field.data["video"]["demod_05"])
        pulses = lddu.findpulses(
            field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
        )

        if len(pulses) == 0:
            # can't do anything about this
            return pulses

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

        if len(vsync_means) == 0:
            synclevel = self.field_state.getSyncLevel()
            if synclevel is None:
                return None
        else:
            synclevel = np.median(vsync_means)
            self.field_state.setSyncLevel(synclevel)
            self.field_state.setLocs(vsync_locs)

        if np.abs(field.rf.hztoire(synclevel) - field.rf.SysParams["vsync_ire"]) < 5:
            # sync level is close enough to use
            return pulses

        if vsync_locs is None or not len(vsync_locs):
            vsync_locs = self.field_state.getLocs()

        # Now compute black level and try again

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

        # Set to nan if empty to avoid warning.
        blacklevel = math.nan if len(black_means) == 0 else np.median(black_means)

        if np.isnan(blacklevel).any() or np.isnan(synclevel).any():
            # utils.plot_scope(field.data["video"]["demod_05"], title='Failed field demod05')
            bl, sl = self.field_state.getLevels()
            if bl is not None and sl is not None:
                blacklevel, synclevel = bl, sl
            else:
                return None
        else:
            self.field_state.setLevels(blacklevel, synclevel)

        pulse_hz_min = synclevel - (field.rf.SysParams["hz_ire"] * 10)
        pulse_hz_max = (blacklevel + synclevel) / 2

        return lddu.findpulses(
            field.data["video"]["demod_05"], pulse_hz_min, pulse_hz_max
        )
