from dataclasses import dataclass
import numpy as np
from numba import njit
import math

import vhsdecode.formats as vhs_formats


@dataclass
class DodOptions:
    dod_threshold_p: float
    dod_threshold_a: float
    dod_hysteresis: float


@njit(cache=True)
def find_crossings(data, threshold):
    """Find where the data crosses the set threshold."""

    # We do this by constructing array where positions above
    # the threshold are marked as true, other sfalse,
    # and use diff to mark where the value changes.
    crossings = np.diff(data < threshold)
    # TODO: See if we can avoid reduntantly looking for both up and
    # down crossing when we just need one of them.
    return crossings


@njit(cache=True, nogil=True, fastmath=True)
def find_dropouts_rf(env, start_rf, end_rf, threshold, hysteresis, merge_threshold):
    # list of tuples containing start and end
    down_thresh = threshold
    up_thresh = threshold * hysteresis

    dropouts = []
    dropout_idx = -1
    
    for i in range(start_rf, end_rf):
        v = env[i]
        dropout_ended = False

        if v <= down_thresh:
            if dropout_idx == -1 or (dropout_ended := dropouts[dropout_idx][1] != -1 and i - dropouts[dropout_idx][1] > merge_threshold):
                # start dropout if none exist or distance from previous is greater than the merge threshold
                dropout_idx += 1
                dropouts.append((i, -1))
            elif dropout_ended:
                # continue existing dropout
                dropouts[dropout_idx] = (dropouts[dropout_idx][0], -1)
        elif v >= up_thresh:
            # end dropout
            if dropout_idx != -1 and dropouts[dropout_idx][1] == -1:
                dropouts[dropout_idx] = (dropouts[dropout_idx][0], i)

    if dropout_idx != -1 and dropouts[dropout_idx][1] == -1:
        # set the dropout ending to the last sample when the dropout happens at the end of the field
        dropouts[dropout_idx] = (dropouts[dropout_idx][0], end_rf)

    return dropouts


def detect_dropouts_rf(field, dod_options):
    """Look for dropouts in the input data, based on rf envelope amplitude.
    Uses either an percentage of the frame average rf level, or an absolute value.
    TODO: A more advanced algorithm with hysteresis etc.
    """
    env = field.data["video"]["envelope"]
    threshold_p = dod_options.dod_threshold_p
    threshold_abs = dod_options.dod_threshold_a
    hysteresis = dod_options.dod_hysteresis

    threshold = 0.0
    field_average = np.mean(field.data["video"]["envelope"])
    # Store the average for later.
    field.rf.field_averages.rf_level.push(field_average)
    if threshold_abs is not None:
        threshold = threshold_abs
    else:
        # Generate a threshold based on the field envelope average.
        # This may not be ideal on a field with a lot of droputs,
        # so we may want to use statistics of the previous averages
        # to avoid the threshold ending too low.
        threshold = field_average * threshold_p

    start_line = field.lineoffset + 1
    end_line = min(len(field.linelocs) - 1, field.linecount + start_line + 1)

    start_rf = math.floor(field.linelocs[start_line])
    end_rf = min(len(env), math.ceil(field.linelocs[end_line]))

    dropouts_rf = find_dropouts_rf(env, start_rf, end_rf, threshold, hysteresis, vhs_formats.DOD_MERGE_THRESHOLD)

    # Drop very short dropouts that were not merged.
    # We do this after mergin to avoid removing short consecutive dropouts that
    # could be merged.
    dropouts_rf = list(filter(lambda s: s[1] - s[0] > vhs_formats.DOD_MIN_LENGTH, dropouts_rf))

    debug_plot = getattr(field.rf, "debug_plot", None)
    if debug_plot and debug_plot.is_plot_requested("luma_noise"):
        from vhsdecode.debug_plot import plot_luma_noise

        plot_luma_noise(
            env,
            field.data["video"]["demod"],
            dropouts_rf,
            threshold,
            hysteresis,
            start_rf,
            end_rf,
            field.linelocs,
            field.rf.hztoire,
            field.rf.dod_options.dod_threshold_p,
        )

    return map_dropouts_rf_to_tbc(dropouts_rf, start_line, end_line, field.linelocs, field.outlinelen, field.lineoffset)

def map_dropouts_rf_to_tbc(errlist, start_line_idx, end_line_idx, linelocs, outlinelen, lineoffset):
    rv_lines = []
    rv_starts = []
    rv_ends = []

    line_idx = start_line_idx
    line_start_rf = linelocs[line_idx]
    line_end_rf = linelocs[line_idx + 1]

    for (start_rf, end_rf) in errlist:
        while line_idx < end_line_idx:
            # find the line that contains start of the dropout
            line_len = line_end_rf - line_start_rf
            if (start_rf >= line_start_rf or line_idx == start_line_idx) and start_rf < line_end_rf and line_len > 0:
                rv_lines.append(line_idx - lineoffset)
                
                # scale down to tbc line position
                start_rf_linepos = start_rf - line_start_rf
                start_linepos = math.floor(start_rf_linepos / line_len * outlinelen)

                rv_starts.append(max(0, start_linepos))
                break
            else:
                line_idx += 1
                line_start_rf = linelocs[line_idx]
                line_end_rf = linelocs[line_idx + 1]

        while line_idx < end_line_idx:
            line_len = line_end_rf - line_start_rf
            if end_rf < line_end_rf and line_len > 0:
                # dropout is contained within this line
                # scale down to tbc line position
                end_rf_linepos = end_rf - line_start_rf
                end_linepos = math.ceil(end_rf_linepos / line_len * outlinelen)

                rv_ends.append(min(outlinelen, end_linepos))
                break
            else:
                # dropout spans multiple lines
                rv_ends.append(outlinelen)
                line_idx += 1

                if line_idx < end_line_idx:
                    # continue the dropout to the next line
                    line_start_rf = linelocs[line_idx]
                    line_end_rf = linelocs[line_idx + 1]
                    
                    rv_starts.append(0)
                    rv_lines.append(line_idx - lineoffset)

    return rv_lines, rv_starts, rv_ends

