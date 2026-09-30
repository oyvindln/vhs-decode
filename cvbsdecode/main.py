#!/usr/bin/env python3
import argparse
import sys
import signal


import lddecode.utils as lddu
from lddecode.utils_logging import init_logging
from cvbsdecode.process import CVBSDecode
from vhsdecode.cmdcommons import (
    common_parser,
    select_sample_freq,
    select_system,
    get_basics_or_exit,
    check_overwrite,
    decode_fields,
    get_rf_options,
    get_extra_options,
)


def main(args=None):
    parser, _ = common_parser("Extracts video from raw cvbs captures")
    parser.add_argument(
        "-S",
        "--seek",
        metavar="seek",
        type=int,
        default=-1,
        help="seek to frame n of capture",
    )
    parser.add_argument(
        "-A",
        "--auto_sync",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--no_auto_sync",
        dest="no_auto_sync",
        action="store_true",
        default=False,
        help="Disable auto sync level detection.",
    )
    parser.add_argument(
        "-C",
        "--clamp_agc",
        dest="clamp_agc",
        action="store_true",
        default=False,
        help="Clamp signal (black level) and enable automatic gain control.",
    )
    parser.add_argument(
        "--agc_speed",
        metavar="speed",
        type=float,
        default=0.1,
        help="sets how fast the AGC should react (0 = never, 1 = immediate).",
    )
    parser.add_argument(
        "--agc_gain_factor",
        metavar="factor",
        type=float,
        default=1.0,
        help="adjusts the AGC white level by given factor.",
    )
    parser.add_argument(
        "--agc_set_gain",
        metavar="gain",
        type=float,
        default=0.0,
        help="override gain of AGC.",
    )
    parser.add_argument(
        "--right_hand_hsync",
        dest="rhs_hsync",
        action="store_true",
        default=False,
        help="Additionally use right hand side of hsync for line start detection. Improves accuracy on tape sources but might cause issues on high bandwidth stable ones",
    )
    parser.add_argument(
        "--wow_level_adjust_smoothing",
        type=float,
        default=0,
        help=(
            "Adjusts the amount of smoothing in lines that is performed when compensating for brightness variations caused by wow. (default 0)"
            "\n  Wow calculation is based on position of hsync pulses which is affected by the accuracy of the TBC. "
            "\n  If you see vertical brightness variations (banding), setting to a value larger than 0 will smooth the wow adjustment."
        )
    )

    args = parser.parse_args(args)
    filename, outname, firstframe, req_frames = get_basics_or_exit(parser, args)
    system = select_system(args)
    sample_freq = select_sample_freq(args)
    check_overwrite(args, outname, [".tbc", ".log", ".tbc.json"])

    try:
        loader = lddu.make_loader(filename, sample_freq)
    except ValueError as e:
        print(e)
        sys.exit(1)

    rf_options = get_rf_options(args)
    rf_options["auto_sync"] = not args.no_auto_sync
    rf_options["clamp_agc"] = args.clamp_agc
    rf_options["agc_speed"] = args.agc_speed
    rf_options["agc_gain_factor"] = args.agc_gain_factor
    rf_options["agc_set_gain"] = args.agc_set_gain
    rf_options["rhs_hsync"] = args.rhs_hsync

    extra_options = get_extra_options(args)
    extra_options["cvbs"] = True

    # Wrap the LDdecode creation so that the signal handler is not taken by sub-threads,
    # allowing SIGINT/control-C's to be handled cleanly
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = init_logging(outname + ".log")

    # Initialize CVBS decoder
    # Note, we pass 40 as sample frequency, as any other will be resampled by the
    # loader function.
    vhsd = CVBSDecode(
        filename,
        outname,
        loader,
        logger,
        system=system,
        threads=args.threads,
        inputfreq=40,
        level_adjust=0.2,
        # level_adjust=args.level_adjust,
        rf_options=rf_options,
        extra_options=extra_options,
    )

    signal.signal(signal.SIGINT, original_sigint_handler)

    if args.start_fileloc != -1:
        vhsd.roughseek(args.start_fileloc, False)
    else:
        vhsd.roughseek(firstframe * 2)

    if system == "NTSC" and not args.ntscj:
        vhsd.blackIRE = 7.5

    if args.seek != -1:
        if vhsd.seek(args.seek if firstframe == 0 else firstframe, args.seek) is None:
            print("ERROR: Seeking failed", file=sys.stderr)
            sys.exit(1)

    cleanup = decode_fields(vhsd, args, outname, req_frames)

    if "lowest_agc_gain" in vhsd.rf.DecoderParams:
        print("Automatic gain control statistics:", file=sys.stderr)
        print(
            " Lowest detected gain:  ",
            vhsd.rf.DecoderParams["lowest_agc_gain"],
            file=sys.stderr,
        )
        print(
            " Highest detected gain: ",
            vhsd.rf.DecoderParams["highest_agc_gain"],
            file=sys.stderr,
        )
        print(
            " Lowest used gain:      ",
            vhsd.rf.DecoderParams["lowest_used_agc_gain"],
            file=sys.stderr,
        )
        print(
            " Highest used gain:     ",
            vhsd.rf.DecoderParams["highest_used_agc_gain"],
            file=sys.stderr,
        )
    print("saving JSON and exiting")
    cleanup()
    sys.exit(0)
