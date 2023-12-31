#!/usr/bin/env python3
import os
import sys
import signal
import json
import traceback
import lddecode.utils as lddu
from lddecode.utils_logging import init_logging
from cvbsdecode.process import CVBSDecode
import vhsdecode.formats as f
from vhsdecode.cmdcommons import (
    common_parser,
    select_sample_freq,
    select_system,
    get_basics,
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
        dest="auto_sync",
        action="store_true",
        default=False,
        help="Enable auto sync level detection.",
    )
    parser.add_argument(
        "--right_hand_hsync",
        dest="rhs_hsync",
        action="store_true",
        default=False,
        help="Additionally use right hand side of hsync for line start detection. Improves accuracy on tape sources but might cause issues on high bandwidth stable ones",
    )

    args = parser.parse_args(args)
    filename, outname, firstframe, req_frames = get_basics(args)
    system = select_system(args)
    sample_freq = select_sample_freq(args)

    if not args.overwrite:
        conflicts_ext = [".tbc", ".log", ".tbc.json"]
        conflicts = []

        for ext in conflicts_ext:
            if os.path.isfile(outname + ext):
                conflicts.append(outname + ext)

        if conflicts:
            print("Existing decode files found, remove them or run command with --overwrite")
            for conflict in conflicts:
                print("\t", conflict)
            sys.exit(1)

    try:
        loader = lddu.make_loader(filename, sample_freq)
    except ValueError as e:
        print(e)
        sys.exit(1)

    rf_options = get_rf_options(args)
    rf_options["auto_sync"] = args.auto_sync
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

    # if args.MTF is not None:
    #    ldd.rf.mtf_mult = args.MTF

    # if args.MTF_offset is not None:
    #    ldd.rf.mtf_offset = args.MTF_offset

    def write_json(vhsd, outname):
        jsondict = vhsd.build_json()

        fp = open(outname + ".tbc.json.tmp", "w")
        json.dump(jsondict, fp, indent=4)
        fp.write("\n")
        fp.close()

        os.rename(outname + ".tbc.json.tmp", outname + ".tbc.json")

    done = False

    jsondumper = lddu.jsondump_thread(vhsd, outname)

    def cleanup(outname):
        jsondumper.put(vhsd.build_json())
        vhsd.close()
        jsondumper.put(None)

    while not done and vhsd.fields_written < (req_frames * 2):
        try:
            f = vhsd.readfield()
        except KeyboardInterrupt:
            print("Terminated, saving JSON and exiting")
            cleanup(outname)
            sys.exit(1)
        except Exception as err:
            print(
                "\nERROR - please paste the following into a bug report:",
                file=sys.stderr,
            )
            print("current sample:", vhsd.fdoffset, file=sys.stderr)
            print("arguments:", args, file=sys.stderr)
            print("Exception:", err, " Traceback:", file=sys.stderr)
            traceback.print_tb(err.__traceback__)
            cleanup(outname)
            sys.exit(1)

        if f is None:
            # or (args.ignoreleadout == False and vhsd.leadOut == True):
            done = True
        else:
            f.prevfield = None

        if vhsd.fields_written < 100 or ((vhsd.fields_written % 500) == 0):
            jsondumper.put(vhsd.build_json())

    print("saving JSON and exiting")
    cleanup(outname)
    sys.exit(0)
