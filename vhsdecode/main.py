import argparse
import os
import sys
import signal
import traceback
import time

import numpy

# We want to run this before importing other modules to make sure modules are auto-compiled.
# Black will separate this statement and cause warnings so make it ignore it.
# fmt: off
import pyximport; pyximport.install(language_level=3, setup_args={'include_dirs': numpy.get_include()}, reload_support=True)  # noqa: E702
# fmt: on

import lddecode.utils as lddu
from lddecode.utils_logging import init_logging
from vhsdecode.process import VHSDecode
from vhsdecode.cmdcommons import (
    common_parser,
    select_sample_freq,
    select_system,
    get_basics,
    get_rf_options,
    get_extra_options,
    IOArgsException,
    test_input_file,
    test_output_file,
)

supported_tape_formats = {
    "VHS",
    "VHSHQ",
    "SVHS",
    "UMATIC",
    "UMATIC_HI",
    "BETAMAX",
    "BETAMAX_HIFI",
    "VIDEO8",
    "HI8",
    "EIAJ",
    "VCR",
    "VCR_LP",
    "TYPEC",
    "TYPEB",
}


def main(args=None, use_gui=False):
    import vhsdecode.formats as f

    parser, debug_group = common_parser(
        "Extracts video from raw rf captures of colour-under tapes", use_gui=use_gui
    )
    if not use_gui:
        parser.add_argument(
            "--doDOD",
            dest="dodod",
            action="store_true",
            default=False,
            help=argparse.SUPPRESS,
        )
        parser.add_argument(
            "-U",
            "--u-matic",
            dest="umatic",
            action="store_true",
            default=False,
            help=argparse.SUPPRESS,
        )
    parser.add_argument(
        "--tf",
        "--tape_format",
        type=str.upper,
        dest="tape_format",
        metavar="tape_format",
        default="VHS",
        choices=supported_tape_formats,
        help="Tape format, currently VHS (Default), VHSHQ, SVHS, UMATIC, UMATIC_HI, BETAMAX, BETAMAX_HIFI, VIDEO8, HI8 ,EIAJ, VCR, VCR_LP, TYPEC and TYPEB, are supported",
    )
    parser.add_argument(
        "--params_file",
        type=argparse.FileType("r"),
        dest="params_file",
        metavar="params_file_json",
        default=None,
        help="Override format/system parameters with specified json file.",
    )
    luma_group = parser.add_argument_group("Luma decoding options")
    luma_group.add_argument(
        "-L",
        "--level_adjust",
        dest="level_adjust",
        metavar="IRE Multiplier",
        type=float,
        default=0.1,
        help="Multiply top/bottom IRE in json by 1 +/- this value (used to avoid clipping on RGB conversion in chroma decoder).",
    )
    luma_group.add_argument(
        "--high_boost",
        metavar="High frequency boost multiplier",
        type=float,
        default=None,
        help="Multiplier for boost to high rf frequencies, uses default if not specified. Subject to change.",
    )
    luma_group.add_argument(
        "--nodd",
        "--no_diff_demod",
        dest="disable_diff_demod",
        action="store_true",
        default=False,
        help="Disable diff demod",
    )
    if not use_gui:
        luma_group.add_argument(
            "--noclamp",
            "--no_clamping",
            dest="disable_dc_offset",
            action="store_true",
            default=True,
            help=argparse.SUPPRESS,
            # help="Disable blanking DC offset clamping/compensation (no effect as this is the default currently)",
        )
    luma_group.add_argument(
        "--clamp",
        dest="enable_dc_offset",
        action="store_true",
        default=False,
        help="Enable blanking DC offset clamping/compensation",
    )
    luma_group.add_argument(
        "--nld",
        "--non_linear_deemphasis",
        dest="nldeemp",
        action="store_true",
        default=False,
        help="Enable primitive clipping non-linear deemphasis, can help reduce ringing and oversharpening. (WIP).",
    )
    luma_group.add_argument(
        "--sd",
        "--sub_deemphasis",
        dest="subdeemp",
        action="store_true",
        default=False,
        help="Enable non-linear sub deemphasis. (WIP).",
    )
    luma_group.add_argument(
        "--y_comb",
        dest="y_comb",
        metavar="IRE",
        action="store",
        nargs="?",
        type=float,
        default=0,
        const=1.5,
        help="Enable y comb filter, optionally specifying IRE limit.",
    )
    chroma_group = parser.add_argument_group("Chroma decoding options")
    chroma_group.add_argument(
        "--cafc",
        "--chroma_AFC",
        dest="cafc",
        action="store_true",
        default=False,
        help="Tries to detect the chroma carrier frequency on a field basis within some limit instead of using the default one for the format. Mainly useful for debug purposes and used on PAL betamax. implies --recheck_phase",
    )
    chroma_group.add_argument(
        "-T",
        "--track_phase",
        metavar="Track phase",
        type=int,
        default=None,
        help="If set to 0 or 1, force use of video track phase. (No effect on U-matic)",
    )
    chroma_group.add_argument(
        "--recheck_phase",
        dest="recheck_phase",
        action="store_true",
        default=False,
        help="Re-check chroma phase on every field. (No effect on U-matic)",
    )
    chroma_group.add_argument(
        "--no_comb",
        dest="disable_comb",
        action="store_true",
        default=False,
        help="Disable internal chroma comb filter.",
    )
    chroma_group.add_argument(
        "--skip_chroma",
        dest="skip_chroma",
        action="store_true",
        default=False,
        help="Don't output chroma even for formats that may have it and possibly skip some of the chroma processing.",
    )
    plot_options = "demodblock, deemphasis, raw_pulses, line_locs"
    debug_group.add_argument(
        "--dp",
        "--debug_plot",
        dest="debug_plot",
        help="Do a plot for the requested data, separated by whitespace. Current options are: "
        + plot_options
        + ".",
    )
    debug_group.add_argument(
        "--drh",
        "--disable_right_hsync",
        dest="disable_right_hsync",
        action="store_true",
        default=False,
        help="Disable use of right side of hsync for lineloc detection (old behaviour)",
    )
    debug_group.add_argument(
        "--level_detect_divisor",
        dest="level_detect_divisor",
        metavar="value",
        type=int,
        default=3,
        help="Use only every nth sample for vsync serration code - may improve speed at cost of minor accuracy. Limited to max 10.",
    )
    debug_group.add_argument(
        "--no_resample",
        dest="no_resample",
        action="store_true",
        default=False,
        help="Skip resampling input to 40 mhz (needs testing).",
    )
    debug_group.add_argument(
        "--fallback_vsync",
        dest="fallback_vsync",
        action="store_true",
        default=False,
        help="Enable vsync detect fallback. Will be enabled by default once more tested, so expect this option to change. Always enabled when using TypeC tape format",
    )
    debug_group.add_argument(
        "--use_saved_levels",
        dest="saved_levels",
        action="store_true",
        default=False,
        help="Try re-using video levels detected from the first decoded fields instead of re-calculating each frame. Will be done by default once well tested",
    )
    debug_group.add_argument(
        "--export_raw_tbc",
        dest="export_raw_tbc",
        action="store_true",
        default=False,
        help="export a raw TBC without deemphasis applied for filter tuning",
    )
    dodgroup = parser.add_argument_group("Dropout detection options")
    dodgroup.add_argument(
        "--noDOD",
        dest="nodod",
        action="store_true",
        default=False,
        help="Disable dropout detector.",
    )
    dodgroup.add_argument(
        "-D",
        "--dod_t",
        "--dod_threshold_p",
        dest="dod_threshold_p",
        metavar="value",
        type=float,
        default=None,
        help="RF level fraction threshold for dropouts as percentage of average (in decimal).",
    )
    dodgroup.add_argument(
        "--dod_t_abs",
        "--dod_threshold_abs",
        dest="dod_threshold_a",
        metavar="value",
        type=float,
        default=None,
        help="RF level threshold absolute value. Note that RF levels vary greatly between tapes and recording setups.",
    )
    dodgroup.add_argument(
        "--dod_h",
        "--dod_hysteresis",
        dest="dod_hysteresis",
        metavar="value",
        type=float,
        default=f.DEFAULT_HYSTERESIS,
        help="Dropout detection hysteresis, the rf level needs to go above the dropout threshold multiplied by this for a dropout to end.",
    )

    args = parser.parse_args(args)

    try:
        filename, outname, firstframe, req_frames = get_basics(args)
    except IOArgsException as e:
        parser.print_help()
        print(e)
        print(f"ERROR: input file '{args.infile}' not found" if not test_input_file(args.infile) else "Input file: OK")
        print(f"ERROR: output file '{args.outfile}' is not writable" if not test_output_file(args.outfile) else "Output file: OK")
        sys.exit(1)

    if not args.overwrite:
        conflicts_ext = [".tbc", "_chroma.tbc", ".log", ".tbc.json"]
        conflicts = []

        for ext in conflicts_ext:
            if os.path.isfile(outname + ext):
                conflicts.append(outname + ext)

        if conflicts:
            print(
                "Existing decode files found, remove them or run command with --overwrite"
            )
            for conflict in conflicts:
                print("\t", conflict)
            sys.exit(1)

    system = select_system(args)
    sample_freq = select_sample_freq(args)

    loader_input_freq = sample_freq if not args.no_resample else None
    if sample_freq == 40 and (filename.endswith(".lds") or filename.endswith(".ldf")):
        # Needs to be set to 0 so the loader does not try to resample.
        # TODO: Fix this properly
        loader_input_freq = None

    if not args.no_resample:
        sample_freq = 40

    try:
        loader = lddu.make_loader(filename, loader_input_freq)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Note: Fallback to ffmpeg, not .lds format
    # Temporary workaround until this is sorted upstream.
    if loader is lddu.load_packed_data_4_40 and not filename.endswith(".lds"):
        loader = lddu.LoadFFmpeg()

    dod_threshold_p = f.DEFAULT_THRESHOLD_P_DDD
    if args.cxadc or args.cxadc3 or args.cxadc_tenbit or args.cxadc3_tenbit:
        dod_threshold_p = f.DEFAULT_THRESHOLD_P_CXADC

    rf_options = get_rf_options(args)
    rf_options["dod_threshold_p"] = dod_threshold_p
    rf_options["dod_threshold_a"] = args.dod_threshold_a
    rf_options["dod_hysteresis"] = args.dod_hysteresis
    rf_options["track_phase"] = args.track_phase
    rf_options["recheck_phase"] = args.recheck_phase
    rf_options["high_boost"] = args.high_boost
    rf_options["disable_diff_demod"] = args.disable_diff_demod
    rf_options["disable_dc_offset"] = not args.enable_dc_offset
    rf_options["disable_comb"] = args.disable_comb
    rf_options["skip_chroma"] = args.skip_chroma
    rf_options["nldeemp"] = args.nldeemp
    rf_options["subdeemp"] = args.subdeemp
    rf_options["y_comb"] = args.y_comb
    rf_options["cafc"] = args.cafc
    rf_options["disable_right_hsync"] = args.disable_right_hsync
    rf_options["level_detect_divisor"] = args.level_detect_divisor
    rf_options["fallback_vsync"] = args.fallback_vsync
    rf_options["saved_levels"] = args.saved_levels
    rf_options["skip_hsync_refine"] = args.skip_hsync_refine
    rf_options["export_raw_tbc"] = args.export_raw_tbc

    extra_options = get_extra_options(args, not use_gui)
    extra_options["params_file"] = args.params_file

    # Wrap the LDdecode creation so that the signal handler is not taken by sub-threads,
    # allowing SIGINT/control-C's to be handled cleanly
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = init_logging(outname + ".log")

    if not use_gui and args.umatic:
        tape_format = "UMATIC"
    else:
        tape_format = args.tape_format.upper()
    if tape_format not in supported_tape_formats:
        logger.warning("Tape format %s not supported! Defaulting to VHS", tape_format)

    if not use_gui and args.dodod:
        logger.warning(
            "--doDOD is deprecated, dod is on by default, use noDOD to turn off."
        )

    debug_plot = None
    if args.debug_plot:
        from vhsdecode.debug_plot import DebugPlot

        debug_plot = DebugPlot(args.debug_plot)

    # Initialize VHS decoder
    # Note, we pass 40 as sample frequency, as any other will be resampled by the
    # loader function.
    vhsd = VHSDecode(
        filename,
        outname,
        loader,
        logger,
        system=system,
        tape_format=tape_format,
        doDOD=not args.nodod,
        threads=args.threads if not debug_plot else 1,
        inputfreq=sample_freq,
        level_adjust=args.level_adjust,
        rf_options=rf_options,
        extra_options=extra_options,
        debug_plot=debug_plot,
    )

    signal.signal(signal.SIGINT, original_sigint_handler)

    if args.start_fileloc != -1:
        vhsd.roughseek(args.start_fileloc, False)
    else:
        vhsd.roughseek(firstframe * 2)

    if system == "NTSC" and not args.ntscj:
        vhsd.blackIRE = 7.5

    done = False

    jsondumper = lddu.jsondump_thread(vhsd, outname)

    def cleanup():
        jsondumper.put(vhsd.build_json())
        vhsd.close()
        jsondumper.put(None)

    # TODO: Put the stuff below this in a function so we can re-use for both vhs and cvbs

    # seconddecode is taken so that setup time is not included in FPS calculation
    firstdecode = time.time()
    seconddecode = None

    while not done and vhsd.fields_written < (req_frames * 2):
        try:
            f = vhsd.readfield()
            if not seconddecode:
                seconddecode = time.time()
        except KeyboardInterrupt:
            print("Terminated, saving JSON and exiting")
            cleanup()
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
            cleanup()
            sys.exit(1)

        if f is None:
            done = True
        else:
            f.prevfield = None

        if vhsd.fields_written < 100 or ((vhsd.fields_written % 500) == 0):
            jsondumper.put(vhsd.build_json())

    if vhsd.fields_written:
        timeused = time.time() - firstdecode
        frames = vhsd.fields_written // 2
        fps = frames / timeused

        print(
            f"\nCompleted: saving JSON and exiting.  Took {timeused:.2f} seconds to decode {frames} frames ({fps:.2f} average FPS)",
            file=sys.stderr,
        )
    else:
        print(f"\nCompleted without handling any frames.", file=sys.stderr)

    cleanup()
    sys.exit(0)
