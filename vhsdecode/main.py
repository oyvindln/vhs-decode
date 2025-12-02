import argparse
import os
import sys
import signal
import traceback
import json

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
from vhsdecode.formats import TAPE_SPEEDS
from vhsd_rust import check_debug

supported_tape_formats = {
    "VHS",
    "VHSHQ",
    "SVHS",
    "SVHS_ET",
    "UMATIC",
    "UMATIC_HI",
    "BETAMAX",
    "BETAMAX_HIFI",
    "SUPERBETA",
    "VIDEO8",
    "HI8",
    "EIAJ",
    "QUADRUPLEX",
    "VCR",
    "VCR_LP",
    "TYPEC",
    "TYPEB",
    "VIDEO2000",
}


def main(args=None, use_gui=False):
    import vhsdecode.formats as f

    format_help = "Tape format - " + " ".join(supported_tape_formats) + "are supported"

    parser, debug_group = common_parser(
        "Extracts video from RAW RF captures of colour-under & composite modulated"
        " tapes",
        use_gui=use_gui,
    )

    parser.add_argument(
        "--tf",
        "--tape_format",
        type=str.upper,
        dest="tape_format",
        metavar="tape_format",
        default="VHS",
        choices=supported_tape_formats,
        help=format_help,
    )
    parser.add_argument(
        "--ts",
        "--tape_speed",
        type=str.lower,
        dest="tape_speed",
        metavar="tape_speed",
        default="sp",
        choices=TAPE_SPEEDS.keys(),
        help=(
            "Tape speed selection for adjusting format parameters. SP (default), LP,"
            " SLP, EP, and VP. Only supported for some formats. SLP and EP refers to"
            " the same speed."
        ),
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
        help=(
            "Multiply top/bottom IRE in json by 1 +/- this value (used to avoid"
            " clipping on RGB conversion in chroma decoder)."
        ),
    )
    luma_group.add_argument(
        "--ire0_adjust",
        dest="ire0_adjust",
        action="store_true",
        default=False,
        help="Automatic adjust of ire0 based blanking level",
    )
    luma_group.add_argument(
        "--high_boost",
        metavar="High frequency boost multiplier",
        type=float,
        default=None,
        help=(
            "Multiplier for boost to high rf frequencies, uses default if not"
            " specified. Subject to change."
        ),
    )
    luma_group.add_argument(
        "--nodd",
        "--no_diff_demod",
        dest="disable_diff_demod",
        action="store_true",
        default=False,
        help="Disable diff demod",
    )
    luma_group.add_argument(
        "--fm_audio_notch",
        dest="fm_audio_notch",
        action="store",
        metavar="Q",
        nargs="?",
        default=0,
        const=10,
        type=float,
        help=(
            "Enable notch filter on FM audio frequencies to filter out wave-like"
            " pattern from interference, mainly useful on VHS. Optional argument to"
            " specify Q factor (filter width)"
        ),
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
        help=(
            "Enable primitive clipping non-linear deemphasis, can help reduce ringing"
            " and oversharpening. (WIP)."
        ),
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
        help=(
            "Tries to detect the chroma carrier frequency on a field basis within some"
            " limit instead of using the default one for the format. Mainly useful for"
            " debug purposes and used on PAL betamax. implies --recheck_phase"
        ),
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
        help=(
            "Don't output chroma even for formats that may have it and possibly skip"
            " some of the chroma processing."
        ),
    )
    plot_options = "demodblock, deemphasis, raw_pulses, line_locs"
    debug_group.add_argument(
        "--dp",
        "--debug_plot",
        dest="debug_plot",
        help=(
            "Do a plot for the requested data, separated by whitespace. Current options"
            " are: "
        )
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
        help=(
            "Use only every nth sample for vsync serration code - may improve speed at"
            " cost of minor accuracy. Limited to max 10."
        ),
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
        help=(
            "Enable vsync detect fallback. Will be enabled by default once more tested,"
            " so expect this option to change. Always enabled when using TypeC tape"
            " format"
        ),
    )
    debug_group.add_argument(
        "--relaxed_line0",
        dest="relaxed_line0",
        action="store_true",
        default=False,
        help=(
            "Enable relaxed line0 detection fallback logic. Useful for damaged tapes where "
            "standard detection fails. Requires --fallback_vsync."
        ),
    )
    debug_group.add_argument(
        "--field_order_confidence",
        dest="field_order_confidence",
        default=100,
        metavar="value",
        type=int,
        help=(
            "Allow field order cadence to change after n percent of field order pulses detected.\n"
            "  Reduce this number if you experience field order issues when decoding sources with multiple recordings, such as home recordings\n"
            "  Increase this number if there is damage to the v-sync area to prevent incorrect field order detection\n"
            "  Range 0-100; Sane Values 50-100\n"
            "    100, (default) all field order pulses must match to change field cadence\n"
            "     50, half of field order pulses must match to change field cadence\n"
            "      0, any field order pulse can match to change field cadence\n"
        ),
    )
    debug_group.add_argument(
        "--field_order_action",
        dest="field_order_action",
        default="detect",
        metavar="value",
        type=lambda x: x if x in ["detect", "duplicate", "drop", "none"] else parser.error('--field_order_action must be one of ["detect", "duplicate", "drop", "none"]'),
        help=(
            "Decides how to handle field order discontinuities,\n"
            "  When the field order cadence is broken such that there are two Top or two Bottom fields.\n"
            "  * `detect`:    (default) use the distance between the dropped field to determine whether to drop or duplicate.\n"
            "  * `duplicate`: always duplicate the last valid field\n"
            "  * `drop`:      always drop the last valid field\n"
            "  * `none`:      do nothing, (field order will be out of sync causing issues for interlaced video)\n"
        ),
    )
    debug_group.add_argument(
        "--use_saved_levels",
        dest="saved_levels",
        action="store_true",
        default=False,
        help=(
            "Try re-using video levels detected from the first decoded fields instead"
            " of re-calculating each frame. Will be done by default once well tested"
        ),
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
        help=(
            "RF level fraction threshold for dropouts as percentage of average (in"
            " decimal)."
        ),
    )
    dodgroup.add_argument(
        "--dod_t_abs",
        "--dod_threshold_abs",
        dest="dod_threshold_a",
        metavar="value",
        type=float,
        default=None,
        help=(
            "RF level threshold absolute value. Note that RF levels vary greatly"
            " between tapes and recording setups."
        ),
    )
    dodgroup.add_argument(
        "--dod_h",
        "--dod_hysteresis",
        dest="dod_hysteresis",
        metavar="value",
        type=float,
        default=f.DEFAULT_HYSTERESIS,
        help=(
            "Dropout detection hysteresis, the rf level needs to go above the dropout"
            " threshold multiplied by this for a dropout to end."
        ),
    )
    dodgroup.add_argument(
        "--gnrc",
        "--gnuradio_rf_afe",
        dest="gnrc_afe",
        action="store_true",
        default=False,
        help=(
            "Open a ZMQ pipe back and forth to GNU Radio for RF AFE/EQ/Group delay"
            " measurements. (WIP)\nYou might want to use this with -t 1"
        ),
    )

    args = parser.parse_args(args)

    try:
        filename, outname, firstframe, req_frames = get_basics(args)
    except IOArgsException as e:
        parser.print_help()
        print(e)
        print(
            f"ERROR: input file '{args.infile}' not found"
            if not test_input_file(args.infile)
            else "Input file: OK"
        )
        print(
            f"ERROR: output file '{args.outfile}' is not writable"
            if not test_output_file(args.outfile)
            else "Output file: OK"
        )
        sys.exit(1)

    if not args.overwrite:
        conflicts_ext = [".tbc", "_chroma.tbc", ".log", ".tbc.json"]
        conflicts = []

        for ext in conflicts_ext:
            if os.path.isfile(outname + ext):
                conflicts.append(outname + ext)

        if conflicts:
            print(
                "Existing decode files found, remove them or run command with"
                " --overwrite"
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
    rf_options["fm_audio_notch"] = args.fm_audio_notch
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
    rf_options["relaxed_line0"] = args.relaxed_line0
    rf_options["field_order_confidence"] = int(max(0, min(100, args.field_order_confidence)))
    rf_options["saved_levels"] = args.saved_levels
    rf_options["skip_hsync_refine"] = args.skip_hsync_refine
    rf_options["export_raw_tbc"] = args.export_raw_tbc
    rf_options["tape_speed"] = args.tape_speed
    rf_options["ire0_adjust"] = args.ire0_adjust
    rf_options["gnrc_afe"] = args.gnrc_afe

    extra_options = get_extra_options(args, not use_gui)
    extra_options["params_file"] = args.params_file

    # Wrap the LDdecode creation so that the signal handler is not taken by sub-threads,
    # allowing SIGINT/control-C's to be handled cleanly
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)

    logger = init_logging(outname + ".log")

    tape_format = args.tape_format.upper()
    if tape_format not in supported_tape_formats:
        logger.warning("Tape format %s not supported! Defaulting to VHS", tape_format)

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
        threads=args.threads if not debug_plot else 0,
        inputfreq=sample_freq,
        level_adjust=args.level_adjust,
        rf_options=rf_options,
        extra_options=extra_options,
        debug_plot=debug_plot,
        field_order_action=args.field_order_action
    )

    if check_debug():
        logger.warning(
            "Rust modules are compiled in debug mode! vhs-decode will run slower."
        )

    signal.signal(signal.SIGINT, original_sigint_handler)

    if args.start_fileloc != -1:
        vhsd.roughseek(args.start_fileloc, False)
    else:
        vhsd.roughseek(firstframe * 2)

    if system == "NTSC" and not args.ntscj:
        vhsd.blackIRE = 7.5

    done = False

    jsondumper = lddu.JSONDumper(vhsd, outname)

    def cleanup():
        jsondumper.close()
        vhsd.close()

    logger.debug("Sys Parameters: \n" + json.dumps(vhsd.rf.SysParams, sort_keys=True, indent=4))
    logger.debug("RF Parameters: \n" + json.dumps(vhsd.rf.DecoderParams, sort_keys=True, indent=4))

    while not done and vhsd.fields_written < (req_frames * 2):
        try:
            f = vhsd.readfield()
        except KeyboardInterrupt:
            print("\nTerminated, saving JSON and exiting")
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
            jsondumper.write()

    if vhsd.fields_written:
        print("\nCompleted: saving JSON and exiting.", file=sys.stderr)
    else:
        print("\nCompleted without handling any frames.", file=sys.stderr)

    cleanup()
    sys.exit(0)
