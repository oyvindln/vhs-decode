import argparse
import lddecode.utils as lddu


CXADC_FREQ = (8 * 315.0) / 88.0  # 28.636363636
CXADC_FREQ_HIGH = 3150.0 / 88.0  # 35.795454545
CXADC_TENBIT_FREQ = (8 * 315.0) / 88.0 / 2.0  # 14.318181818
CXADC_TENBIT_FREQ_HIGH = 3150.0 / 88.0 / 2.0  # 17.897727272


def add_argument_hidden_in_gui(parser, use_gui, *args, **kwargs):
    if use_gui:
        parser.add_argument(*args, **kwargs, gooey_options={"visible": False})
    else:
        parser.add_argument(*args, **kwargs)


def common_parser(meta_title, use_gui=False):
    if not use_gui:
        return common_parser_cli(meta_title)
    else:
        return common_parser_gui(meta_title)


def common_parser_gui(meta_title):
    from gooey import Gooey, GooeyParser

    @Gooey(program_name="VHS decode")
    def common_parser_gui_inner(meta_title):
        parser = GooeyParser(description=meta_title)
        parser.add_argument(
            "infile",
            metavar="infile",
            type=str,
            help="source file",
            widget="FileChooser",
        )
        parser.add_argument(
            "outfile",
            metavar="outfile",
            type=str,
            help="source file",
            widget="FileSaver",
        )
        return common_parser_inner(parser, True)

    return common_parser_gui_inner(meta_title)


def common_parser_cli(meta_title):
    parser = argparse.ArgumentParser(description=meta_title)
    parser.add_argument("infile", metavar="infile", type=str, help="source file")
    parser.add_argument(
        "outfile", metavar="outfile", type=str, help="base name for destination files"
    )
    # help="Disable AGC (deprecated, already disabled by default)
    parser.add_argument(
        "--noAGC",
        dest="noAGC",
        action="store_true",
        default=argparse.SUPPRESS,
        help=argparse.SUPPRESS,
    )
    # help="Enable AGC"
    parser.add_argument(
        "--AGC", dest="AGC", action="store_true", default=False, help=argparse.SUPPRESS
    )
    return common_parser_inner(parser)


def common_parser_inner(parser, use_gui=False):
    parser.add_argument(
        "--system",
        metavar="system",
        type=str.upper,
        help="video system (overriden by individual options)",
        default="NTSC",
        choices=["PAL", "MPAL", "PALM", "NTSC", "MESECAM"],
    )
    file_options_group = parser.add_argument_group("File options")
    file_options_group.add_argument(
        "-s",
        "--start",
        metavar="start",
        type=int,
        default=0,
        help="rough jump to frame n of capture (default is 0)",
    )
    file_options_group.add_argument(
        "--start_fileloc",
        metavar="start_fileloc",
        type=float,
        default=-1,
        help="jump to precise sample # in the file",
    )
    file_options_group.add_argument(
        "-l",
        "--length",
        metavar="length",
        type=int,
        default=99999999,
        help="limit length to n frames",
    )
    input_format_group = parser.add_argument_group("Input format")
    input_format_group.add_argument(
        "-f",
        "--frequency",
        dest="inputfreq",
        metavar="FREQ",
        type=lddu.parse_frequency,
        default=None,
        help="RF sampling frequency in source file (default is 40MHz)",
    )
    input_format_group.add_argument(
        "--cxadc",
        dest="cxadc",
        action="store_true",
        default=False,
        help="Use cxadc input frequency (~28,63 Mhz)",
    )
    input_format_group.add_argument(
        "--cxadc3",
        dest="cxadc3",
        action="store_true",
        default=False,
        help="Use cxadc ten fsc input frequency (~35,79 Mhz)",
    )
    input_format_group.add_argument(
        "--10cxadc",
        dest="cxadc_tenbit",
        action="store_true",
        default=False,
        help="Use cxadc input frequency in ten bit mode (~14,31 Mhz)",
    )
    input_format_group.add_argument(
        "--10cxadc3",
        dest="cxadc3_tenbit",
        action="store_true",
        default=False,
        help="Use cxadc ten fsc input frequency in ten bit mode (~17,89 Mhz)",
    )
    parser.add_argument(
        "-t",
        "--threads",
        metavar="threads",
        type=int,
        default=4,
        help="number of CPU threads to use",
    )

    extra_filtering_group = parser.add_argument_group("Extra filtering")
    extra_filtering_group.add_argument(
        "--ct",
        "--chroma_trap",
        dest="chroma_trap",
        action="store_true",
        default=False,
        help="Enable filter to reduce chroma interference on luma.",
    )
    extra_filtering_group.add_argument(
        "--sl",
        "--sharpness",
        dest="sharpness",
        metavar="sharpness",
        type=int,
        default=0,
        help="Sharpness level (0~100)",
    )
    extra_filtering_group.add_argument(
        "--notch",
        dest="notch",
        metavar="notch",
        type=lddu.parse_frequency,
        default=None,
        help="Center frequency of optional notch filter on rf and chroma.",
    )
    extra_filtering_group.add_argument(
        "--notch_q",
        dest="notch_q",
        metavar="notch_q",
        type=float,
        default=10.0,
        help="Q factor for notch filter",
    )

    system_group = parser.add_argument_group("Video system options")
    add_argument_hidden_in_gui(
        system_group,
        use_gui,
        "-p",
        "--pal",
        dest="pal",
        action="store_true",
        help="source is in PAL format",
    )
    add_argument_hidden_in_gui(
        system_group,
        use_gui,
        "-n",
        "--ntsc",
        dest="ntsc",
        action="store_true",
        help="source is in NTSC format",
    )
    add_argument_hidden_in_gui(
        system_group,
        use_gui,
        "--pm",
        "--palm",
        dest="palm",
        action="store_true",
        help="source is in PAL-M format",
    )
    system_group.add_argument(
        "--NTSCJ",
        dest="ntscj",
        action="store_true",
        help="source is in NTSC-J (IRE 0 black) format",
    )
    debug_group = parser.add_argument_group("Debug options")
    debug_group.add_argument(
        "--debug",
        dest="debug",
        action="store_true",
        default=False,
        help="Set log legel to DEBUG.",
    )
    return parser, debug_group


def select_sample_freq(args):
    sample_freq = (
        CXADC_FREQ
        if args.cxadc
        else CXADC_FREQ_HIGH
        if args.cxadc3
        else CXADC_TENBIT_FREQ
        if args.cxadc_tenbit
        else CXADC_TENBIT_FREQ_HIGH
        if args.cxadc3_tenbit
        else args.inputfreq
    )
    return sample_freq


def select_system(args):

    if args.pal:
        system = "PAL"
    elif args.palm:
        system = "MPAL"
    elif args.ntsc:
        system = "NTSC"
    elif args.system:
        system = args.system
        if system == "PALM":
            system = "MPAL"
    else:
        system = "NTSC"

    if args.pal and args.ntsc:
        print("ERROR: Can only be PAL or NTSC")
        exit(1)

    if args.palm and args.pal:
        print("ERROR: Can only be PAL-M or PAL")
        exit(1)

    if args.palm and args.ntsc:
        print("ERROR: Can only be PAL-M or NTSC")
        exit(1)

    return system


def get_basics(args):
    return args.infile, args.outfile, args.start, args.length


def get_rf_options(args):
    rf_options = {
        "chroma_trap": args.chroma_trap,
        "sharpness": args.sharpness,
        "notch": args.notch,
        "notch_q": args.notch_q,
    }
    return rf_options


def get_extra_options(args, checkagc=False):
    extra_options = {
        "useAGC": False,
        # Only used for ld, but could maybe be used for vhs too.
        "deemp_coeff": (0, 0),
    }
    if checkagc:
        extra_options["useAGC"]: args.AGC and not args.noAGC
    return extra_options
