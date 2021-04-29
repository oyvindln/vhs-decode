import argparse
import lddecode.utils as lddu

CXADC_FREQ = (8 * 315.0) / 88.0  # 28.636363636
CXADC_FREQ_HIGH = 3150.0 / 88.0  # 35.795454545
CXADC_TENBIT_FREQ = (8 * 315.0) / 88.0 / 2.0  # 14.318181818
CXADC_TENBIT_FREQ_HIGH = 3150.0 / 88.0 / 2.0  # 17.897727272


def common_parser(meta_title):
    parser = argparse.ArgumentParser(description=meta_title)
    parser.add_argument("infile", metavar="infile", type=str, help="source file")
    parser.add_argument(
        "outfile", metavar="outfile", type=str, help="base name for destination files"
    )
    parser.add_argument(
        "-s",
        "--start",
        metavar="start",
        type=int,
        default=0,
        help="rough jump to frame n of capture (default is 0)",
    )
    parser.add_argument(
        "--start_fileloc",
        metavar="start_fileloc",
        type=float,
        default=-1,
        help="jump to precise sample # in the file",
    )
    parser.add_argument(
        "-l",
        "--length",
        metavar="length",
        type=int,
        default=99999999,
        help="limit length to n frames",
    )
    parser.add_argument(
        "-p", "--pal", dest="pal", action="store_true", help="source is in PAL format"
    )
    parser.add_argument(
        "-n",
        "--ntsc",
        dest="ntsc",
        action="store_true",
        help="source is in NTSC format",
    )
    parser.add_argument(
        "-pm",
        "--palm",
        dest="palm",
        action="store_true",
        help="source is in PAL-M format",
    )
    parser.add_argument(
        "-t",
        "--threads",
        metavar="threads",
        type=int,
        default=1,
        help="number of CPU threads to use",
    )
    parser.add_argument(
        "-f",
        "--frequency",
        dest="inputfreq",
        metavar="FREQ",
        type=lddu.parse_frequency,
        default=None,
        help="RF sampling frequency in source file (default is 40MHz)",
    )
    parser.add_argument(
        "--NTSCJ",
        dest="ntscj",
        action="store_true",
        help="source is in NTSC-J (IRE 0 black) format (untested)",
    )
    parser.add_argument(
        "--cxadc",
        dest="cxadc",
        action="store_true",
        default=False,
        help="Use cxadc input frequency (~28,63 Mhz)",
    )
    parser.add_argument(
        "--cxadc3",
        dest="cxadc3",
        action="store_true",
        default=False,
        help="Use cxadc ten fsc input frequency (~35,79 Mhz)",
    )
    parser.add_argument(
        "--10cxadc",
        dest="cxadc_tenbit",
        action="store_true",
        default=False,
        help="Use cxadc input frequency in ten bit mode (~14,31 Mhz)",
    )
    parser.add_argument(
        "--10cxadc3",
        dest="cxadc3_tenbit",
        action="store_true",
        default=False,
        help="Use cxadc ten fsc input frequency in ten bit mode (~17,89 Mhz)",
    )
    parser.add_argument(
        "--noAGC", dest="noAGC", action="store_true", default=False, help="Disable AGC"
    )
    parser.add_argument(
        "-ct",
        "--chroma_trap",
        dest="chroma_trap",
        action="store_true",
        default=False,
        help="Enable filter to reduce chroma interference on luma.",
    )
    parser.add_argument(
        "-sl",
        "--sharpness",
        metavar="sharpness",
        type=int,
        default=0,
        help="Sharpness level (0~100)",
    )
    parser.add_argument(
        "--notch",
        dest="notch",
        metavar="notch",
        type=lddu.parse_frequency,
        default=None,
        help="Center frequency of optional notch filter on rf and chroma.",
    )
    parser.add_argument(
        "--notch_q",
        dest="notch_q",
        metavar="notch_q",
        type=float,
        default=10.0,
        help="Q factor for notch filter",
    )
    return parser


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


def get_extra_options(args):
    extra_options = {
        "useAGC": not args.noAGC,
    }
    return extra_options
