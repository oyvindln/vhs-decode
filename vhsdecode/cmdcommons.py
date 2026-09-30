import argparse
import os
import shutil
import sys
import time
import traceback
from typing import Optional

import lddecode.utils as lddu

DDD_FREQ = 40
CXADC_FREQ = (8 * 315.0) / 88.0  # 28.636363636
CXADC_FREQ_HIGH = 3150.0 / 88.0  # 35.795454545
CXADC_TENBIT_FREQ = (8 * 315.0) / 88.0 / 2.0  # 14.318181818
CXADC_TENBIT_FREQ_HIGH = 3150.0 / 88.0 / 2.0  # 17.897727272

DEFAULT_THREADS = 4
DEFAULT_INPUT_FILENAME = ""
DEFAULT_OUTPUT_FILENAME = ""


def _parse_frequency(frequency: str) -> float:
    if frequency == "cxadc":
        return CXADC_FREQ
    else:
        return lddu.parse_frequency(frequency)


# size bytes to human-readable string
def sizeof_fmt(num: int, suffix: str = "B") -> str:
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return f"{num:3.1f} {unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f} Yi{suffix}"


# checks if the input file can be read
def test_input_file(input_file: Optional[str]) -> bool:
    if input_file == "-":
        return True

    if input_file is DEFAULT_INPUT_FILENAME:
        print("WARN: input file not specified")
        return False

    try:
        with open(input_file, "rb") as f:
            f.close()
            pass
    except FileNotFoundError:
        print(f"WARN: input file '{input_file}' not found")
        return False
    return True


# checks if the output file can be written
def test_output_file(output_file: Optional[str]) -> bool:

    # checks for free space on the output file directory
    output_file_dir = (
        "." if os.path.dirname(output_file) == "" else os.path.dirname(output_file)
    )
    if not os.access(output_file_dir, os.W_OK):
        print(f"Error: output file directory '{output_file_dir}' is not writable")
        return False

    # get the free space in the output file directory
    try:
        statvfs = os.statvfs(output_file_dir)
        free_space = statvfs.f_frsize * statvfs.f_bavail
        if free_space < 1024 * 1024 * 1024:
            print(
                f"WARN: output file directory {output_file_dir} has {sizeof_fmt(free_space)} free space"
            )
    except (AttributeError, BlockingIOError):
        pass

    try:
        with open(output_file, "ab") as f:
            f.close()
            pass
        # deletes output file if size is 0
        if os.path.getsize(output_file) == 0:
            os.remove(output_file)
    except FileNotFoundError:
        print(f"WARN: output file '{output_file}' not found")
        return False

    return True


# checks if the input file can be read and the output file can be written
def test_io(input_file: Optional[str], output_file: Optional[str]) -> bool:
    if not test_input_file(input_file):
        return False
    if not test_output_file(output_file):
        return False
    return True


class TestInputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if test_input_file(values):
            setattr(namespace, self.dest, values)


class TestOutputFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if test_output_file(values):
            setattr(namespace, self.dest, values)


def common_parser(meta_title, default_threads=DEFAULT_THREADS + 1):
    parser = argparse.ArgumentParser(
        description=meta_title, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument(
        "infile",
        metavar="infile",
        type=str,
        help="source file",
        nargs="?",
        default=DEFAULT_INPUT_FILENAME,
        action=TestInputFile,
    )
    parser.add_argument(
        "outfile",
        metavar="outfile",
        type=str,
        help="base name for destination files",
        nargs="?",
        default=DEFAULT_OUTPUT_FILENAME,
        action=TestOutputFile,
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
    return common_parser_inner(parser, default_threads=default_threads)


def common_parser_inner(parser, default_threads=DEFAULT_THREADS):
    parser.add_argument(
        "--system",
        metavar="system",
        type=str.upper,
        help="video system (overriden by individual options)",
        default="NTSC",
        choices=["PAL", "PAL_M", "PALM", "NTSC", "SECAM", "MESECAM", "405", "819", "NLINHA"],
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
    file_options_group.add_argument(
        "--overwrite",
        dest="overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing decode files.",
    )
    file_options_group.add_argument(
        "--write_db",
        dest="write_db",
        action="store_true",
        default=False,
        help="Enable sqlite output (slow on hdds), not finalized, option may change")
    input_format_group = parser.add_argument_group("Input format")
    input_format_group.add_argument(
        "-f",
        "--frequency",
        dest="inputfreq",
        metavar="FREQ",
        type=_parse_frequency,
        default=None,
        help="RF sampling frequency in source file (default is 40MHz)",
    )
    input_format_group.add_argument(
        "--cxadc",
        dest="cxadc",
        action="store_true",
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "-t",
        "--threads",
        metavar="threads",
        type=int,
        default=default_threads,
        help="number of CPU threads to use",
    )

    extra_filtering_group = parser.add_argument_group("Extra filtering")
    extra_filtering_group.add_argument(
        "--ct",
        "--chroma_trap",
        dest="chroma_trap",
        action="store_true",
        default=False,
        help="Enable luma filter that filters in some areas around the color frequency. This will soften the image and have a noticeable impact on decoding speed.",
    )
    extra_filtering_group.add_argument(
        "--sl",
        "--sharpness",
        dest="sharpness",
        metavar="sharpness",
        type=int,
        default=0,
        help="Enable a crude sharpness filter - Sharpness level (0~100)",
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
    system_group.add_argument(
        "-p",
        "--pal",
        dest="pal",
        action="store_true",
        help="source is in PAL format",
    )
    system_group.add_argument(
        "-n",
        "--ntsc",
        dest="ntsc",
        action="store_true",
        help="source is in NTSC format",
    )
    system_group.add_argument(
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
    parser.add_argument(
        "--wow_interpolation_method",
        type=str,
        default="linear",
        choices=["linear", "quadratic", "cubic"],
        help=(
            "Sets the type of interpolation spline used to correct wow."
            "\n  linear     [default]"
            "\n  quadratic"
            "\n  cubic"
        ),
    )
    debug_group.add_argument(
        "--skip_hsync_refine",
        dest="skip_hsync_refine",
        action="store_true",
        default=False,
        help="Skip refining line locations using hsync - less accurate line start detection but may avoid issues in some cases..",
    )
    return parser, debug_group


def select_sample_freq(args):
    sample_freq = (
        CXADC_FREQ
        if args.cxadc
        else args.inputfreq if args.inputfreq is not None else DDD_FREQ
    )
    return sample_freq


def select_system(args):
    if args.pal:
        system = "PAL"
    elif args.palm:
        system = "PAL_M"
    elif args.ntsc:
        system = "NTSC"
    elif args.system:
        system = args.system
        if system == "PALM":
            system = "PAL_M"
    else:
        system = "NTSC"

    if args.pal and args.ntsc:
        print("ERROR: Can only be PAL or NTSC")
        sys.exit(1)

    if args.palm and args.pal:
        print("ERROR: Can only be PAL-M or PAL")
        sys.exit(1)

    if args.palm and args.ntsc:
        print("ERROR: Can only be PAL-M or NTSC")
        sys.exit(1)

    return system


class IOArgsException(Exception):
    pass


def get_basics_or_exit(parser, args):
    """get_basics(), exiting with the help text and which of the paths failed on error."""
    try:
        return get_basics(args)
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


def check_overwrite(args, outname, extensions):
    """Refuse to clobber existing decode outputs unless --overwrite was given."""
    if args.overwrite:
        return
    conflicts = [outname + ext for ext in extensions if os.path.isfile(outname + ext)]
    if conflicts:
        print("Existing decode files found, remove them or run command with --overwrite")
        for conflict in conflicts:
            print("\t", conflict)
        sys.exit(1)


def _wait_for_disk_space(outname, cleanup):
    # 500 fields need around 675MB; leave margin for other writers so the disk can't fill
    # before the next check.
    limit = 1024 * 1024 * 1024 * 10
    output_dir = os.path.dirname(os.path.abspath(outname))
    try:
        if shutil.disk_usage(output_dir).free >= limit:
            return
        print(
            "\nLess than 10GB of free disk space is remaining, decoding paused. Decoding will resume once there is more space, or press Ctrl+C to exit.",
            file=sys.stderr,
        )
        while True:
            try:
                time.sleep(1)
                if shutil.disk_usage(output_dir).free >= limit:
                    print("\nDisk space available, resuming decode.", file=sys.stderr)
                    break
            except KeyboardInterrupt:
                print("\nTerminated, saving JSON and exiting")
                cleanup()
                sys.exit(1)
    except OSError:
        pass  # Ignore if we can't check disk space


def decode_fields(decoder, args, outname, req_frames):
    """Run the readfield loop, writing the JSON as fields accumulate.

    Exits on interrupt or decoder error and pauses when the output disk runs low.
    Returns the cleanup function that closes the JSON and the decoder.
    """
    jsondumper = lddu.JSONDumper(decoder, outname)

    def cleanup():
        jsondumper.close()
        decoder.close()

    done = False
    while not done and decoder.fields_written < (req_frames * 2):
        try:
            f = decoder.readfield()
        except KeyboardInterrupt:
            print("\nTerminated, saving JSON and exiting")
            cleanup()
            sys.exit(1)
        except Exception as err:
            print(
                "\nERROR - please paste the following into a bug report:",
                file=sys.stderr,
            )
            print("current sample:", decoder.fdoffset, file=sys.stderr)
            print("arguments:", args, file=sys.stderr)
            print("Exception:", err, " Traceback:", file=sys.stderr)
            traceback.print_tb(err.__traceback__)
            cleanup()
            sys.exit(1)

        if f is None:
            done = True
        else:
            f.prevfield = None

        if decoder.fields_written < 100 or ((decoder.fields_written % 500) == 0):
            jsondumper.write()
            _wait_for_disk_space(outname, cleanup)

    return cleanup


def get_basics(args):
    can_io = test_io(args.infile, args.outfile)
    if not "UI" in args:
        if not can_io:
            raise IOArgsException("Input/output file error")
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
        "debug": args.debug,
        "wow_level_adjust_smoothing": args.wow_level_adjust_smoothing,
        "wow_interpolation_method": args.wow_interpolation_method,
        "write_db": args.write_db
    }
    if checkagc:
        extra_options["useAGC"]: args.AGC and not args.noAGC
    return extra_options
