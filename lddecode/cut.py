#!/usr/bin/env python3
"""ld-cut - extract a sample area from raw RF laserdisc captures."""

import os
import signal
import sys
import argparse

import numpy as np

from lddecode.core import *
from lddecode.lds import LdsWriter
from lddecode.utils import *
from lddecode.utils_logging import *


def main(args=None):
    # Handle --version early before argparse requires positional arguments
    check_args = args if args is not None else sys.argv[1:]
    if "--version" in check_args or "-v" in check_args:
        from lddecode import __version__
        print(__version__)
        sys.exit(0)

    # Enable IO debug logging automatically in CI to help diagnose hangs when
    # ffmpeg fallback is used instead of ld-ldf-reader.
    if os.getenv("GITHUB_ACTIONS") and not os.getenv("LDDECODE_DEBUG_IO"):
        os.environ["LDDECODE_DEBUG_IO"] = "1"

    parser = argparse.ArgumentParser(
        description="Extract a sample area from raw RF laserdisc captures.  (Similar to ld-decode, except it outputs samples)"
    )
    parser.add_argument("infile", metavar="infile", type=str, help="source file")
    parser.add_argument(
        "outfile",
        metavar="outfile",
        type=str,
        help="destination file (recommended to use .lds suffix)",
    )

    parser.add_argument(
        "-s",
        "--start",
        metavar="start",
        type=float,
        default=0,
        help="rough jump to frame n of capture (default is 0)",
    )
    parser.add_argument(
        "-l",
        "--length",
        metavar="length",
        type=int,
        default=-1,
        help="limit length to n frames",
    )

    parser.add_argument(
        "-S",
        "--seek",
        metavar="seek",
        type=int,
        default=-1,
        help="seek to frame n of capture",
    )
    parser.add_argument(
        "-E", "--end", metavar="end", type=int, default=-1, help="cutting: last frame"
    )

    parser.add_argument(
        "-p", "--pal", dest="pal", action="store_true", help="source is in PAL format"
    )
    parser.add_argument(
        "-n", "--ntsc", dest="ntsc", action="store_true", help="source is in NTSC format"
    )
    parser.add_argument(
        "-C",
        "--ldf-compression-level",
        dest="ldfcomp",
        type=int,
        default=11,
        help="compression level for .ldf files",
    )

    parser.add_argument(
        "-F",
        "--fmpeg-options",
        dest="ffmpeg_options",
        type=str,
        default=None,
        help="custom ffmpeg format options"
    )

    args = parser.parse_args(args)

    filename = args.infile
    outname = args.outfile

    if outname == '-':
        outname = "/dev/stdout"

    if args.pal and args.ntsc:
        print("ERROR: Can only be PAL or NTSC")
        sys.exit(1)

    try:
        loader = make_loader(filename, None)
    except ValueError as e:
        print(e)
        sys.exit(1)

    makelds = True if outname[-3:] == "lds" else False
    makeldf = True if outname[-3:] == "ldf" else False

    system = "PAL" if args.pal else "NTSC"

    # Wrap the LDdecode creation so that the signal handler is not taken by sub-threads,
    # allowing SIGINT/control-C's to be handled cleanly
    original_sigint_handler = signal.signal(signal.SIGINT, signal.SIG_IGN)
    logger = init_logging(None)
    ldd = LDdecode(filename, None, loader, system=system, doDOD=False, _logger=logger)
    signal.signal(signal.SIGINT, original_sigint_handler)

    # note that endloc and startloc are in field #'s

    if args.seek != -1:
        startloc = ldd.seek(args.seek if args.start == 0 else args.start, args.seek)
        if startloc is None:
            print("ERROR: Seeking failed")
            sys.exit(1)
        elif startloc > 1:
            startloc -= 1
    else:
        startloc = args.start * 2

    if args.end != -1:
        endloc = ldd.seek(startloc, args.end)
        if endloc is None:
            print("ERROR: Seeking failed")
            sys.exit(1)
    elif args.length != -1:
        endloc = startloc + (args.length * 2) + 2
    else:
        # Set end location to well after any reasonable length so EOF is reached
        endloc = startloc + (2 ** 40)

    ldd.roughseek(startloc)
    startidx = int(ldd.fdoffset)

    ldd.roughseek(endloc)
    endidx = int(ldd.fdoffset)

    if args.ffmpeg_options is not None:
        process, fd = ffmpeg_pipe(outname, args.ffmpeg_options)
    elif makelds:
        # Pack in-process rather than piping through the ld-lds-converter-py
        # command, so .lds output does not depend on PATH
        fd = LdsWriter(outname)
    elif makeldf:
        process, fd = ldf_pipe(outname, args.ldfcomp)
    else:
        fd = open(outname, "wb")

    for i in range(startidx, endidx + 16384, 16384):
        l = endidx - i

        if l > 16384:
            l = 16384
        else:
            break

        data = ldd.freader(ldd.infile, i, l)
        if data is not None and len(data) == l:
            dataout = np.array(data, dtype=np.int16)
            fd.write(dataout)
        else:
            break

    fd.close()


if __name__ == "__main__":
    main(sys.argv[1:])
