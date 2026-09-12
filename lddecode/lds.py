#!/usr/bin/env python3
"""
ld-lds-converter-py - 10-bit to 16-bit .lds converter for ld-decode

SPDX-License-Identifier: GPL-3.0-or-later
SPDX-FileCopyrightText: 2018-2020 Simon Inns
SPDX-FileCopyrightText: 2026 ld-decode contributors

This is a Python reimplementation of the C++ ld-lds-converter tool, which used
to live in tools/ld-lds-converter and is now only shipped by ld-decode-tools.
The packing and unpacking are bit-identical to the C++ version, including its
truncate-towards-zero division when packing.

The command is deliberately named ld-lds-converter-py, following
ld-ldf-reader-py, so that it neither shadows nor collides with the C++
ld-lds-converter when both ld-decode and ld-decode-tools are installed.

The .lds format stores 4 consecutive 10-bit samples in 5 bytes:

    Unpacked:                 Packed:
    0: xxxx xx00 0000 0000    0: 0000 0000 0011 1111
    1: xxxx xx11 1111 1111    2: 1111 2222 2222 2233
    2: xxxx xx22 2222 2222    4: 3333 3333
    3: xxxx xx33 3333 3333

Unpacked samples are in the DdD 16-bit format: signed, centred on zero and
left-shifted by 6, i.e. sample = (tenbit - 512) * 64.

Note that lddecode.utils has its own numba-jitted unpacker (unpack_data_4_40)
used on the decoder's hot path; the two must agree on the format above.
"""

import argparse
import signal
import sys

import numpy as np

# .lds and s16 streams are little-endian regardless of host byte order.
S16 = np.dtype("<i2")

# 4 samples per 5 packed bytes
PACKED_GROUP = 5
UNPACKED_GROUP = 8

# Read size, a whole number of both packed and unpacked groups (20 MiB), which
# matches the buffer size the C++ tool used.
CHUNK_SIZE = 20 * 1024 * 1024

# RIFF WAV header for 40 kHz, mono, signed 16-bit.  The RIFF and data chunk
# sizes are left as 0xFFFFFFFF because the length is not known up front when
# streaming - only FlaCCL/flaldf (with --ignore-chunk-sizes) accepts this.
RIFF_HEADER = bytes.fromhex(
    "52494646FFFFFFFF57415645"
    "666D74201000000001000100409C00008858010002001000"
    "4C4953541A000000494E464F495346540E0000004C61766635382E32392E313030"
    "0064617461FFFFFFFF"
)


def unpack_samples(packed):
    """Unpack packed 10-bit .lds bytes into signed 16-bit samples.

    Any trailing bytes that do not form a complete 5-byte group are ignored.
    """
    groups = len(packed) // PACKED_GROUP
    indata = np.frombuffer(packed, dtype=np.uint8, count=groups * PACKED_GROUP)
    # uint16 so the shifts below cannot lose the high bits
    indata = indata.reshape(groups, PACKED_GROUP).astype(np.uint16)

    tenbit = np.empty((groups, 4), dtype=np.uint16)
    tenbit[:, 0] = (indata[:, 0] << 2) | (indata[:, 1] >> 6)
    tenbit[:, 1] = ((indata[:, 1] & 0x3F) << 4) | (indata[:, 2] >> 4)
    tenbit[:, 2] = ((indata[:, 2] & 0x0F) << 6) | (indata[:, 3] >> 2)
    tenbit[:, 3] = ((indata[:, 3] & 0x03) << 8) | indata[:, 4]

    # Convert to the DdD 16-bit format.  -512 * 64 .. 511 * 64 fits int16
    # exactly, but do the arithmetic in int32 so the shift never overflows.
    samples = (tenbit.astype(np.int32) - 512) << 6

    return samples.astype(S16).reshape(-1)


def pack_samples(samples):
    """Pack signed 16-bit samples into packed 10-bit .lds bytes.

    Any trailing samples that do not form a complete group of 4 are ignored.
    """
    groups = len(samples) // 4
    indata = np.asarray(samples[: groups * 4]).reshape(groups, 4).astype(np.int32)

    # The C++ tool divided by 64, which truncates towards zero, whereas a right
    # shift rounds towards -infinity.  Bias the negative samples first so the
    # output stays bit-identical for them.
    tenbit = np.where(indata < 0, indata + 63, indata) >> 6
    tenbit += 512

    packed = np.empty((groups, PACKED_GROUP), dtype=np.uint8)
    packed[:, 0] = (tenbit[:, 0] & 0x03FC) >> 2
    packed[:, 1] = ((tenbit[:, 0] & 0x0003) << 6) | ((tenbit[:, 1] & 0x03F0) >> 4)
    packed[:, 2] = ((tenbit[:, 1] & 0x000F) << 4) | ((tenbit[:, 2] & 0x03C0) >> 6)
    packed[:, 3] = ((tenbit[:, 2] & 0x003F) << 2) | ((tenbit[:, 3] & 0x0300) >> 8)
    packed[:, 4] = tenbit[:, 3] & 0x00FF

    return packed.reshape(-1)


class LdsWriter:
    """File-like sink that packs signed 16-bit samples into a .lds file.

    Only whole groups of 4 samples can be packed, so samples are buffered until
    a group is complete.  On close, a trailing partial group is padded with
    mid-scale samples rather than dropped, so no captured data is discarded.
    """

    def __init__(self, output):
        """Write to output, which is either a filename or an open binary file.

        A file object is left open on close, so that callers which are not
        writing to a real file - ld-compress -v hashes the packed stream rather
        than storing it - can supply a sink of their own.
        """
        if hasattr(output, "write"):
            self._file = output
            self._owns_file = False
        else:
            self._file = open(output, "wb")
            self._owns_file = True

        self._leftover = np.empty(0, dtype=S16)

    def write(self, samples):
        samples = np.asarray(samples, dtype=S16).reshape(-1)

        if self._leftover.size:
            samples = np.concatenate((self._leftover, samples))

        whole = (samples.size // 4) * 4
        # copy() so we do not pin the caller's buffer until the next write
        self._leftover = samples[whole:].copy()

        if whole:
            self._file.write(pack_samples(samples[:whole]).tobytes())

    def close(self):
        if self._leftover.size:
            padding = np.zeros(4 - self._leftover.size, dtype=S16)
            padded = np.concatenate((self._leftover, padding))
            self._file.write(pack_samples(padded).tobytes())
            self._leftover = np.empty(0, dtype=S16)

        if self._owns_file:
            self._file.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        self.close()
        return False


def unpack_stream(infile, outfile, riff=False):
    """Unpack a 10-bit .lds stream into a signed 16-bit stream."""
    if riff:
        outfile.write(RIFF_HEADER)

    return _convert_stream(
        infile,
        outfile,
        PACKED_GROUP,
        lambda buf: unpack_samples(buf).tobytes(),
    )


def pack_stream(infile, outfile):
    """Pack a signed 16-bit stream into a 10-bit .lds stream."""
    return _convert_stream(
        infile,
        outfile,
        UNPACKED_GROUP,
        lambda buf: pack_samples(np.frombuffer(buf, dtype=S16)).tobytes(),
    )


def _convert_stream(infile, outfile, group_size, convert):
    """Convert infile to outfile in chunks aligned to whole sample groups.

    Returns the number of trailing bytes that had to be discarded because the
    input did not end on a group boundary.
    """
    leftover = b""

    while True:
        chunk = infile.read(CHUNK_SIZE)
        if not chunk:
            break

        if leftover:
            chunk = leftover + chunk

        # Only convert whole groups, carrying any remainder into the next read
        usable = len(chunk) - (len(chunk) % group_size)
        leftover = chunk[usable:]

        if usable:
            outfile.write(convert(chunk[:usable]))

    return len(leftover)


def _version_string():
    try:
        from lddecode import __version__

        return __version__
    except ImportError:
        return "unknown"


def main(argv=None):
    """Main entry point"""
    parser = argparse.ArgumentParser(
        prog="ld-lds-converter-py",
        description="ld-lds-converter-py - 10-bit to 16-bit .lds converter for ld-decode\n\n"
        "(c)2018-2020 Simon Inns\n"
        "(c)2026 Python implementation\n"
        "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-i",
        "--input",
        metavar="file",
        help="Specify input laserdisc sample file (default is stdin)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="file",
        help="Specify output laserdisc sample file (default is stdout)",
    )
    parser.add_argument(
        "-u",
        "--unpack",
        action="store_true",
        help="Unpack 10-bit data into 16-bit (default)",
    )
    parser.add_argument(
        "-p", "--pack", action="store_true", help="Pack 16-bit data into 10-bit"
    )
    parser.add_argument(
        "-r",
        "--riff",
        action="store_true",
        help="Unpack 10-bit data into 16-bit with RIFF WAV headers "
        "(use this ONLY for FlaCCL)",
    )
    parser.add_argument(
        "-d", "--debug", action="store_true", help="Show debug information"
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Suppress warning messages"
    )
    parser.add_argument(
        "-v", "--version", action="version", version=_version_string()
    )

    args = parser.parse_args(argv)

    if args.unpack and args.pack:
        print("Specify only --unpack (-u) or --pack (-p) - not both!", file=sys.stderr)
        return 1

    if args.riff and not args.unpack:
        print("You can only write RIFF headers with --unpack (-u)", file=sys.stderr)
        return 1

    # A downstream process closing the pipe is normal, so die the way a C tool
    # would rather than raising BrokenPipeError out of a write.
    if hasattr(signal, "SIGPIPE"):
        signal.signal(signal.SIGPIPE, signal.SIG_DFL)

    infile = None
    outfile = None

    try:
        infile = open(args.input, "rb") if args.input else sys.stdin.buffer
        outfile = open(args.output, "wb") if args.output else sys.stdout.buffer

        if args.debug:
            print(
                "{} {} to {}".format(
                    "Packing" if args.pack else "Unpacking",
                    args.input if args.input else "stdin",
                    args.output if args.output else "stdout",
                ),
                file=sys.stderr,
            )

        if args.pack:
            discarded = pack_stream(infile, outfile)
        else:
            discarded = unpack_stream(infile, outfile, riff=args.riff)

        outfile.flush()
    except OSError as e:
        print("Error: {}".format(e), file=sys.stderr)
        return 1
    finally:
        if args.input and infile is not None:
            infile.close()
        if args.output and outfile is not None:
            outfile.close()

    if discarded and not args.quiet:
        group = UNPACKED_GROUP if args.pack else PACKED_GROUP
        print(
            "Warning: input did not end on a {}-byte sample group boundary; "
            "ignored {} trailing byte(s)".format(group, discarded),
            file=sys.stderr,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
