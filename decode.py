#!/usr/bin/env python3
import argparse
import sys
from importlib import import_module

# from vhsdecode.main import main


def print_options():
    print("Options are vhs, cvbs")


def main(argv):
    if len(argv) < 1:
        print_options()
        return

    # This seems a bit hacky but it works
    to_run = sys.argv.pop(1).lower()

    if to_run == "vhs":
        from vhsdecode.main import main as vhsmain

        vhsmain()
    elif to_run == "cvbs":
        from cvbsdecode.main import main as cvbsmain

        cvbsmain()
    elif to_run == 'ld':
        from lddecode.main import main as ldmain
        ldmain(sys.argv[1:])
    else:
        print_options()


if __name__ == "__main__":
    main(sys.argv[1:])
