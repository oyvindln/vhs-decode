#!/usr/bin/env python3

from setuptools import setup
import os

# Release/performance safety: default Rust extension builds to cargo's
# release profile unless a caller explicitly overrides it.
os.environ.setdefault("SETUPTOOLS_RUST_CARGO_PROFILE", "release")

setup(
    packages=[
        "lddecode",
        "vhsdecode",
        "vhsdecode/addons",
        "vhsdecode/format_defs",
        "cvbsdecode",
        "vhsdecode/hifi",
        "filter_tune",
    ],
    # TODO: should be done in pyproject.toml but did not find any way
    # of including without making them modules.
    scripts=[
        "ld-cut",
        "scripts/cx-expander",
        "decode.py",
    ],
)
