#!/usr/bin/env python3

from setuptools import setup

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
