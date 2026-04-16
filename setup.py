#!/usr/bin/env python3

from setuptools import setup
import shutil
import os
import distutils.ccompiler
from distutils.extension import Extension
from Cython.Build import cythonize

# Uncomment to view C code generated from Cython files
# import Cython.Compiler.Options
# Cython.Compiler.Options.annotate = True

import numpy

compiler = distutils.ccompiler.new_compiler()

if compiler.compiler_type == "unix":
    if shutil.which("clang"):
        os.environ["CC"] = "clang"
        os.environ["CXX"] = "clang++"

    extra_compile_args=["-O3", "-flto"]
    extra_link_args=["-O3", "-flto"]
else:
    extra_compile_args=[]
    extra_link_args=[]

# Release/performance safety: default Rust extension builds to cargo's
# release profile unless a caller explicitly overrides it.
os.environ.setdefault("SETUPTOOLS_RUST_CARGO_PROFILE", "release")

setup(
    # name='ld-decode',
    # version='7',
    # description='Software defined LaserDisc decoder',
    # url='https://github.com/happycube/ld-decode',
    # keywords=['video', 'LaserDisc'],
    # classifiers=[
    #    'Environment :: Console',
    #    'Environment :: X11 Applications :: Qt',
    #    'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
    #    'Programming Language :: C++',
    #    'Programming Language :: Python :: 3',
    #    'Topic :: Multimedia :: Video :: Capture',
    # ],
    setup_requires=["cython"],
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
    # scripts=[
    #    'cx-expander',
    #    'ld-cut',
    #    'ld-decode',
    #    'scripts/ld-compress',
    #    'vhs-decode',
    #    'cvbs-decode',
    #    'hifi-decode',
    # ],
    ext_modules=cythonize([
        Extension(
            "vhsdecode.sync",
            ["vhsdecode/sync.pyx"],
            language_level=3,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        ),
        Extension(
            "vhsdecode.hilbert",
            ["vhsdecode/hilbert.pyx"],
            language_level=3,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        ),
        Extension(
            "vhsdecode.linear_filter",
            ["vhsdecode/linear_filter.pyx"],
            language_level=3,
            extra_compile_args=extra_compile_args,
            extra_link_args=extra_link_args
        )
    ]),
    # Needed for using numpy in cython.
    include_dirs=[numpy.get_include()],
    # These are just the minimal runtime dependencies for the Python scripts --
    # see the documentation for the full list of dependencies.
    provides=["lddecode"],
    requires=["matplotlib", "numba", "numpy", "scipy", "Cython"],
)
