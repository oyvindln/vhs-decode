
from setuptools import setup
from Cython.Build import cythonize
import numpy

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
    ],
    # TODO: should be done in pyproject.toml but did not find any way
    # of including without making them modules.
    scripts=[
        'ld-cut',
        'cx-expander',
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
    ext_modules=cythonize(["vhsdecode/*.pyx"], language_level=3),
    # Needed for using numpy in cython.
    include_dirs=[numpy.get_include()],
    # These are just the minimal runtime dependencies for the Python scripts --
    # see the documentation for the full list of dependencies.
    provides=["lddecode"],
    requires=["matplotlib", "numba", "numpy", "scipy", "Cython"],
)
