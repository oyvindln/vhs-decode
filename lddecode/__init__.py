#!/usr/bin/python3
# Initialisation for the lddecode package.

import os

# Read PEP-440 compliant version from version file
# Format: base_version[+local.identifiers]
# Examples: 7.2.0, 7.2.0+git.abc123, 7.2.0+branch.main.abc123.dirty
try:
    _version_file = os.path.join(os.path.dirname(__file__), "version")
    if os.path.exists(_version_file):
        with open(_version_file, 'r') as f:
            __version__ = f.read().strip()
    else:
        __version__ = "unknown"
except Exception:
    __version__ = "unknown"

__all__ = [
    "audio",
    "commpy_filters",
    "core",
    "efm_pll",
    "fdls",
    "fft8",
    "utils",
    "utils_logging",
    "utils_plotting",
    "__version__",
]
