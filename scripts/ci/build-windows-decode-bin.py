import os
from pathlib import Path
import PyInstaller.__main__
import PyQt6
from pyinstaller_versionfile import create_versionfile
# Release/performance safety: never default Windows release artifacts to a
# debug Rust profile when invoked locally outside CI.
os.environ.setdefault("SETUPTOOLS_RUST_CARGO_PROFILE", "release")

print("Building Windows binary")

# Make sure build directory exists since we place versionfile.txt there.
build_path = ".\\build"
if not os.path.isdir(build_path):
    os.makedirs(build_path)

create_versionfile(
    output_file="build\\versionfile.txt",
    product_name="decode",
    original_filename="decode.exe",
    file_description="Software defined decoder",
    # version=version,
)


def _pyqt_runtime_binaries() -> list[str]:
    pyqt_root = Path(PyQt6.__file__).resolve().parent
    qt_bin_candidates = (pyqt_root / "Qt6" / "bin", pyqt_root / "Qt" / "bin")

    qt_bin = None
    for candidate in qt_bin_candidates:
        if candidate.is_dir():
            qt_bin = candidate
            break

    if qt_bin is None:
        return []

    runtime_dlls = sorted(qt_bin.glob("Qt6*.dll"))
    runtime_dlls += [
        qt_bin / "libEGL.dll",
        qt_bin / "libGLESv2.dll",
        qt_bin / "opengl32sw.dll",
        qt_bin / "d3dcompiler_47.dll",
    ]

    extra_args: list[str] = []
    seen: set[str] = set()
    for dll_path in runtime_dlls:
        if not dll_path.is_file():
            continue

        dll_key = str(dll_path).lower()
        if dll_key in seen:
            continue

        seen.add(dll_key)
        extra_args.extend(["--add-binary", f"{dll_path};PyQt6"])

    return extra_args

PyInstaller.__main__.run(
    [
        "decode.py",
        "--collect-all",
        "vhsd_rust",
        "--add-data",
        "vhsdecode/format_defs;vhsdecode/format_defs",
        "--collect-data",
        "lddecode",
        "--collect-all",
        "PyQt6",
        "--collect-all",
        "numba",
        "--collect-all",
        "llvmlite",
        "--collect-all",
        "sounddevice",
        "--collect-all",
        "_sounddevice_data",
        *_pyqt_runtime_binaries(),
        "--hidden-import",
        "vhsdecode.decode_launcher",
        "--hidden-import",
        "vhsdecode.main",
        "--hidden-import",
        "cvbsdecode.main",
        "--hidden-import",
        "lddecode.main",
        "--hidden-import",
        "vhsdecode.hifi.main",
        "--hidden-import",
        "filter_tune.filter_tune",
        "--hidden-import",
        "numba._dispatcher",
        "--hidden-import",
        "numba.core.runtime._nrt_python",
        "--icon",
        "assets\\icons\\vhs-decode.ico",
        "--version-file",
        "build\\versionfile.txt",
        "--onefile",
        "--name",
        "decode",
    ]
)
