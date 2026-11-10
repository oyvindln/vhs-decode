import os
import subprocess
import sys
import PyInstaller.__main__
import PyInstaller.utils.osx as osxutils
import plistlib
from pathlib import Path
from shutil import move
# Release/performance safety: never default macOS release artifacts to a
# debug Rust profile when invoked locally outside CI.
os.environ.setdefault("SETUPTOOLS_RUST_CARGO_PROFILE", "release")
def _generate_build_version() -> str:
    version_script = Path("scripts/generate_version.py")
    if not version_script.is_file():
        return ""
    try:
        return subprocess.check_output([sys.executable, str(version_script)], text=True).strip()
    except Exception:
        return ""


def _ensure_lddecode_version_file() -> None:
    version_path = Path("lddecode/version")
    current_version = ""
    if version_path.is_file():
        current_version = version_path.read_text(encoding="utf-8").strip()

    generated_version = _generate_build_version()
    version = generated_version or current_version or "unknown"
    if current_version == version:
        print(f"Using {version_path} = {version}")
        return

    version_path.parent.mkdir(parents=True, exist_ok=True)
    version_path.write_text(f"{version}\n", encoding="utf-8")
    print(f"Wrote {version_path} = {version}")


print("Building macOS binary version")

# The Rust crates compiled into vhsd_rust require their licence texts to ship
# with the binaries.
subprocess.check_call([sys.executable, "scripts/generate_rust_licenses.py", "THIRD_PARTY_LICENSES_RUST.txt"])
_ensure_lddecode_version_file()


def _collect_static_ffmpeg_binaries() -> list[str]:
    """Pre-download ffmpeg/ffprobe via static_ffmpeg and return
    PyInstaller --add-binary args to bundle them.

    Without this, the bundled decode.app would try to download ffmpeg
    at runtime (which fails in offline/sandboxed environments).
    """
    extra_args: list[str] = []
    try:
        import static_ffmpeg
        static_ffmpeg.add_paths()
        from static_ffmpeg import run as sff_run
        ffmpeg_path, ffprobe_path = sff_run.get_or_fetch_platform_executables_else_raise()
        print(f"static_ffmpeg ffmpeg: {ffmpeg_path}")
        print(f"static_ffmpeg ffprobe: {ffprobe_path}")
        if ffmpeg_path and os.path.isfile(ffmpeg_path):
            extra_args += ["--add-binary", f"{ffmpeg_path}:static_ffmpeg"]
        if ffprobe_path and os.path.isfile(ffprobe_path):
            extra_args += ["--add-binary", f"{ffprobe_path}:static_ffmpeg"]
    except Exception as exc:
        print(f"WARN: could not pre-download static_ffmpeg binaries: {exc}")
    return extra_args


# --- Workaround for PyInstaller COLLECT duplicate framework symlinks --------
# Qt 6.12 introduced the QtQuickControls2Macos framework, and with
# `--collect-all PyQt6` PyInstaller (6.22.x) ends up collecting the same
# framework symlink twice. The second os.symlink() call in COLLECT.assemble()
# then aborts the build with:
#   FileExistsError: [Errno 17] File exists:
#     'Versions/Current/QtQuickControls2Macos' ->
#     '.../PyQt6/Qt6/lib/QtQuickControls2Macos.framework/QtQuickControls2Macos'
# Recreating an identical symlink is a no-op, so make os.symlink idempotent
# for the duration of the PyInstaller run. Conflicting symlinks (different
# target) still raise, so genuine problems are not masked.
_real_os_symlink = os.symlink


def _idempotent_symlink(src, dst, target_is_directory=False, *, dir_fd=None):
    try:
        _real_os_symlink(src, dst, target_is_directory=target_is_directory, dir_fd=dir_fd)
    except FileExistsError:
        if os.path.islink(dst) and os.readlink(dst) == src:
            return  # identical symlink already in place
        raise


os.symlink = _idempotent_symlink


PyInstaller.__main__.run(
    [
        "decode.py",
        "--collect-all",
        "vhsd_rust",
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
        "--collect-all",
        "static_ffmpeg",
        "--add-data",
        "vhsdecode/format_defs:vhsdecode/format_defs",
        "--collect-data",
        "lddecode",
        "--add-data",
        "assets:assets",
        "--add-data",
        "THIRD_PARTY_LICENSES_RUST.txt:.",
        *_collect_static_ffmpeg_binaries(),
        "--hidden-import",
        "vhsdecode.windows_bootstrap",
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
        "assets/icons/vhs-decode.icns",
        "--add-data",
        "lddecode/version:lddecode",
        # onefile + .app is deprecated on macOS in newer PyInstaller releases.
        "--onedir",
        "--windowed",
        "--name",
        "decode",
    ]
)
macos_dir = Path("dist/decode.app/Contents/MacOS")
target_binary = macos_dir / "decode"
legacy_binary = macos_dir / "vhs-decode"
if legacy_binary.exists():
    if target_binary.exists():
        target_binary.unlink()
    move(str(legacy_binary), str(target_binary))
elif not target_binary.exists():
    raise FileNotFoundError(f"Expected bundled binary at {target_binary} or {legacy_binary}")

with Path("dist/decode.app/Contents/Info.plist").open(mode="rb+") as file:
    plist = plistlib.load(file)

    # update binary location
    plist["CFBundleExecutable"] = "decode"
    #plist["CFBundleShortVersionString"] = version
    file.seek(0)
    file.write(plistlib.dumps(plist))
    file.truncate()

# re-sign
osxutils.sign_binary("dist/decode.app", deep=True)
