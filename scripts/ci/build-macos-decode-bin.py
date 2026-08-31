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
_ensure_lddecode_version_file()

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
        "--add-data",
        "vhsdecode/format_defs:vhsdecode/format_defs",
        "--collect-data",
        "lddecode",
        "--add-data",
        "assets:assets",
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
