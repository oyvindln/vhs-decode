#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import logging
import os
import shutil
import shlex
import subprocess
import sys
import sysconfig
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from vhsdecode.drop_paths import extract_dropped_file_paths
try:
    from PyQt6.QtCore import QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QColor, QIcon, QPalette
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStyleFactory,
        QVBoxLayout,
        QWidget,
    )
    ALIGN_TOP = Qt.AlignmentFlag.AlignTop
except ImportError:
    from PyQt5.QtCore import QTimer, Qt, pyqtSignal
    from PyQt5.QtGui import QColor, QIcon, QPalette
    from PyQt5.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QLineEdit,
        QMessageBox,
        QPushButton,
        QSpinBox,
        QStyleFactory,
        QVBoxLayout,
        QWidget,
    )
    ALIGN_TOP = Qt.AlignTop


@dataclass(frozen=True)
class ToolSpec:
    label: str
    subcommand: str
    default_args: tuple[str, ...] = ()
    prefer_native_gui: bool = False
    notes: str = ""


TOOLS = [
    ToolSpec(
        label="hifi-decode (native GUI)",
        subcommand="hifi",
        default_args=("--gui",),
        prefer_native_gui=True,
        notes="Launches hifi-decode with --gui.",
    ),
    ToolSpec(
        label="Filter Tune (native GUI)",
        subcommand="filter-tune",
        prefer_native_gui=True,
        notes="Launches Filter Tune Qt frontend.",
    ),
    ToolSpec(
        label="vhs-decode (terminal)",
        subcommand="vhs",
        notes="Opens a terminal command for vhs-decode.",
    ),
    ToolSpec(
        label="cvbs-decode (terminal)",
        subcommand="cvbs",
        notes="Opens a terminal command for cvbs-decode.",
    ),
    ToolSpec(
        label="ld-decode (terminal)",
        subcommand="ld",
        notes="Opens a terminal command for ld-decode.",
    ),
]


DECODER_SUBCOMMANDS = {"vhs", "cvbs", "ld"}
SYSTEM_OPTIONS_DECODE = [
    "NTSC",
    "NTSCJ",
    "PAL",
    "PAL_M",
    "SECAM",
    "MESECAM",
    "405",
    "819",
    "NLINHA",
]
SYSTEM_OPTIONS_LD = ["NTSC", "NTSCJ", "PAL"]

# Tape format options are (menu label, CLI value)
TAPE_FORMATS_VHS = [
    ("VHS", "VHS"),
    ("VHSHQ", "VHSHQ"),
    ("SVHS", "SVHS"),
    ("SVHS_ET", "SVHS_ET"),
    ("UMATIC", "UMATIC"),
    ("UMATIC_HI", "UMATIC_HI"),
    ("βetamax", "BETAMAX"),
    ("βetamax HiFi", "BETAMAX_HIFI"),
    ("SUPERBETA", "SUPERBETA"),
    ("Video8", "VIDEO8"),
    ("Hi8", "HI8"),
    ("EIAJ", "EIAJ"),
    ("Quaduplex", "QUADRUPLEX"),
    ("Phlips VCR", "VCR"),
    ("Phlips VCR_LP", "VCR_LP"),
    ("SMPTE-C", "TYPEC"),
    ("SMPTE-B", "TYPEB"),
    ("VIDEO2000", "VIDEO2000"),
]
TAPE_FORMATS_CVBS = [("CVBS", "CVBS")]
TAPE_FORMATS_LD = [("LASERDISC", "LASERDISC")]
TAPE_FORMATS_BY_TOOL = {
    "vhs": TAPE_FORMATS_VHS,
    "cvbs": TAPE_FORMATS_CVBS,
    "ld": TAPE_FORMATS_LD,
}

# Supported tape speeds
TAPE_SPEEDS = ["SP", "LP", "EP", "SLP", "VP"]

# Filename keyword -> tape format CLI value. Order matters: more specific /
# rarer formats are checked first so "vhs" doesn't shadow "svhs" etc.
# Each entry is (list_of_keywords, tape_format_cli_value, tool_subcommand).
# tool_subcommand is None when the format is valid for the currently selected
# decoder and should not force a tool switch.
FILENAME_TAPE_FORMAT_HINTS: list[tuple[tuple[str, ...], str, Optional[str]]] = [
    (("svhs_et", "svhs-et", "s-vhs_et", "s-vhs-et"), "SVHS_ET", None),
    (("svhs", "s-vhs", "supervhs", "super-vhs"), "SVHS", None),
    (("vhshq", "vhs-hq", "vhs_hq"), "VHSHQ", None),
    (("vhs",), "VHS", None),
    (("umatic_hi", "umatic-hi", "umatic hi"), "UMATIC_HI", None),
    (("umatic", "u-matic"), "UMATIC", None),
    (("betamax_hifi", "betamax-hifi", "betamax hifi", "βetamax hifi"), "BETAMAX_HIFI", "hifi"),
    (("superbeta", "super-beta"), "SUPERBETA", None),
    (("betamax", "beta-max", "βetamax"), "BETAMAX", None),
    (("video8", "video-8"), "VIDEO8", None),
    (("hi8", "hi-8"), "HI8", None),
    (("eiaj",), "EIAJ", None),
    (("quaduplex", "quadruplex", "quad",), "QUADRUPLEX", None),
    (("vcr_lp", "vcr-lp", "vcr lp", "philips vcr_lp"), "VCR_LP", None),
    (("vcr", "philips vcr"), "VCR", None),
    (("smpte-c", "smpte c", "typec", "type-c"), "TYPEC", None),
    (("smpte-b", "smpte b", "typeb", "type-b"), "TYPEB", None),
    (("video2000", "video-2000"), "VIDEO2000", None),
    (("betacam", "beta-cam"), None, "hifi"),  # placeholder - hifi GUI handles it
]

# Filename keyword -> TV system combo label.
FILENAME_SYSTEM_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("ntscj", "ntsc-j", "ntsc_j"), "NTSCJ"),
    (("ntsc",), "NTSC"),
    (("pal_m", "pal-m", "palm"), "PAL_M"),
    (("mesecam",), "MESECAM"),
    (("secam",), "SECAM"),
    (("pal",), "PAL"),
]

# Filename keyword -> tape speed combo label.
FILENAME_TAPE_SPEED_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("slp",), "SLP"),
    (("ep",), "EP"),
    (("lp",), "LP"),
    (("vp",), "VP"),
    (("sp",), "SP"),
]

# Filename keyword -> hifi GUI format combo label (used when launching hifi).
# Mirrors vhsdecode.hifi.HifiUi auto-detect so launcher and hifi GUI agree.
FILENAME_HIFI_FORMAT_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("betamax",), "Betamax"),
    (("betacam",), "Betacam"),
    (("video8", "hi8", "hi-8"), "Video8/Hi8"),
    (("svhs", "s-vhs", "supervhs", "super-vhs", "vhs"), "VHS"),
]

TOOL_ENTRYPOINTS = {
    "hifi": "hifi-decode",
    "filter-tune": "filter-tune",
    "vhs": "vhs-decode",
    "cvbs": "cvbs-decode",
    "ld": "ld-decode",
}

# Mirrors vhsdecode.hifi.HifiUi.FileIODialogUI.OUTPUT_FILE_SUFFIX so the
# launcher can derive the same default output name the hifi GUI does without
# importing the Qt-based UI module here.
HIFI_OUTPUT_SUFFIX = "_HiFi_Decoded.flac"
HIFI_AUDIO_OUTPUT_EXTENSIONS = {".flac", ".wav"}


def _load_app_icon() -> QIcon:
    icon_dir = Path(__file__).resolve().parents[1] / "assets" / "icons"

    if sys.platform == "darwin":
        candidates = (
            icon_dir / "vhs-decode.icns",
            icon_dir / "vhs-decode.png",
        )
    elif os.name == "nt":
        candidates = (
            icon_dir / "vhs-decode.ico",
            icon_dir / "vhs-decode.png",
        )
    else:
        candidates = (
            icon_dir / "vhs-decode.png",
            icon_dir / "vhs-decode.ico",
        )

    for candidate in candidates:
        if candidate.is_file():
            icon = QIcon(str(candidate))
            if not icon.isNull():
                return icon

    return QIcon()


def _current_app_icon() -> QIcon:
    app = QApplication.instance()
    if app is not None:
        icon = app.windowIcon()
        if icon is not None and not icon.isNull():
            return icon
    return _load_app_icon()


def _split_user_args(extra_args: str, *, strict: bool = True) -> list[str]:
    if not extra_args.strip():
        return []
    try:
        parsed = shlex.split(extra_args, posix=os.name != "nt")
        if os.name == "nt":
            return [
                arg[1:-1]
                if len(arg) >= 2 and arg[0] == arg[-1] and arg[0] in {"\"", "'"}
                else arg
                for arg in parsed
            ]
        return parsed
    except ValueError:
        if strict:
            raise
        return [extra_args]

def _is_json_file_path(path: str) -> bool:
    return Path(path).suffix.lower() == ".json"


def _extract_dropped_file_paths(mime_data) -> list[str]:
    return extract_dropped_file_paths(mime_data)



def _candidate_script_directories() -> list[Path]:
    candidates: list[Path] = []
    scripts_path = sysconfig.get_path("scripts")
    if scripts_path:
        candidates.append(Path(scripts_path))

    exe_dir = Path(sys.executable).resolve().parent
    candidates.append(exe_dir)
    if os.name == "nt":
        candidates.append(exe_dir / "Scripts")
    else:
        candidates.append(exe_dir / "bin")

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        key = str(path.resolve(strict=False))
        if key not in seen:
            seen.add(key)
            deduped.append(path)
    return deduped

def _appimage_name_variants(command_name: str) -> list[str]:
    if os.name == "nt" or sys.platform == "darwin":
        return []

    base_name = command_name
    if base_name.lower().endswith(".appimage"):
        base_name = base_name[: -len(".appimage")]

    variants = [
        f"{base_name}.AppImage",
        f"{base_name}.appimage",
        f"{base_name}-x86_64.AppImage",
        f"{base_name}-x86_64.appimage",
    ]

    deduped: list[str] = []
    seen: set[str] = set()
    for variant in variants:
        if variant not in seen:
            seen.add(variant)
            deduped.append(variant)
    return deduped


def _resolve_command_binary(command_name: str) -> Optional[list[str]]:
    appimage_variants = _appimage_name_variants(command_name)
    candidate_names = [
        f"{command_name}.exe",
        command_name,
        f"{command_name}-script.py",
        f"{command_name}.py",
    ] + appimage_variants

    for directory in _candidate_script_directories():
        for candidate_name in candidate_names:
            candidate = directory / candidate_name
            if not candidate.exists() or not candidate.is_file():
                continue
            if candidate.suffix.lower() == ".py":
                return [sys.executable, str(candidate)]
            return [str(candidate)]
    path_candidates = [command_name, f"{command_name}.py"] + appimage_variants
    for candidate_name in path_candidates:
        on_path = shutil.which(candidate_name)
        if not on_path:
            continue
        if on_path.lower().endswith(".py"):
            return [sys.executable, on_path]
        return [on_path]
    return None


def _self_dispatch_executable() -> Optional[str]:
    # In AppImage runtime, APPIMAGE points to the user-invoked image path.
    appimage_path = os.environ.get("APPIMAGE", "").strip()
    if appimage_path:
        return appimage_path

    executable = Path(sys.executable).resolve(strict=False)
    executable_name = executable.name.lower()

    if getattr(sys, "frozen", False):
        return str(executable)

    if os.name != "nt" and sys.platform != "darwin":
        if executable_name.endswith(".appimage"):
            return str(executable)

    return None


def _decode_prefix_command() -> Optional[list[str]]:
    self_dispatch = _self_dispatch_executable()
    if self_dispatch is not None:
        # In bundled binaries/AppImage runtime, the executable is the decode dispatcher.
        return [self_dispatch]
    # Source/dev mode: run through decode.py so subcommands resolve consistently.
    decode_script = Path(__file__).resolve().parents[1] / "decode.py"
    if decode_script.is_file():
        return [sys.executable, str(decode_script)]

    # Installed mode fallback (common on Windows): resolve decode entrypoint from
    # the active Python scripts directory or PATH.
    return _resolve_command_binary("decode")


def _resolve_tool_entrypoint(tool: ToolSpec) -> Optional[list[str]]:
    command_name = TOOL_ENTRYPOINTS.get(tool.subcommand)
    if not command_name:
        return None
    return _resolve_command_binary(command_name)


def _build_decode_command(tool: ToolSpec, extra_args: str) -> list[str]:
    # Use direct decode dispatch whenever available; this keeps argument shape
    # consistent with manual CLI usage and avoids AppImage double-script paths.
    decode_prefix = _decode_prefix_command()
    if decode_prefix is not None:
        command = decode_prefix + [tool.subcommand] + list(tool.default_args)
    else:
        fallback_command = _resolve_tool_entrypoint(tool)
        if fallback_command is None:
            raise RuntimeError(
                "Could not locate decode dispatcher or per-tool entrypoints "
                f"for '{tool.subcommand}'."
            )
        command = fallback_command + list(tool.default_args)
    if extra_args:
        command += _split_user_args(extra_args)
    return command


def _build_basic_decoder_args(
    tool: ToolSpec,
    input_path: str,
    output_path: str,
    frequency: str,
    system: str,
    tape_format: str,
    tape_speed: str,
    threads: int,
) -> list[str]:
    if tool.subcommand == "hifi":
        args: list[str] = []
        input_path = input_path.strip()
        output_path = output_path.strip()
        if input_path:
            args.append(input_path)
            if output_path:
                # The launcher's Output base field is an extension-less base
                # name for the terminal decoders. The hifi GUI expects a full
                # output filename; derive the same suffixed name it would when
                # the user loads the input directly, unless the user already
                # typed an audio extension (.flac/.wav).
                out_ext = Path(output_path).suffix.lower()
                if out_ext not in HIFI_AUDIO_OUTPUT_EXTENSIONS:
                    output_path = f"{output_path}{HIFI_OUTPUT_SUFFIX}"
                args.append(output_path)
        if frequency.strip():
            args += ["-f", frequency.strip()]
        args += ["-t", str(threads)]
        if system == "PAL":
            args += ["--pal"]
        elif system == "NTSC":
            args += ["--ntsc"]
        return args
    if tool.subcommand not in DECODER_SUBCOMMANDS:
        return []

    args = [input_path, output_path]

    if frequency.strip():
        args += ["-f", frequency.strip()]

    args += ["-t", str(threads)]

    # Add tape format for VHS decoder
    if tool.subcommand == "vhs":
        args += ["--tape_format", tape_format]
        args += ["--tape_speed", tape_speed.lower()]

    if tool.subcommand in {"vhs", "cvbs"}:
        if system == "NTSCJ":
            args += ["--NTSCJ", "--system", "NTSC"]
        else:
            args += ["--system", system]
    elif tool.subcommand == "ld":
        if system == "PAL":
            args += ["--pal"]
        elif system == "NTSCJ":
            args += ["--NTSCJ"]

    return args


def _shell_join(parts: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in parts)

def _shell_join_windows(parts: list[str]) -> str:
    return subprocess.list2cmdline(parts)

def _shell_join_platform(parts: list[str]) -> str:
    return _shell_join_windows(parts) if os.name == "nt" else _shell_join(parts)


def _open_linux_terminal(shell_command: str) -> None:
    shell = os.environ.get("SHELL", "/bin/bash")
    shell_args = [shell, "-lc", shell_command]
    terminal_candidates: list[tuple[str, list[str]]] = [
        ("x-terminal-emulator", ["x-terminal-emulator", "-e", *shell_args]),
        ("gnome-terminal", ["gnome-terminal", "--", *shell_args]),
        ("kgx", ["kgx", "--", *shell_args]),
        ("konsole", ["konsole", "-e", *shell_args]),
        ("mate-terminal", ["mate-terminal", "--", *shell_args]),
        ("xfce4-terminal", ["xfce4-terminal", "--command", f"{shell} -lc {shlex.quote(shell_command)}"]),
        ("lxterminal", ["lxterminal", "-e", f"{shell} -lc {shlex.quote(shell_command)}"]),
        ("kitty", ["kitty", "--hold", *shell_args]),
        ("alacritty", ["alacritty", "-e", *shell_args]),
        ("xterm", ["xterm", "-hold", "-e", *shell_args]),
    ]

    for binary, command in terminal_candidates:
        if shutil.which(binary):
            subprocess.Popen(command)
            return

    raise RuntimeError(
        "Could not find a supported Linux terminal emulator (e.g. gnome-terminal, konsole, xterm)."
    )


def _open_terminal(command_parts: list[str], working_directory: Path) -> None:
    if os.name == "nt":
        # Direct argument-array launch for Windows to avoid cmd/start parsing
        # edge-cases and keep behavior equivalent to standard CLI invocation.
        creation_flags = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)
        subprocess.Popen(
            command_parts,
            cwd=str(working_directory),
            creationflags=creation_flags,
        )
        return

    command = _shell_join(command_parts)
    shell = os.environ.get("SHELL", "/bin/bash")
    shell_command = (
        f"cd {shlex.quote(str(working_directory))} && {command}; "
        "echo; "
        'echo "[decode-launcher] process finished with exit code $?"; '
        f"exec {shlex.quote(shell)} -l"
    )

    if sys.platform == "darwin":
        escaped = shell_command.replace("\\", "\\\\").replace('"', '\\"')
        subprocess.Popen(
            [
                "osascript",
                "-e",
                'tell application "Terminal" to activate',
                "-e",
                f'tell application "Terminal" to do script "{escaped}"',
            ]
        )
        return

    _open_linux_terminal(shell_command)


class FileDropLineEdit(QLineEdit):
    fileDropped = pyqtSignal(str)
    def __init__(self, *, json_only: bool = False, allow_json: bool = True):
        super().__init__()
        self._json_only = json_only
        self._allow_json = allow_json
        self.setAcceptDrops(True)

    def _accepts_path(self, path: str) -> bool:
        is_json = _is_json_file_path(path)
        if self._json_only and not is_json:
            return False
        if not self._allow_json and is_json:
            return False
        return True

    def _matching_path(self, dropped_paths: list[str]) -> Optional[str]:
        for path in dropped_paths:
            if self._accepts_path(path):
                return path
        return None

    def dragEnterEvent(self, event) -> None:
        dropped_paths = _extract_dropped_file_paths(event.mimeData())
        if self._matching_path(dropped_paths) is not None:
            event.acceptProposedAction()
            return
        if dropped_paths:
            event.ignore()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        dropped_paths = _extract_dropped_file_paths(event.mimeData())
        if self._matching_path(dropped_paths) is not None:
            event.acceptProposedAction()
            return
        if dropped_paths:
            event.ignore()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        dropped_paths = _extract_dropped_file_paths(event.mimeData())
        dropped_path = self._matching_path(dropped_paths)
        if dropped_path is None:
            if dropped_paths:
                event.ignore()
                return
            super().dropEvent(event)
            return
        self.setText(dropped_path)
        self.fileDropped.emit(dropped_path)
        event.acceptProposedAction()

class DecodeLauncherWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setAcceptDrops(True)
        self._tools = TOOLS
        self.setWindowTitle("Decode Launcher")
        icon = _current_app_icon()
        if icon is not None and not icon.isNull():
            self.setWindowIcon(icon)
        self.resize(720, 280)

        self.tool_combo = QComboBox()
        for tool in self._tools:
            self.tool_combo.addItem(tool.label)
        # Default to basic VHS decode flow.
        self.tool_combo.setCurrentIndex(2)

        self.note_label = QLabel("")
        self.note_label.setWordWrap(True)

        self.extra_args_edit = QLineEdit("")
        self.params_json_check = QCheckBox("Use params JSON file")
        self.params_json_label = QLabel("Params JSON")
        self.params_json_edit = FileDropLineEdit(json_only=True)
        self.params_json_edit.setPlaceholderText("Drop params .json file here")
        self.params_json_browse_button = QPushButton("Params JSON…")
        self.params_json_label.setVisible(False)
        self.params_json_edit.setVisible(False)
        self.params_json_browse_button.setVisible(False)
        self.force_terminal_check = QCheckBox("Force terminal launch")
        self.command_preview = QLineEdit("")
        self.command_preview.setReadOnly(True)
        self.input_edit = FileDropLineEdit(allow_json=False)
        self.input_edit.setPlaceholderText("Drop RF input file here")
        self.input_browse_button = QPushButton("Input…")
        self.output_edit = QLineEdit("")
        self.output_browse_button = QPushButton("Output…")
        self.frequency_edit = QLineEdit("40")
        self.system_combo = QComboBox()
        self.system_combo.addItems(SYSTEM_OPTIONS_DECODE)
        self.tape_format_combo = QComboBox()
        self._active_tape_format_options = list(TAPE_FORMATS_VHS)
        for label, _value in self._active_tape_format_options:
            self.tape_format_combo.addItem(label)
        self.tape_format_combo.setCurrentIndex(0)
        self.tape_speed_combo = QComboBox()
        self.tape_speed_combo.addItems(TAPE_SPEEDS)
        self.tape_speed_combo.setCurrentText("SP")
        self.threads_spin = QSpinBox()
        self.threads_spin.setRange(1, 64)
        self.threads_spin.setValue(4)
        self._output_manually_set = False
        self._last_decoder_tbc_path: Optional[Path] = None
        self._has_launched_decoder = False
        self._hosted_launches: list[object] = []
        self._native_gui_warmup_started = False
        self._native_gui_warmup_done = threading.Event()
        self._force_terminal_by_tool: dict[str, bool] = {"hifi": True}
        self._active_tool_subcommand: Optional[str] = None

        self.launch_button = QPushButton("Launch selected tool")
        self.launch_tbc_tools_button = QPushButton("Launch tbc-tools / ld-analyse")
        self.close_button = QPushButton("Close")

        self._build_layout()
        self._wire_events()
        self._refresh_tool_state()

    def showEvent(self, event) -> None:
        super().showEvent(event)
        if not self._native_gui_warmup_started:
            self._native_gui_warmup_started = True
            QTimer.singleShot(250, self._start_native_gui_warmup)

    def _build_layout(self) -> None:
        root = QVBoxLayout()
        root.setAlignment(ALIGN_TOP)

        launch_group = QGroupBox("")
        launch_layout = QGridLayout()
        launch_group.setLayout(launch_layout)

        launch_layout.addWidget(QLabel("Tool"), 0, 0)
        launch_layout.addWidget(self.tool_combo, 0, 1, 1, 3)
        launch_layout.addWidget(QLabel("Input file"), 2, 0)
        launch_layout.addWidget(self.input_edit, 2, 1, 1, 2)
        launch_layout.addWidget(self.input_browse_button, 2, 3)

        launch_layout.addWidget(QLabel("Output base"), 3, 0)
        launch_layout.addWidget(self.output_edit, 3, 1, 1, 2)
        launch_layout.addWidget(self.output_browse_button, 3, 3)

        launch_layout.addWidget(QLabel("Frequency (MHz)"), 4, 0)
        launch_layout.addWidget(self.frequency_edit, 4, 1)
        launch_layout.addWidget(QLabel("TV system"), 4, 2)
        launch_layout.addWidget(self.system_combo, 4, 3)

        launch_layout.addWidget(QLabel("Tape format"), 5, 0)
        launch_layout.addWidget(self.tape_format_combo, 5, 1)
        launch_layout.addWidget(QLabel("Tape speed"), 5, 2)
        launch_layout.addWidget(self.tape_speed_combo, 5, 3)

        launch_layout.addWidget(QLabel("Threads"), 6, 0)
        launch_layout.addWidget(self.threads_spin, 6, 1)

        launch_layout.addWidget(QLabel("Extra arguments"), 7, 0)
        launch_layout.addWidget(self.extra_args_edit, 7, 1, 1, 3)
        launch_layout.addWidget(self.params_json_check, 8, 0, 1, 4)

        launch_layout.addWidget(self.params_json_label, 9, 0)
        launch_layout.addWidget(self.params_json_edit, 9, 1, 1, 2)
        launch_layout.addWidget(self.params_json_browse_button, 9, 3)

        launch_layout.addWidget(self.force_terminal_check, 10, 0, 1, 4)

        launch_layout.addWidget(QLabel("Terminal preview"), 11, 0)
        launch_layout.addWidget(self.command_preview, 11, 1, 1, 3)

        launch_layout.addWidget(self.note_label, 12, 0, 1, 4)

        action_row = QHBoxLayout()
        action_row.addWidget(self.launch_button)
        action_row.addWidget(self.launch_tbc_tools_button)
        action_row.addWidget(self.close_button)

        root.addWidget(launch_group)
        root.addLayout(action_row)
        self.setLayout(root)

    def _wire_events(self) -> None:
        self.tool_combo.currentIndexChanged.connect(self._refresh_tool_state)
        self.extra_args_edit.textChanged.connect(self._refresh_tool_state)
        self.params_json_check.toggled.connect(self._refresh_tool_state)
        self.params_json_edit.textChanged.connect(self._refresh_tool_state)
        self.params_json_edit.fileDropped.connect(self._on_params_json_dropped)
        self.input_edit.textChanged.connect(self._on_input_changed)
        self.output_edit.textChanged.connect(self._refresh_tool_state)
        self.output_edit.textEdited.connect(self._on_output_edited)
        self.frequency_edit.textChanged.connect(self._refresh_tool_state)
        self.system_combo.currentIndexChanged.connect(self._refresh_tool_state)
        self.tape_format_combo.currentIndexChanged.connect(self._refresh_tool_state)
        self.tape_speed_combo.currentIndexChanged.connect(self._refresh_tool_state)
        self.threads_spin.valueChanged.connect(self._refresh_tool_state)
        self.force_terminal_check.toggled.connect(self._on_force_terminal_toggled)
        self.params_json_browse_button.clicked.connect(self._browse_params_json_file)
        self.input_browse_button.clicked.connect(self._browse_input_file)
        self.output_browse_button.clicked.connect(self._browse_output_file)
        self.launch_button.clicked.connect(self._launch_selected_tool)
        self.launch_tbc_tools_button.clicked.connect(self._launch_tbc_tools)
        self.close_button.clicked.connect(self.close)

    def _on_force_terminal_toggled(self, checked: bool) -> None:
        tool = self._selected_tool()
        self._force_terminal_by_tool[tool.subcommand] = checked
        self._refresh_tool_state()

    def _selected_tool(self) -> ToolSpec:
        return self._tools[self.tool_combo.currentIndex()]

    def _track_hosted_launch(self, tracked_object: object, window: QWidget) -> None:
        self._hosted_launches.append(tracked_object)

        def _cleanup(*_args):
            try:
                self._hosted_launches.remove(tracked_object)
            except ValueError:
                pass

        window.destroyed.connect(_cleanup)

    def _delete_on_close_attribute(self):
        if hasattr(Qt, "WidgetAttribute"):
            return Qt.WidgetAttribute.WA_DeleteOnClose
        return Qt.WA_DeleteOnClose

    def _params_json_args(self, tool: ToolSpec, *, strict: bool) -> list[str]:
        if tool.subcommand != "vhs" or not self.params_json_check.isChecked():
            return []
        params_json = self.params_json_edit.text().strip()
        if not params_json:
            if strict:
                raise RuntimeError("Select a params JSON file.")
            return []
        return ["--params_file", params_json]

    def _start_native_gui_warmup(self) -> None:
        threading.Thread(
            target=self._warm_native_gui_modules,
            name="decode_launcher_native_gui_warmup",
            daemon=True,
        ).start()

    def _warm_native_gui_modules(self) -> None:
        try:
            # Only warm Filter Tune. Importing hifi here can trigger substantial
            # optional-runtime initialization side effects even when the user is
            # not launching hifi from the launcher.
            importlib.import_module("filter_tune.filter_tune")
        except Exception as exc:
            print(f"[decode-launcher] native GUI warmup skipped: {exc}")
        finally:
            self._native_gui_warmup_done.set()

    def _launch_filter_tune_in_process(self, extra: str) -> None:
        extra_args = _split_user_args(extra) if extra else []
        tape_format = extra_args[0] if extra_args else "VHS"

        from filter_tune.filter_tune import VHStune

        window = VHStune(tape_format, logging.getLogger("vhstune"))
        icon = _current_app_icon()
        if icon is not None and not icon.isNull():
            window.setWindowIcon(icon)
        window.setAttribute(self._delete_on_close_attribute(), True)
        window.show()
        pos = window.pos()
        if pos.x() < 0 or pos.y() < 0:
            window.move(0, 0)
        self._track_hosted_launch(window, window)

    def _launch_hifi_in_process(self, basic_args: list[str], extra: str) -> None:
        extra_args = list(basic_args)
        if extra:
            extra_args += _split_user_args(extra)

        from vhsdecode.hifi.main import launch_hosted_ui

        controller = launch_hosted_ui(extra_args, app=QApplication.instance())
        icon = _current_app_icon()
        if icon is not None and not icon.isNull():
            controller.window.setWindowIcon(icon)
        controller.window.setAttribute(self._delete_on_close_attribute(), True)
        self._track_hosted_launch(controller, controller.window)

    def _launch_native_gui_in_process(
        self, tool: ToolSpec, basic_args: list[str], extra: str
    ) -> bool:
        if tool.subcommand == "filter-tune":
            self._launch_filter_tune_in_process(extra)
            return True
        if tool.subcommand == "hifi":
            self._launch_hifi_in_process(basic_args, extra)
            return True
        return False

    def _terminal_preview_command(self, tool: ToolSpec) -> str:
        base_cmd = _build_decode_command(tool, "")
        basic_args = _build_basic_decoder_args(
            tool=tool,
            input_path=self.input_edit.text().strip(),
            output_path=self.output_edit.text().strip(),
            frequency=self.frequency_edit.text().strip(),
            system=self.system_combo.currentText(),
            tape_format=self._selected_tape_format_value(),
            tape_speed=self.tape_speed_combo.currentText(),
            threads=self.threads_spin.value(),
        )
        basic_args += self._params_json_args(tool, strict=False)
        extra = self.extra_args_edit.text().strip()
        if extra:
            basic_args += _split_user_args(extra, strict=False)
        return _shell_join_platform(base_cmd + basic_args)

    def _refresh_tool_state(self) -> None:
        tool = self._selected_tool()
        if self._active_tool_subcommand != tool.subcommand:
            self._active_tool_subcommand = tool.subcommand
            desired_force_terminal = self._force_terminal_by_tool.get(
                tool.subcommand,
                tool.subcommand == "hifi",
            )
            self.force_terminal_check.blockSignals(True)
            self.force_terminal_check.setChecked(desired_force_terminal)
            self.force_terminal_check.blockSignals(False)
        self._sync_system_options_for_tool(tool)
        self._sync_tape_format_options_for_tool(tool)
        params_json_allowed = tool.subcommand == "vhs"
        params_json_visible = params_json_allowed and self.params_json_check.isChecked()
        self.params_json_check.setEnabled(params_json_allowed)
        self.params_json_label.setVisible(params_json_visible)
        self.params_json_edit.setVisible(params_json_visible)
        self.params_json_browse_button.setVisible(params_json_visible)
        self.command_preview.setText(self._terminal_preview_command(tool))
        self.note_label.setText(tool.notes)
        decoder_selected = tool.subcommand in DECODER_SUBCOMMANDS
        hifi_selected = tool.subcommand == "hifi"
        file_inputs_enabled = decoder_selected or hifi_selected
        for widget in (
            self.input_edit,
            self.input_browse_button,
            self.output_edit,
            self.output_browse_button,
            self.frequency_edit,
            self.system_combo,
        ):
            widget.setEnabled(file_inputs_enabled)
        self.tape_format_combo.setEnabled(tool.subcommand == "vhs")
        self.tape_speed_combo.setEnabled(tool.subcommand == "vhs")
        self.threads_spin.setEnabled(decoder_selected or hifi_selected)

    def _sync_system_options_for_tool(self, tool: ToolSpec) -> None:
        current = self.system_combo.currentText()
        wanted = SYSTEM_OPTIONS_LD if tool.subcommand == "ld" else SYSTEM_OPTIONS_DECODE
        existing = [self.system_combo.itemText(i) for i in range(self.system_combo.count())]
        if existing == wanted:
            return
        self.system_combo.blockSignals(True)
        self.system_combo.clear()
        self.system_combo.addItems(wanted)
        self.system_combo.setCurrentText(current if current in wanted else "NTSC")
        self.system_combo.blockSignals(False)

    def _sync_tape_format_options_for_tool(self, tool: ToolSpec) -> None:
        current_value = self._selected_tape_format_value()
        wanted = list(TAPE_FORMATS_BY_TOOL.get(tool.subcommand, TAPE_FORMATS_VHS))
        if self._active_tape_format_options == wanted:
            return
        self._active_tape_format_options = wanted
        self.tape_format_combo.blockSignals(True)
        self.tape_format_combo.clear()
        for label, _value in wanted:
            self.tape_format_combo.addItem(label)
        if wanted:
            next_index = next(
                (i for i, (_label, value) in enumerate(wanted) if value == current_value),
                0,
            )
            self.tape_format_combo.setCurrentIndex(next_index)
        self.tape_format_combo.blockSignals(False)

    def _selected_tape_format_value(self) -> str:
        current_index = self.tape_format_combo.currentIndex()
        if 0 <= current_index < len(self._active_tape_format_options):
            return self._active_tape_format_options[current_index][1]
        return self.tape_format_combo.currentText()

    def _infer_default_output_base(self, input_path: str) -> str:
        if not input_path.strip():
            return ""
        path = Path(input_path.strip()).expanduser()
        parent = path.parent if path.parent.as_posix() != "." else Path.cwd()
        stem = path.stem or path.name
        return str(parent / stem)

    def _on_input_changed(self, value: str) -> None:
        if not self._output_manually_set:
            self.output_edit.setText(self._infer_default_output_base(value))
        self._auto_detect_config_from_filename(value)
        self._refresh_tool_state()

    def _auto_detect_config_from_filename(self, input_path: str) -> None:
        """Auto-set tool / tape format / tape speed / TV system from keywords
        found in the input filename. Only updates a control when a matching
        keyword is present; absent keywords leave the current selection alone.
        """
        if not input_path:
            return
        name_lower = os.path.basename(input_path).lower()
        if not name_lower:
            return

        # Tool / tape format: pick the first matching hint (order already
        # favors more-specific formats over generic ones).
        target_tool: Optional[str] = None
        target_format_value: Optional[str] = None
        for keywords, fmt_value, tool_sub in FILENAME_TAPE_FORMAT_HINTS:
            if any(kw in name_lower for kw in keywords):
                target_format_value = fmt_value
                target_tool = tool_sub
                break

        # If a hint wants the hifi tool (e.g. betamax/betacam), switch to it
        # only when the user hasn't already selected a terminal decoder that
        # also accepts the format. We never downgrade from hifi back to vhs.
        if target_tool == "hifi" and self._selected_tool().subcommand != "hifi":
            for i, tool in enumerate(self._tools):
                if tool.subcommand == "hifi":
                    self.tool_combo.setCurrentIndex(i)
                    break

        # Tape format combo: only set when the current tool exposes the value.
        if target_format_value is not None:
            self._set_tape_format_by_value(target_format_value)

        # TV system.
        for keywords, system_label in FILENAME_SYSTEM_HINTS:
            if any(kw in name_lower for kw in keywords):
                if self.system_combo.findText(system_label) >= 0:
                    self.system_combo.setCurrentText(system_label)
                break

        # Tape speed.
        for keywords, speed_label in FILENAME_TAPE_SPEED_HINTS:
            if any(kw in name_lower for kw in keywords):
                if self.tape_speed_combo.findText(speed_label) >= 0:
                    self.tape_speed_combo.setCurrentText(speed_label)
                break

    def _set_tape_format_by_value(self, value: str) -> None:
        for i, (_label, fmt_value) in enumerate(self._active_tape_format_options):
            if fmt_value == value:
                self.tape_format_combo.setCurrentIndex(i)
                return

    def _on_output_edited(self, value: str) -> None:
        self._output_manually_set = bool(value.strip())


    def _on_params_json_dropped(self, dropped_path: str) -> None:
        if not _is_json_file_path(dropped_path):
            return
        if not self.params_json_check.isChecked():
            self.params_json_check.setChecked(True)

    def _split_window_drop_paths(self, mime_data) -> tuple[Optional[str], Optional[str]]:
        input_path: Optional[str] = None
        params_json_path: Optional[str] = None
        for dropped_path in _extract_dropped_file_paths(mime_data):
            if _is_json_file_path(dropped_path):
                if params_json_path is None:
                    params_json_path = dropped_path
            elif input_path is None:
                input_path = dropped_path
            if input_path and params_json_path:
                break
        return input_path, params_json_path

    def dragEnterEvent(self, event) -> None:
        input_path, params_json_path = self._split_window_drop_paths(event.mimeData())
        if input_path or params_json_path:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        input_path, params_json_path = self._split_window_drop_paths(event.mimeData())
        if input_path or params_json_path:
            event.acceptProposedAction()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event) -> None:
        input_path, params_json_path = self._split_window_drop_paths(event.mimeData())
        handled = False
        if input_path:
            self.input_edit.setText(input_path)
            # _on_input_changed fires via the textChanged signal and runs the
            # filename auto-detect; no need to call it again here.
            handled = True
        if params_json_path:
            self.params_json_check.setChecked(True)
            self.params_json_edit.setText(params_json_path)
            handled = True
        if handled:
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    def _effective_working_directory(self) -> Path:
        input_path = self.input_edit.text().strip()
        if input_path:
            parent = Path(input_path).expanduser().parent
            if parent.is_dir():
                return parent.resolve()
        return Path(os.getcwd()).resolve()

    def _browse_input_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select input file",
            self.input_edit.text().strip() or str(self._effective_working_directory()),
        )
        if selected:
            self.input_edit.setText(selected)

    def _browse_output_file(self) -> None:
        selected, _ = QFileDialog.getSaveFileName(
            self,
            "Select output base name",
            self.output_edit.text().strip() or str(self._effective_working_directory()),
        )
        if selected:
            self._output_manually_set = True
            self.output_edit.setText(selected)

    def _browse_params_json_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Select params JSON file",
            self.params_json_edit.text().strip()
            or str(self._effective_working_directory()),
            "JSON files (*.json);;All files (*)",
        )
        if selected:
            self.params_json_edit.setText(selected)
    def _output_to_tbc_candidate(self, output_value: str) -> Optional[Path]:
        if not output_value.strip():
            return None

        output_path = Path(output_value.strip()).expanduser()
        if not output_path.is_absolute():
            output_path = self._effective_working_directory() / output_path
        output_path = output_path.resolve(strict=False)

        if output_path.suffix.lower() == ".tbc":
            return output_path
        return Path(str(output_path) + ".tbc")

    def _candidate_tbc_path(self) -> Optional[Path]:
        tbc_path = self._output_to_tbc_candidate(self.output_edit.text())
        decoder_selected = self._selected_tool().subcommand in DECODER_SUBCOMMANDS

        # Use explicit output path when available.
        if tbc_path is not None and (tbc_path.exists() or decoder_selected):
            return tbc_path

        # If no explicit output is currently set, fall back to the latest decoder
        # launch from this session to keep ld-analyse aligned with active decodes.
        if (
            tbc_path is None
            and self._has_launched_decoder
            and self._last_decoder_tbc_path is not None
        ):
            return self._last_decoder_tbc_path
        return None

    def _find_tbc_tools_executable(self) -> Optional[Path]:
        search_roots: list[Path] = [
            self._effective_working_directory(),
            Path(os.getcwd()).resolve(strict=False),
        ]
        if sys.platform == "darwin":
            search_roots.extend([Path("/Applications"), Path.home() / "Applications"])
        elif os.name == "nt":
            search_roots.extend(
                [
                    Path(os.environ.get("ProgramFiles", r"C:\Program Files")),
                    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")),
                    Path.home() / "AppData" / "Local" / "Programs",
                ]
            )

        if getattr(sys, "frozen", False):
            decode_exe = Path(sys.executable).resolve()
            search_roots.extend(
                [decode_exe.parent, decode_exe.parent.parent, decode_exe.parent.parent.parent]
            )
            # In app bundles, also search where the .app lives.
            for parent in decode_exe.parents:
                if parent.name == "MacOS" and parent.parent.name == "Contents":
                    app_dir = parent.parent.parent
                    search_roots.extend([app_dir.parent, app_dir.parent.parent])
                    break
        else:
            repo_root = Path(__file__).resolve().parents[1]
            search_roots.extend([repo_root, repo_root.parent, self._effective_working_directory()])

        # De-duplicate while preserving order.
        deduped_roots: list[Path] = []
        seen: set[str] = set()
        for root in search_roots:
            key = str(root.resolve()) if root.exists() else str(root)
            if key not in seen:
                seen.add(key)
                deduped_roots.append(root)
        def root_candidates(root: Path) -> list[Path]:
            if os.name == "nt":
                return [
                    root / "ld-analyse.exe",
                    root / "tbc-analyse.exe",
                    root / "tbc-tools.exe",
                    root / "tbc-tools" / "ld-analyse.exe",
                    root / "tbc-tools" / "tbc-analyse.exe",
                    root / "release" / "ld-analyse.exe",
                    root / "release" / "tbc-analyse.exe",
                ]
            if sys.platform == "darwin":
                return [
                    root / "tbc-tools.app" / "Contents" / "MacOS" / "ld-analyse",
                    root / "tbc-tools.app" / "Contents" / "MacOS" / "tbc-tools",
                    root / "ld-analyse.app" / "Contents" / "MacOS" / "ld-analyse",
                    root / "ld-analyse.app" / "Contents" / "MacOS" / "tbc-tools",
                    root / "ld-analyse",
                ]
            return [
                root / "ld-analyse",
                root / "tbc-tools.AppImage",
                root / "tbc-tools.appimage",
                root / "tbc-tools-x86_64.AppImage",
                root / "tbc-tools-x86_64.appimage",
                root / "tbc-tools" / "ld-analyse",
            ]

        def child_candidates(child: Path) -> list[Path]:
            if os.name == "nt":
                return [
                    child / "ld-analyse.exe",
                    child / "tbc-analyse.exe",
                    child / "tbc-tools.exe",
                    child / "tbc-tools" / "ld-analyse.exe",
                    child / "tbc-tools" / "tbc-analyse.exe",
                    child / "release" / "ld-analyse.exe",
                    child / "release" / "tbc-analyse.exe",
                ]
            if sys.platform == "darwin":
                if child.suffix.lower() == ".app":
                    return [
                        child / "Contents" / "MacOS" / "ld-analyse",
                        child / "Contents" / "MacOS" / "tbc-tools",
                    ]
                return [
                    child / "tbc-tools.app" / "Contents" / "MacOS" / "ld-analyse",
                    child / "tbc-tools.app" / "Contents" / "MacOS" / "tbc-tools",
                    child / "ld-analyse.app" / "Contents" / "MacOS" / "ld-analyse",
                    child / "ld-analyse",
                ]
            return [
                child / "ld-analyse",
                child / "tbc-tools.AppImage",
                child / "tbc-tools.appimage",
                child / "tbc-tools-x86_64.AppImage",
                child / "tbc-tools-x86_64.appimage",
                child / "tbc-tools" / "ld-analyse",
            ]

        for root in deduped_roots:
            candidates = root_candidates(root)

            # Also check one level down from the root (common unpack layouts).
            try:
                children = list(root.iterdir()) if root.exists() else []
            except OSError:
                children = []

            for child in children:
                if child.is_dir():
                    candidates.extend(child_candidates(child))

            for candidate in candidates:
                if candidate.exists() and candidate.is_file():
                    return candidate

        # Global path fallback for non-local installs.
        for name in (
            "ld-analyse",
            "tbc-analyse",
            "tbc-tools",
            "tbc-tools.AppImage",
            "tbc-tools.appimage",
            "tbc-tools-x86_64.AppImage",
            "tbc-tools-x86_64.appimage",
        ):
            on_path = shutil.which(name)
            if on_path:
                return Path(on_path)
        return None

    def _macos_app_bundle_for_binary(self, executable: Path) -> Optional[Path]:
        if sys.platform != "darwin":
            return None
        for parent in executable.resolve(strict=False).parents:
            if parent.suffix.lower() == ".app":
                return parent
        return None

    def _launch_tbc_tools(self) -> None:
        executable = self._find_tbc_tools_executable()
        if executable is None:
            QMessageBox.critical(
                self,
                "tbc-tools not found",
                "Could not find tbc-tools / ld-analyse near the decode binary or working folder.",
            )
            return
        tbc_candidate = self._candidate_tbc_path()
        app_bundle = self._macos_app_bundle_for_binary(executable)
        if app_bundle is not None:
            command = ["open", "-a", str(app_bundle)]
            if tbc_candidate is not None:
                command += ["--args", str(tbc_candidate)]
        else:
            command = [str(executable)]
            if tbc_candidate is not None:
                command.append(str(tbc_candidate))

        try:
            subprocess.Popen(command, cwd=str(self._effective_working_directory()))
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))
            return

    def _launch_selected_tool(self) -> None:
        tool = self._selected_tool()
        working_directory = self._effective_working_directory()

        if not working_directory.is_dir():
            QMessageBox.critical(
                self,
                "Invalid working directory",
                f"Directory does not exist:\n{working_directory}",
            )
            return

        extra = self.extra_args_edit.text().strip()

        try:
            command = _build_decode_command(tool, "")
            basic_args = _build_basic_decoder_args(
                tool=tool,
                input_path=self.input_edit.text().strip(),
                output_path=self.output_edit.text().strip(),
                frequency=self.frequency_edit.text().strip(),
                system=self.system_combo.currentText(),
                tape_format=self._selected_tape_format_value(),
                tape_speed=self.tape_speed_combo.currentText(),
                threads=self.threads_spin.value(),
            )

            if tool.subcommand in DECODER_SUBCOMMANDS:
                if not self.input_edit.text().strip():
                    QMessageBox.critical(self, "Missing input file", "Select an input file.")
                    return
                if not self.output_edit.text().strip():
                    QMessageBox.critical(self, "Missing output path", "Select an output base path.")
                    return
                self._last_decoder_tbc_path = self._output_to_tbc_candidate(
                    self.output_edit.text()
                )
                self._has_launched_decoder = self._last_decoder_tbc_path is not None

            command += basic_args
            command += self._params_json_args(tool, strict=True)
            if extra:
                command += _split_user_args(extra)

            if tool.prefer_native_gui and not self.force_terminal_check.isChecked():
                try:
                    if self._launch_native_gui_in_process(
                        tool, basic_args, extra
                    ):
                        return
                except Exception as hosted_exc:
                    print(f"[decode-launcher] hosted GUI launch fallback: {hosted_exc}")
                if os.name == "nt":
                    subprocess.Popen(command, cwd=str(working_directory))
                else:
                    subprocess.Popen(command, cwd=str(working_directory))
            else:
                _open_terminal(command, working_directory)
        except Exception as exc:
            QMessageBox.critical(self, "Launch failed", str(exc))
            return

def _apply_fusion_dark_mode(app: QApplication) -> None:
    fusion_style = QStyleFactory.create("Fusion")
    if fusion_style is not None:
        app.setStyle(fusion_style)
    else:
        app.setStyle("Fusion")

    role = QPalette.ColorRole if hasattr(QPalette, "ColorRole") else QPalette
    group = QPalette.ColorGroup if hasattr(QPalette, "ColorGroup") else QPalette

    palette = QPalette()
    palette.setColor(role.Window, QColor(53, 53, 53))
    palette.setColor(role.WindowText, QColor(225, 225, 225))
    palette.setColor(role.Base, QColor(35, 35, 35))
    palette.setColor(role.AlternateBase, QColor(53, 53, 53))
    palette.setColor(role.ToolTipBase, QColor(30, 30, 30))
    palette.setColor(role.ToolTipText, QColor(225, 225, 225))
    palette.setColor(role.Text, QColor(225, 225, 225))
    palette.setColor(role.Button, QColor(53, 53, 53))
    palette.setColor(role.ButtonText, QColor(225, 225, 225))
    palette.setColor(role.BrightText, QColor(255, 80, 80))
    palette.setColor(role.Highlight, QColor(42, 130, 218))
    palette.setColor(role.HighlightedText, QColor(20, 20, 20))
    palette.setColor(group.Disabled, role.Text, QColor(120, 120, 120))
    palette.setColor(group.Disabled, role.ButtonText, QColor(120, 120, 120))
    palette.setColor(group.Disabled, role.WindowText, QColor(120, 120, 120))
    app.setPalette(palette)
    app.setStyleSheet(
        "QToolTip { color: #e1e1e1; background-color: #2b2b2b; border: 1px solid #4a4a4a; }"
    )


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Decode Launcher (Qt) for starting decode/tools commands"
    )
    parser.parse_args(argv)

    app = QApplication(sys.argv)
    icon = _load_app_icon()
    if icon is not None and not icon.isNull():
        app.setWindowIcon(icon)
    _apply_fusion_dark_mode(app)
    window = DecodeLauncherWindow()
    window.show()
    return app.exec() if hasattr(app, "exec") else app.exec_()


if __name__ == "__main__":
    raise SystemExit(main())
