import os.path
import sys
import math
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np

from vhsdecode.hifi.utils import parse_flac_streaminfo
from vhsdecode.drop_paths import extract_dropped_file_paths

try:
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QDial,
        QCheckBox,
        QComboBox,
        QPushButton,
        QLineEdit,
        QFileDialog,
        QDialog,
        QMessageBox,
        QSpinBox,
        QFrame,
        QSizePolicy,
        QGridLayout,
    )
    from PyQt6 import QtGui, QtCore
except ImportError:
    from PyQt5.QtGui import QIcon
    from PyQt5.QtWidgets import (
        QApplication,
        QMainWindow,
        QWidget,
        QVBoxLayout,
        QHBoxLayout,
        QLabel,
        QDial,
        QCheckBox,
        QComboBox,
        QPushButton,
        QLineEdit,
        QFileDialog,
        QDialog,
        QMessageBox,
        QSpinBox,
        QFrame,
        QSizePolicy,
        QGridLayout,
    )
    from PyQt5 import QtGui, QtCore

from vhsdecode.hifi.constants import (
    DEFAULT_8MM_AUDIO_MODE,
    DEFAULT_8MM_DEEMPHASIS_TAU_1,
    DEFAULT_8MM_DEEMPHASIS_TAU_2,
    DEFAULT_8MM_EXPANDER_ATTACK_TAU,
    DEFAULT_8MM_EXPANDER_GAIN,
    DEFAULT_8MM_EXPANDER_HOLD_TAU,
    DEFAULT_8MM_EXPANDER_RATIO,
    DEFAULT_8MM_EXPANDER_RELEASE_TAU,
    DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS,
    DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION,
    DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_1,
    DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_2,
    DEFAULT_8MM_NR_DEEMPHASIS_TAU_1,
    DEFAULT_8MM_NR_DEEMPHASIS_TAU_2,
    DEFAULT_DEMOD,
    DEFAULT_DOC_MODE,
    DEFAULT_RESAMPLER_QUALITY,
    DEFAULT_SPECTRAL_NR_AMOUNT,
    DEFAULT_VHS_AUDIO_MODE,
    DEFAULT_VHS_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_DEEMPHASIS_TAU_2,
    DEFAULT_VHS_EXPANDER_ATTACK_TAU,
    DEFAULT_VHS_EXPANDER_GAIN,
    DEFAULT_VHS_EXPANDER_HOLD_TAU,
    DEFAULT_VHS_EXPANDER_RATIO,
    DEFAULT_VHS_EXPANDER_RELEASE_TAU,
    DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS,
    DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,
    DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,
    DEFAULT_VHS_NR_DEEMPHASIS_TAU_1,
    DEFAULT_VHS_NR_DEEMPHASIS_TAU_2,
    DEFAULT_ENV_DETECTION,
    DEMOD_HILBERT,
    DEMOD_QUADRATURE,
    audio_mode_to_ui,
    doc_mode_to_ui,
    expander_env_detection_to_ui,
    ui_to_audio_mode,
    ui_to_doc_mode,
    ui_to_expander_env_detection
)
from vhsdecode.hifi.HiFiDecode import (
    get_standard
)
_PLOT_DEEMPHASIS = None
_PLOT_EXPANDER = None


def _ensure_plot_helpers():
    global _PLOT_DEEMPHASIS, _PLOT_EXPANDER

    if _PLOT_DEEMPHASIS is None or _PLOT_EXPANDER is None:
        from vhsdecode.hifi.HiFiDecode import Deemphasis, Expander

        _PLOT_DEEMPHASIS = Deemphasis
        _PLOT_EXPANDER = Expander

    return _PLOT_DEEMPHASIS, _PLOT_EXPANDER

STOP_STATE = 0
PLAY_STATE = 1
PAUSE_STATE = 2
PREVIEW_STATE = 3

_SPINBOX_ARROW_CACHE = {}


def _spinbox_arrow_icon_path(direction: str, color: str = "#eeeeee") -> str:
    """Render a small arrow to a PNG for use in the stylesheet.

    Qt's stylesheet engine has no reliable way to draw triangles, so the
    up/down spin box arrows are rendered to temp images once per process.
    Requires a QApplication to exist.
    """
    cache_key = (direction, color)
    cached = _SPINBOX_ARROW_CACHE.get(cache_key)
    if cached and os.path.isfile(cached):
        return cached

    import tempfile

    width, height = 8, 5
    pixmap = QtGui.QPixmap(width, height)
    pixmap.fill(QtCore.Qt.GlobalColor.transparent)
    painter = QtGui.QPainter(pixmap)
    painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing, True)
    painter.setBrush(QtGui.QColor(color))
    painter.setPen(QtCore.Qt.PenStyle.NoPen)
    if direction == "up":
        points = [
            QtCore.QPoint(0, height),
            QtCore.QPoint(width, height),
            QtCore.QPoint(width // 2, 0),
        ]
    else:
        points = [
            QtCore.QPoint(0, 0),
            QtCore.QPoint(width, 0),
            QtCore.QPoint(width // 2, height),
        ]
    painter.drawPolygon(QtGui.QPolygon(points))
    painter.end()

    fd, path = tempfile.mkstemp(prefix=f"hifi_spin_{direction}_", suffix=".png")
    os.close(fd)
    pixmap.save(path, "PNG")
    _SPINBOX_ARROW_CACHE[cache_key] = path
    return path


class MainUIParameters:
    def __init__(self):
        self.volume: float = 1.0
        self.normalize = False
        self.expander_gain: float = DEFAULT_VHS_EXPANDER_GAIN
        self.expander_ratio: float = DEFAULT_VHS_EXPANDER_RATIO
        # UI label, not the internal constant (the combos and the
        # ui_to_* lookups in ui_parameters_to_decode_options expect labels)
        self.expander_env_detection = expander_env_detection_to_ui[
            DEFAULT_ENV_DETECTION
        ]
        self.expander_attack_tau: float = DEFAULT_VHS_EXPANDER_ATTACK_TAU
        self.expander_hold_tau: float = DEFAULT_VHS_EXPANDER_HOLD_TAU
        self.expander_release_tau: float = DEFAULT_VHS_EXPANDER_RELEASE_TAU
        self.expander_weighting_low_tau: float = DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1
        self.expander_weighting_high_tau: float = DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2
        self.expander_weighting_low_pass: float = DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS
        self.expander_weighting_low_pass_transition: float = DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION
        self.deemphasis_low_tau: float = DEFAULT_VHS_DEEMPHASIS_TAU_1
        self.deemphasis_high_tau: float = DEFAULT_VHS_DEEMPHASIS_TAU_2
        self.nr_deemphasis_low_tau: float = DEFAULT_VHS_NR_DEEMPHASIS_TAU_1
        self.nr_deemphasis_high_tau: float = DEFAULT_VHS_NR_DEEMPHASIS_TAU_2
        self.afe_left_carrier_deviation = 0
        self.afe_right_carrier_deviation = 0
        self.afe_left_carrier = 0
        self.afe_right_carrier = 0
        self.spectral_nr_amount = DEFAULT_SPECTRAL_NR_AMOUNT
        self.enable_expander: bool = True
        self.enable_deemphasis: bool = True
        self.automatic_fine_tuning: bool = True
        self.bias_guess: bool = False
        self.grc = False
        self.audio_sample_rate: int = 48000
        self.standard: str = "NTSC"
        self.format: str = "VHS"
        self.audio_mode: str = audio_mode_to_ui[DEFAULT_VHS_AUDIO_MODE]
        self.resampler_quality = DEFAULT_RESAMPLER_QUALITY
        self.demod_type: str = DEFAULT_DEMOD.capitalize()
        # in Hz: setValues() converts to MHz for display (matches the
        # decode_options "input_rate" value this field mirrors)
        self.input_sample_rate: float = 40.0e6
        self.input_file: str = ""
        self.output_file: str = ""
        # bool: setChecked() requires a bool (PyQt6 rejects the old "on" str)
        self.head_switching_interpolation = True
        self.doc = doc_mode_to_ui[DEFAULT_DOC_MODE]
        self.threads: int = cpu_count()
        self.preview_real_time: bool = False
        self.preview_only: bool = False


def decode_options_to_ui_parameters(decode_options):
    values = MainUIParameters()
    values.volume = decode_options["gain"]
    values.normalize = decode_options["normalize"]
    values.enable_expander = decode_options["enable_expander"]
    values.enable_deemphasis = decode_options["enable_deemphasis"]
    values.expander_gain = decode_options["expander_gain"]
    values.expander_ratio = decode_options["expander_ratio"]
    values.expander_env_detection = expander_env_detection_to_ui[decode_options["expander_env_detection"]]
    values.expander_attack_tau = decode_options["expander_attack_tau"]
    values.expander_hold_tau = decode_options["expander_hold_tau"]
    values.expander_release_tau = decode_options["expander_release_tau"]
    values.expander_weighting_low_tau = decode_options["expander_weighting_low_tau"]
    values.expander_weighting_high_tau = decode_options["expander_weighting_high_tau"]
    values.expander_weighting_low_pass = decode_options["expander_weighting_low_pass"]
    values.expander_weighting_low_pass_transition = decode_options["expander_weighting_low_pass_transition"]
    values.deemphasis_low_tau = decode_options["deemphasis_low_tau"]
    values.deemphasis_high_tau = decode_options["deemphasis_high_tau"]
    values.nr_deemphasis_low_tau = decode_options["nr_deemphasis_low_tau"]
    values.nr_deemphasis_high_tau = decode_options["nr_deemphasis_high_tau"]
    values.afe_left_carrier_deviation = decode_options["afe_left_carrier_deviation"]
    values.afe_right_carrier_deviation = decode_options["afe_right_carrier_deviation"]
    values.afe_left_carrier = decode_options["afe_left_carrier"]
    values.afe_right_carrier = decode_options["afe_right_carrier"]
    values.spectral_nr_amount = decode_options["spectral_nr_amount"]
    values.automatic_fine_tuning = decode_options["auto_fine_tune"]
    values.bias_guess = decode_options["bias_guess"]
    values.audio_sample_rate = decode_options["audio_rate"]
    values.standard = "PAL" if decode_options["standard"] == "p" else "NTSC"
    values.format = "VHS" if decode_options["format"] == "vhs" else "Video8/Hi8"
    values.audio_mode = audio_mode_to_ui[decode_options["mode"]]
    values.resampler_quality = decode_options["resampler_quality"]
    values.demod_type = decode_options["demod_type"].capitalize()
    values.input_sample_rate = decode_options["input_rate"]
    values.input_file = decode_options["input_file"]
    values.output_file = decode_options["output_file"]
    values.head_switching_interpolation = decode_options["head_switching_interpolation"]
    values.doc = doc_mode_to_ui[decode_options["doc"]]
    values.threads = decode_options.get("threads", values.threads)
    values.preview_real_time = decode_options.get("preview_real_time", False)
    values.preview_only = decode_options.get("preview_only", False)
    return values


def ui_parameters_to_decode_options(values: MainUIParameters):
    decode_options = {
        "input_rate": float(values.input_sample_rate) * 1e6,
        "standard": "p" if values.standard == "PAL" else "n",
        "format": "vhs" if values.format == "VHS" else "8mm",
        "demod_type": values.demod_type.lower(),
        "auto_fine_tune": values.automatic_fine_tuning,
        "bias_guess": values.bias_guess,
        "enable_expander": values.enable_expander,
        "enable_deemphasis": values.enable_deemphasis,
        "expander_gain": values.expander_gain,
        "expander_ratio": values.expander_ratio,
        "expander_env_detection": ui_to_expander_env_detection[values.expander_env_detection],
        "expander_attack_tau": values.expander_attack_tau,
        "expander_hold_tau": values.expander_hold_tau,
        "expander_release_tau": values.expander_release_tau,
        "expander_weighting_low_tau": values.expander_weighting_low_tau,
        "expander_weighting_high_tau": values.expander_weighting_high_tau,
        "expander_weighting_low_pass": values.expander_weighting_low_pass,
        "expander_weighting_low_pass_transition": values.expander_weighting_low_pass_transition,
        "deemphasis_low_tau": values.deemphasis_low_tau,
        "deemphasis_high_tau": values.deemphasis_high_tau,
        "nr_deemphasis_low_tau": values.nr_deemphasis_low_tau,
        "nr_deemphasis_high_tau": values.nr_deemphasis_high_tau,
        "afe_left_carrier_deviation": values.afe_left_carrier_deviation,
        "afe_right_carrier_deviation": values.afe_right_carrier_deviation,
        "afe_left_carrier": values.afe_left_carrier,
        "afe_right_carrier": values.afe_right_carrier,
        "spectral_nr_amount": values.spectral_nr_amount,
        "grc": values.grc,
        "audio_rate": values.audio_sample_rate,
        "gain": values.volume,
        "normalize": values.normalize,
        "input_file": values.input_file,
        "output_file": values.output_file,
        "input_format_override": None,
        "resampler_quality": values.resampler_quality,
        "head_switching_interpolation": values.head_switching_interpolation,
        "doc": ui_to_doc_mode[values.doc],
        "mode": ui_to_audio_mode[values.audio_mode],
        "threads": max(1, int(values.threads)),
        "preview_real_time": values.preview_real_time,
        "preview_only": values.preview_only,
    }
    return decode_options


class InputDialog(QDialog):
    def __init__(self, label: str, title: str, value=None, validator=None):
        super().__init__()
        self.validator = validator
        self.label = label
        self.value = value
        self.title = title
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle(self.title)

        layout = QHBoxLayout()

        self.input_line = QLineEdit(self)
        self.input_line.setText(str(self.value))

        if self.validator is not None:
            self.input_line.setValidator(self.validator)
        layout.addWidget(self.input_line)

        label = QLabel(self.label, self)
        layout.addWidget(label)

        ok_button = QPushButton("Accept", self)
        ok_button.clicked.connect(self.accept_user_input)
        layout.addWidget(ok_button)

        self.setLayout(layout)
        self.setStyleSheet(
            """
            QDialog {
                background-color: #333;
                color: #eee;
            }
            QLineEdit {
                background-color: #555;
                color: #eee;
                border: 1px solid #777;
            }
            QPushButton {
                background-color: #555;
                color: #eee;
                border: 1px solid #777;
            }
            QLabel {
                color: #eee;
            }
        """
        )

        # Set dialog size
        self.setFixedWidth(int(self.sizeHint().width()))
        self.setFixedHeight(self.sizeHint().height())

    def accept_user_input(self):
        self.value = self.input_line.text()
        self.accept()

    def get_input_value(self):
        self.exec()
        return self.value


class HifiUi(QMainWindow):
    def __init__(
        self,
        params: MainUIParameters,
        title: str = "HiFi Main Controls",
        main_layout_callback=None,
    ):
        super(HifiUi, self).__init__()

        self._transport_state = STOP_STATE

        # Set up the main window
        self.setWindowTitle(title)

        self.collapsableSections = []

        # Create central widget and layout
        self.central_widget = QWidget(self)
        self.setCentralWidget(self.central_widget)
        self.central_widget.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Minimum
        )

        self.main_layout: QHBoxLayout = QVBoxLayout(self.central_widget)
        # Qt6/macOS style metrics are larger than the historical v0.4.0 app
        # runtime; set explicit compact geometry so spacing matches the release
        # GUI visual density regardless of Qt style defaults.
        self.main_layout.setContentsMargins(4, 4, 4, 4)
        self.main_layout.setSpacing(2)
        main_layout_callback(self.main_layout)

        # Inner controls layout
        controls_layout = QVBoxLayout()
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(2)
        self.main_layout.addLayout(controls_layout)

        # Input sample rate selection
        input_options_section = self.build_input_options_section()
        controls_layout.addLayout(input_options_section)

        # Format options Analog Front End adjustments
        format_system_options = self.build_format_options_section()
        controls_layout.addLayout(format_system_options)

        # Demodulation options section
        demodulation_options = self.build_demodulation_options_section()
        controls_layout.addLayout(demodulation_options)

        # Deemphasis options
        expander_deemphasis_options = self.build_expander_deemphasis_section()
        controls_layout.addLayout(expander_deemphasis_options)

        # Noise Reduction Options section
        noise_reduction_options = self.build_noise_reduction_section()
        controls_layout.addLayout(noise_reduction_options)

        # Audio Processing Options section
        audio_processing_options = self.build_audio_processing_options_section()
        controls_layout.addLayout(audio_processing_options)

        # Transport controls
        transport_controls_layout = self.build_transport_controls()
        transport_controls_layout.setContentsMargins(0, 0, 0, 0)
        transport_controls_layout.setSpacing(6)
        self.main_layout.addLayout(transport_controls_layout)

        # Weighting / Deemphasis plot
        self.build_plot_window()

        # Apply dark theme with improved text visibility
        stylesheet = """
            QMainWindow {
                background-color: #333;
                color: #eee;
            }
            QDial, QLineEdit, QCheckBox, QComboBox, QPushButton, QSpinBox {
                background-color: #555;
                color: #eee;
                border: 1px solid #777;
            }
            QDial {
                min-width: 50px;
                max-width: 50px;
                min-height: 50px;
                max-height: 50px;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::item {
                background-color: #555;
                color: #eee;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                background-color: #666;
                border: 1px solid #777;
                width: 14px;
            }
            QSpinBox::up-button:hover, QSpinBox::down-button:hover {
                background-color: #777;
            }
            QSpinBox::up-arrow {
                image: url(__UP_ARROW__);
                width: 8px;
                height: 5px;
            }
            QSpinBox::down-arrow {
                image: url(__DOWN_ARROW__);
                width: 8px;
                height: 5px;
            }
            QLabel {
                color: #eee;
            }
        """
        up_arrow = _spinbox_arrow_icon_path("up").replace("\\", "/")
        down_arrow = _spinbox_arrow_icon_path("down").replace("\\", "/")
        self.setStyleSheet(
            stylesheet.replace("__UP_ARROW__", up_arrow).replace(
                "__DOWN_ARROW__", down_arrow
            )
        )
        self.change_button_color(self.stop_button, "#eee")
        # disables maximize button
        self.setWindowFlags(
            self.windowFlags() & ~QtCore.Qt.WindowType.WindowMaximizeButtonHint
        )
        self.resize_window()
        # sets default window icon
        self._set_status_window_icon("document-open")
        self.setValues(params)

        # update this at the end so the window width is calculated with everything expanded
        for collapsableSection in self.collapsableSections:
            collapsableSection.setDefaultCollapseState()

    def _fallback_window_icon(self):
        current_icon = self.windowIcon()
        if current_icon is not None and not current_icon.isNull():
            return current_icon

        app = QApplication.instance()
        if app is not None:
            app_icon = app.windowIcon()
            if app_icon is not None and not app_icon.isNull():
                return app_icon

        repo_icon = Path(__file__).resolve().parents[2] / "assets" / "icons" / "vhs-decode.png"
        if repo_icon.is_file():
            file_icon = QIcon(str(repo_icon))
            if not file_icon.isNull():
                return file_icon

        return QIcon()

    def _set_status_window_icon(self, theme_name: str):
        themed_icon = QIcon.fromTheme(theme_name)
        if themed_icon is not None and not themed_icon.isNull():
            self.setWindowIcon(themed_icon)
            return

        fallback_icon = self._fallback_window_icon()
        if fallback_icon is not None and not fallback_icon.isNull():
            self.setWindowIcon(fallback_icon)

    def resize_window(self, axis="hv"):
        self.central_widget.adjustSize()
        # disables resize
        if "h" in axis:
            self.setFixedWidth(max(320, int(self.minimumSizeHint().width()) - 150))
        # sets fixed height
        if "v" in axis:
            self.setFixedHeight(self.minimumSizeHint().height())

    def build_transport_controls(self):
        transport_controls_layout = QHBoxLayout()

        # Playback controls
        self.preview_button = QPushButton("Preview", self)
        self.play_button = QPushButton("▶", self)  # Play symbol
        self.pause_button = QPushButton("||", self)  # Pause symbol
        self.stop_button = QPushButton("■", self)  # Stop symbol
        max_button_height = max(
            self.preview_button.sizeHint().height(),
            self.play_button.sizeHint().height(),
            self.pause_button.sizeHint().height(),
            self.stop_button.sizeHint().height(),
        )
        # Keep transport controls compact/consistent across Qt styles.
        compact_height = min(max_button_height, self.fontMetrics().height() + 9)
        compact_height = max(compact_height, 21)
        self.preview_button.setFixedHeight(compact_height)
        self.play_button.setFixedHeight(compact_height)
        self.pause_button.setFixedHeight(compact_height)
        self.stop_button.setFixedHeight(compact_height)
        transport_controls_layout.addWidget(self.preview_button)
        transport_controls_layout.addWidget(self.play_button)
        transport_controls_layout.addWidget(self.pause_button)
        transport_controls_layout.addWidget(self.stop_button)

        self.preview_button.clicked.connect(self.on_preview_clicked)
        self.play_button.clicked.connect(self.on_play_clicked)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        return transport_controls_layout

    def build_input_options_section(self):
        layout = QVBoxLayout()

        input_options_frame = CollapsableSection(self, "Input Options")
        layout.addLayout(input_options_frame)

        # input sample rate
        input_samplerate_layout = QHBoxLayout()
        input_samplerate_label = QLabel("Input Sample Rate (MHz)")
        self.input_samplerate_combo = QComboBox(self)
        self.input_samplerate_combo.addItems(
            [
                "DdD (40)",
                "Clockgen (10)",
                "RTLSDR (8)",
                "cxadc (28.64)",
                "cxadc3 (35.8)",
                "10cxadc (14.32)",
                "10cxadc3 (17.9)",
                "Other",
            ]
        )
        self._input_combo_rates = [
            40.0,
            10.0,
            8.0,
            28.64,
            35.8,
            14.32,
            17.9,
        ]
        input_samplerate_layout.addWidget(input_samplerate_label)
        input_samplerate_layout.addWidget(self.input_samplerate_combo)
        self.input_samplerate_combo.setCurrentIndex(-1)
        self.input_samplerate_combo.currentIndexChanged.connect(
            self.on_input_samplerate_changed
        )
        input_options_frame.inner_layout.addLayout(input_samplerate_layout)

        # decode thread count
        threads_layout = QHBoxLayout()
        threads_label = QLabel("Decode Threads")
        self.threads_spinbox = QSpinBox(self)
        self.threads_spinbox.setMinimum(1)
        self.threads_spinbox.setMaximum(256)
        self.threads_spinbox.setValue(cpu_count())
        self.threads_spinbox.setToolTip(
            f"Number of CPU threads to use for decoding ({cpu_count()} available on this machine)"
        )
        threads_layout.addWidget(threads_label)
        threads_layout.addWidget(self.threads_spinbox)
        input_options_frame.inner_layout.addLayout(threads_layout)

        return layout

    def build_format_options_section(self):
        layout = QVBoxLayout()

        format_options_frame = CollapsableSection(self, "Format Options")
        layout.addLayout(format_options_frame)

        # standard option
        standard_layout = QHBoxLayout()
        standard_label = QLabel("Standard")
        self.standard_combo = QComboBox(self)
        self.standard_combo.addItems(["PAL", "NTSC"])
        standard_layout.addWidget(standard_label)
        standard_layout.addWidget(self.standard_combo)
        self.standard_combo.currentIndexChanged.connect(self.on_standard_change)
        format_options_frame.inner_layout.addLayout(standard_layout)

        # format option
        format_layout = QHBoxLayout()
        format_label = QLabel("Format")
        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["VHS", "Video8/Hi8"])
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)
        self.format_combo.currentIndexChanged.connect(self.on_format_change)
        format_options_frame.inner_layout.addLayout(format_layout)

        advanced_format_options_frame = CollapsableSection(
            self, "Advanced Format Options", default_collapsed=True
        )
        layout.addLayout(advanced_format_options_frame)

        # auto fine tune
        self.automatic_fine_tuning_checkbox = QCheckBox("Automatic fine tuning")
        self.automatic_fine_tuning_checkbox.setToolTip(
            "Automatically adjust bias during decode. Not applicable to Quadrature demodulation."
        )
        advanced_format_options_frame.inner_layout.addWidget(
            self.automatic_fine_tuning_checkbox
        )

        # bias guess
        self.bias_guess_checkbox = QCheckBox("Bias Guess")
        self.bias_guess_checkbox.setToolTip("Attempt to guess the carrier frequencies")
        advanced_format_options_frame.inner_layout.addWidget(self.bias_guess_checkbox)

        # left carrier adjustment
        afe_left_carrier_layout = QHBoxLayout()
        afe_left_carrier_spinbox_label = QLabel("Left Carrier (Hz)")
        self.afe_left_carrier_spinbox = QSpinBox(self)
        self.afe_left_carrier_spinbox.setGroupSeparatorShown(True)
        self.afe_left_carrier_spinbox.setMinimum(int(10e5))
        self.afe_left_carrier_spinbox.setMaximum(int(10e6))
        self.afe_left_carrier_spinbox.setSingleStep(100)
        self.afe_left_carrier_spinbox.setToolTip("Left carrier center frequency")
        afe_left_carrier_layout.addWidget(afe_left_carrier_spinbox_label)
        afe_left_carrier_layout.addWidget(self.afe_left_carrier_spinbox)
        advanced_format_options_frame.inner_layout.addLayout(afe_left_carrier_layout)

        # left vco deviation adjustment
        afe_left_carrier_deviation_layout = QHBoxLayout()
        afe_left_carrier_deviation_spinbox_label = QLabel("Left Carrier Deviation (Hz)")
        self.afe_left_carrier_deviation_spinbox = QSpinBox(self)
        self.afe_left_carrier_deviation_spinbox.setGroupSeparatorShown(True)
        self.afe_left_carrier_deviation_spinbox.setMinimum(int(10e3))
        self.afe_left_carrier_deviation_spinbox.setMaximum(int(10e5))
        self.afe_left_carrier_deviation_spinbox.setSingleStep(10)
        self.afe_left_carrier_deviation_spinbox.setToolTip(
            "Maximum frequency offset + or - from the center frequency"
        )
        afe_left_carrier_deviation_layout.addWidget(afe_left_carrier_deviation_spinbox_label)
        afe_left_carrier_deviation_layout.addWidget(self.afe_left_carrier_deviation_spinbox)
        advanced_format_options_frame.inner_layout.addLayout(afe_left_carrier_deviation_layout)

        # right carrier adjustment
        afe_right_carrier_layout = QHBoxLayout()
        afe_right_carrier_spinbox_label = QLabel("Right Carrier (Hz)")
        self.afe_right_carrier_spinbox = QSpinBox(self)
        self.afe_right_carrier_spinbox.setGroupSeparatorShown(True)
        self.afe_right_carrier_spinbox.setMinimum(int(10e5))
        self.afe_right_carrier_spinbox.setMaximum(int(10e6))
        self.afe_right_carrier_spinbox.setSingleStep(100)
        self.afe_right_carrier_spinbox.setToolTip("Right carrier center frequency")
        afe_right_carrier_layout.addWidget(afe_right_carrier_spinbox_label)
        afe_right_carrier_layout.addWidget(self.afe_right_carrier_spinbox)
        advanced_format_options_frame.inner_layout.addLayout(afe_right_carrier_layout)

        # right vco deviation adjustment
        afe_right_carrier_deviation_layout = QHBoxLayout()
        afe_right_carrier_deviation_spinbox_label = QLabel("Right Carrier Deviation (Hz)")
        self.afe_right_carrier_deviation_spinbox = QSpinBox(self)
        self.afe_right_carrier_deviation_spinbox.setGroupSeparatorShown(True)
        self.afe_right_carrier_deviation_spinbox.setMinimum(int(10e3))
        self.afe_right_carrier_deviation_spinbox.setMaximum(int(10e5))
        self.afe_right_carrier_deviation_spinbox.setSingleStep(10)
        self.afe_right_carrier_deviation_spinbox.setToolTip(
            "Maximum frequency offset + or - from the center frequency"
        )
        afe_right_carrier_deviation_layout.addWidget(afe_right_carrier_deviation_spinbox_label)
        afe_right_carrier_deviation_layout.addWidget(self.afe_right_carrier_deviation_spinbox)
        advanced_format_options_frame.inner_layout.addLayout(afe_right_carrier_deviation_layout)

        return layout

    def build_demodulation_options_section(self):
        layout = QVBoxLayout()

        demodulation_options_frame = CollapsableSection(
            self, "Demodulation Options", default_collapsed=True
        )
        layout.addLayout(demodulation_options_frame)

        # demodulation type option
        demod_type_layout = QHBoxLayout()
        demod_type_label = QLabel("FM Demodulation Type")
        self.demod_type_combo = QComboBox(self)
        self.demod_type_combo.addItems(
            [DEMOD_QUADRATURE.capitalize(), DEMOD_HILBERT.capitalize()]
        )
        demod_type_layout.addWidget(demod_type_label)
        demod_type_layout.addWidget(self.demod_type_combo)
        demodulation_options_frame.inner_layout.addLayout(demod_type_layout)

        return layout

    def build_noise_reduction_section(self):
        layout = QVBoxLayout()

        noise_reduction_options_frame = CollapsableSection(
            self, "Noise Reduction Options", default_collapsed=True
        )
        layout.addLayout(noise_reduction_options_frame)

        noise_reduction_options_layout = QHBoxLayout()

        # Volume dial and numeric textbox
        self.spectral_nr_amount_dial_control = DialControl(
            self, "Spectral NR", QtGui.QDoubleValidator(), 100, 0, 1
        )
        self.spectral_nr_amount_dial_control.dial.setToolTip('Uses "0" in Preview Mode')
        self.spectral_nr_amount_dial_control.textbox.setToolTip(
            'Uses "0" in Preview Mode'
        )

        noise_reduction_checkboxes_layout = QVBoxLayout()
        # Head Switching Interpolation checkbox
        self.head_switching_interpolation_checkbox = QCheckBox(
            "Head Switching Interpolation"
        )
        noise_reduction_checkboxes_layout.addWidget(
            self.head_switching_interpolation_checkbox
        )

        # Dropout compensation combo
        doc_layout = QHBoxLayout()
        doc_label = QLabel("Dropout Compensation")

        self.doc_combo = QComboBox(self)
        self.doc_combo.addItems(ui_to_doc_mode.keys())
        doc_layout.addWidget(doc_label)
        doc_layout.addWidget(self.doc_combo)
        noise_reduction_checkboxes_layout.addLayout(doc_layout)

        noise_reduction_options_layout.addWidget(
            self.spectral_nr_amount_dial_control, 1
        )
        noise_reduction_options_layout.addLayout(noise_reduction_checkboxes_layout, 1)
        noise_reduction_options_frame.inner_layout.addLayout(
            noise_reduction_options_layout
        )

        return layout

    def build_audio_processing_options_section(self):
        layout = QVBoxLayout()

        advanced_format_options_frame = CollapsableSection(self, "Audio Options")
        layout.addLayout(advanced_format_options_frame)

        # gain section
        gain_section_layout = QHBoxLayout()
        advanced_format_options_frame.inner_layout.addLayout(gain_section_layout)

        # Volume dial and numeric textbox
        self.volume_dial_control = DialControl(
            self, "Output Gain", QtGui.QDoubleValidator(), 100, 0, 2
        )
        gain_section_layout.addWidget(self.volume_dial_control, 1)

        # Normalize Checkbox
        normalize_layout = QHBoxLayout()
        self.normalize_checkbox = QCheckBox("Normalize")
        normalize_layout.addWidget(self.normalize_checkbox)
        gain_section_layout.addLayout(normalize_layout, 1)

        # Audio mode (mono L/mono R/stereo/stereo ms)
        audio_mode_layout = QHBoxLayout()
        audio_mode_label = QLabel("Output Channel Mode")
        self.audio_mode_combo = QComboBox(self)
        self.audio_mode_combo.addItems(ui_to_audio_mode.keys())
        audio_mode_layout.addWidget(audio_mode_label)
        audio_mode_layout.addWidget(self.audio_mode_combo)
        advanced_format_options_frame.inner_layout.addLayout(audio_mode_layout)

        # Sample rate options dropdown
        samplerate_layout = QHBoxLayout()
        sample_rate_label = QLabel("Output Sample Rate (Hz)")
        self.sample_rate_combo = QComboBox(self)
        self.sample_rate_combo.addItems(["44100", "48000", "96000", "192000"])
        self.sample_rate_combo.setToolTip('Uses "44100" in Preview Mode')
        samplerate_layout.addWidget(sample_rate_label)
        samplerate_layout.addWidget(self.sample_rate_combo)
        advanced_format_options_frame.inner_layout.addLayout(samplerate_layout)

        # Resampler quality
        resampler_quality_layout = QHBoxLayout()
        resampler_quality_label = QLabel("Output Resampler Quality")
        self.resampler_quality_combo = QComboBox(self)
        self.resampler_quality_combo.addItems(["High", "Medium", "Low"])
        self.resampler_quality_combo.setToolTip('Uses "Low" in Preview Mode')
        resampler_quality_layout.addWidget(resampler_quality_label)
        resampler_quality_layout.addWidget(self.resampler_quality_combo)
        advanced_format_options_frame.inner_layout.addLayout(resampler_quality_layout)

        # Preview mode selection
        preview_mode_layout = QHBoxLayout()
        preview_mode_label = QLabel("Preview Mode")
        self.preview_mode_combo = QComboBox(self)
        self.preview_mode_combo.addItems(
            ["Preview with decode", "Preview real-time", "Preview only"]
        )
        self.preview_mode_combo.setToolTip(
            "Preview with decode: decode to file + play at 1x real-time (smooth)\n"
            "Preview real-time: decode to file + play at decode speed (fast)\n"
            "Preview only: play at decode speed, no file output (quick scrubbing)"
        )
        self.preview_mode_combo.setCurrentIndex(0)
        preview_mode_layout.addWidget(preview_mode_label)
        preview_mode_layout.addWidget(self.preview_mode_combo)
        advanced_format_options_frame.inner_layout.addLayout(preview_mode_layout)

        return layout

    def build_expander_deemphasis_section(self):
        layout = QVBoxLayout()

        # Enable Expander/Deemphasis checkbox
        expander_controls_frame = CollapsableSection(
            self, "Expander Controls", default_collapsed=True
        )
        layout.addLayout(expander_controls_frame)

        self.enable_expander_checkbox = QCheckBox("Enabled")
        expander_controls_frame.inner_layout.addWidget(self.enable_expander_checkbox)

        # Expander Envelope Detection method combo
        expander_env_detection_layout = QHBoxLayout()
        expander_env_detection_label = QLabel("Envelope Detection Method")

        self.expander_env_detection_combo = QComboBox(self)
        self.expander_env_detection_combo.setToolTip(
            "Select the expander envelope detection method.\n Peak detection (JVC standard, IEC 60774-2),\n RMS detection (Some Panasonic and early VCRs)"
        )
        self.expander_env_detection_combo.addItems(ui_to_expander_env_detection.keys())
        expander_env_detection_layout.addWidget(expander_env_detection_label)
        expander_env_detection_layout.addWidget(self.expander_env_detection_combo)

        expander_controls_frame.inner_layout.addLayout(expander_env_detection_layout)

        expander_controls_layout = QHBoxLayout()
        self.expander_gain_dial_control = DialControl(
            self, "Gain (db)", QtGui.QDoubleValidator(), 1, 1, 60, width=2
        )
        expander_controls_layout.addWidget(self.expander_gain_dial_control)
        self.expander_ratio_dial_control = DialControl(
            self, "Ratio", QtGui.QDoubleValidator(), 10, 1, 10, width=2
        )
        expander_controls_layout.addWidget(self.expander_ratio_dial_control)
        self.expander_attack_tau_dial_control = DialControl(
            self, "Attack (𝜏)", QtGui.QDoubleValidator(), 10e3, 10e-4, 10e-3, width=4
        )
        expander_controls_layout.addWidget(self.expander_attack_tau_dial_control)
        self.expander_hold_tau_dial_control = DialControl(
            self, "Hold (𝜏)", QtGui.QDoubleValidator(), 10e3, 10e-4, 10e-3, width=4
        )
        expander_controls_layout.addWidget(self.expander_hold_tau_dial_control)
        self.expander_release_tau_dial_control = DialControl(
            self, "Release (𝜏)", QtGui.QDoubleValidator(), 10e2, 10e-3, 10e-2, width=4
        )
        expander_controls_layout.addWidget(self.expander_release_tau_dial_control)
        expander_controls_frame.inner_layout.addLayout(expander_controls_layout)

        expander_sideband_frame = CollapsableSection(
            self,
            "Expander Sideband Input (High-Pass Shelf Filter)",
            default_collapsed=True,
        )
        layout.addLayout(expander_sideband_frame)
        weighting_layout = QHBoxLayout()
        self.expander_weighting_low_tau_dial_control = DialControl(
            self,
            "Low Shelf (𝜏)",
            QtGui.QDoubleValidator(),
            10e5,
            DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2,
            10e-4,
            width=5
        )
        weighting_layout.addWidget(self.expander_weighting_low_tau_dial_control)
        self.expander_weighting_high_tau_dial_control = DialControl(
            self,
            "High Shelf (𝜏)",
            QtGui.QDoubleValidator(),
            10e5,
            10e-7,
            DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1,
            width=5
        )
        weighting_layout.addWidget(self.expander_weighting_high_tau_dial_control)
        self.expander_weighting_low_pass_dial_control = DialControl(
            self, "Low Pass (Hz)", QtGui.QDoubleValidator(), 1, 5000, 21000, 100, width=4
        )
        weighting_layout.addWidget(self.expander_weighting_low_pass_dial_control)
        self.expander_weighting_low_pass_transition_dial_control = DialControl(
            self, "Low Pass Transition (Hz)", QtGui.QDoubleValidator(), 1, 5000, 100000, 100, width=5
        )
        weighting_layout.addWidget(self.expander_weighting_low_pass_transition_dial_control)
        expander_sideband_frame.inner_layout.addLayout(weighting_layout)

        show_plot_btn_weighting = QPushButton("Plot")
        show_plot_btn_weighting.clicked.connect(self.show_plot)
        weighting_layout.addWidget(show_plot_btn_weighting)

        deemphasis_frame = CollapsableSection(
            self, "Deemphasis (Low-Pass Shelf Filter)", default_collapsed=True
        )
        layout.addLayout(deemphasis_frame)
        deemphasis_layout = QHBoxLayout()

        self.enable_deemphasis_checkbox = QCheckBox("Enabled")
        deemphasis_frame.inner_layout.addWidget(self.enable_deemphasis_checkbox)
        self.deemphasis_low_tau_dial_control = DialControl(
            self,
            "Low Shelf (𝜏)",
            QtGui.QDoubleValidator(),
            10e5,
            DEFAULT_VHS_DEEMPHASIS_TAU_2,
            10e-4,
            width=5
        )
        deemphasis_layout.addWidget(self.deemphasis_low_tau_dial_control)
        self.deemphasis_high_tau_dial_control = DialControl(
            self,
            "High Shelf (𝜏)",
            QtGui.QDoubleValidator(),
            10e5,
            10e-7,
            DEFAULT_VHS_DEEMPHASIS_TAU_1,
            width=5
        )
        deemphasis_layout.addWidget(self.deemphasis_high_tau_dial_control)

        self.nr_deemphasis_low_tau_dial_control = DialControl(
            self,
            "NR Low Shelf (𝜏)",
            QtGui.QDoubleValidator(),
            10e5,
            DEFAULT_VHS_NR_DEEMPHASIS_TAU_2,
            10e-4,
            width=5
        )
        deemphasis_layout.addWidget(self.nr_deemphasis_low_tau_dial_control)
        self.nr_deemphasis_high_tau_dial_control = DialControl(
            self,
            "NR High Shelf (𝜏)",
            QtGui.QDoubleValidator(),
            10e5,
            10e-7,
            DEFAULT_VHS_NR_DEEMPHASIS_TAU_1,
            width=5
        )
        deemphasis_layout.addWidget(self.nr_deemphasis_high_tau_dial_control)

        show_plot_btn_deemphasis = QPushButton("Plot")
        show_plot_btn_deemphasis.clicked.connect(self.show_plot)
        deemphasis_layout.addWidget(show_plot_btn_deemphasis)

        deemphasis_frame.inner_layout.addLayout(deemphasis_layout)

        return layout
    
    def schedule_plot_update(self):
        if (
            self.weighting_deemphasis_plot is not None
            and self.weighting_deemphasis_plot.isVisible()
        ):
            self._plot_update_timer.start(50)
    
    def build_plot_window(self):
        self.weighting_deemphasis_plot = None
        self._plot_update_timer = QtCore.QTimer()
        self._plot_update_timer.setSingleShot(True)
        self._plot_update_timer.timeout.connect(self._update_visible_plot)

        self.expander_gain_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_ratio_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_attack_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_hold_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_release_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_ratio_dial_control.valueChanged.connect(self.schedule_plot_update)

        self.expander_weighting_low_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_weighting_high_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_weighting_low_pass_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.expander_weighting_low_pass_transition_dial_control.valueChanged.connect(self.schedule_plot_update)

        self.deemphasis_low_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.deemphasis_high_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.nr_deemphasis_low_tau_dial_control.valueChanged.connect(self.schedule_plot_update)
        self.nr_deemphasis_high_tau_dial_control.valueChanged.connect(self.schedule_plot_update)

        self.audio_mode_combo.currentIndexChanged.connect(self.schedule_plot_update)
        self.sample_rate_combo.currentIndexChanged.connect(self.schedule_plot_update)

    def _update_visible_plot(self):
        if (
            self.weighting_deemphasis_plot is not None
            and self.weighting_deemphasis_plot.isVisible()
        ):
            self.weighting_deemphasis_plot.update_plot()
    
    def show_plot(self):
        if self.weighting_deemphasis_plot is None:
            self.weighting_deemphasis_plot = PlotWindow(
                "Weighting / Deemphasis", self.getValues
            )
        geo = self.geometry()
        self.weighting_deemphasis_plot.move(geo.x() + geo.width() + 20, geo.y())
        self.weighting_deemphasis_plot.update_plot()
        self.weighting_deemphasis_plot.show()

    @property
    def transport_state(self):
        return self._transport_state if self.isVisible() else 0

    @transport_state.setter
    def transport_state(self, value):
        if value == STOP_STATE:
            self.on_stop_clicked()
        elif value == PLAY_STATE:
            self.on_play_clicked()
        elif value == PAUSE_STATE:
            self.on_pause_clicked()
        elif value == PREVIEW_STATE:
            self.on_preview_clicked()
        else:
            raise ValueError("Invalid transport state value")
        self._transport_state = value

    def setValues(self, values: MainUIParameters):
        self.volume_dial_control.setValue(values.volume)
        self.expander_gain_dial_control.setValue(values.expander_gain)
        self.expander_ratio_dial_control.setValue(values.expander_ratio)
        self.expander_env_detection_combo.setCurrentText(values.expander_env_detection)
        self.expander_env_detection_combo.setCurrentIndex(
            self.expander_env_detection_combo.findText(values.expander_env_detection)
        )
        self.expander_attack_tau_dial_control.setValue(values.expander_attack_tau)
        self.expander_hold_tau_dial_control.setValue(values.expander_hold_tau)
        self.expander_release_tau_dial_control.setValue(values.expander_release_tau)
        self.expander_weighting_low_tau_dial_control.setValue(
            values.expander_weighting_low_tau
        )
        self.expander_weighting_high_tau_dial_control.setValue(
            values.expander_weighting_high_tau
        )
        self.expander_weighting_low_pass_dial_control.setValue(
            values.expander_weighting_low_pass
        )
        self.expander_weighting_low_pass_transition_dial_control.setValue(
            values.expander_weighting_low_pass_transition
        )
        self.deemphasis_low_tau_dial_control.setValue(values.deemphasis_low_tau)
        self.deemphasis_high_tau_dial_control.setValue(values.deemphasis_high_tau)
        self.nr_deemphasis_low_tau_dial_control.setValue(values.nr_deemphasis_low_tau)
        self.nr_deemphasis_high_tau_dial_control.setValue(values.nr_deemphasis_high_tau)
        self.spectral_nr_amount_dial_control.setValue(values.spectral_nr_amount)
        self.normalize_checkbox.setChecked(values.normalize)
        self.doc_combo.setCurrentText(values.doc)
        self.doc_combo.setCurrentIndex(
            self.doc_combo.findText(values.doc)
        )
        self.enable_expander_checkbox.setChecked(values.enable_expander)
        self.enable_deemphasis_checkbox.setChecked(values.enable_deemphasis)
        self.head_switching_interpolation_checkbox.setChecked(
            values.head_switching_interpolation
        )
        self.automatic_fine_tuning_checkbox.setChecked(values.automatic_fine_tuning)
        self.bias_guess_checkbox.setChecked(values.bias_guess)
        self.sample_rate_combo.setCurrentText(str(values.audio_sample_rate))
        self.sample_rate_combo.setCurrentIndex(
            self.sample_rate_combo.findText(str(values.audio_sample_rate))
        )
        self.standard_combo.setCurrentText(values.standard)
        self.standard_combo.setCurrentIndex(
            self.standard_combo.findText(values.standard)
        )
        self.format_combo.setCurrentText(values.format)
        self.format_combo.setCurrentIndex(self.format_combo.findText(values.format))
        self.audio_mode_combo.setCurrentText(values.audio_mode)
        self.audio_mode_combo.setCurrentIndex(
            self.audio_mode_combo.findText(values.audio_mode)
        )
        self.resampler_quality_combo.setCurrentText(values.resampler_quality.title())
        self.resampler_quality_combo.setCurrentIndex(
            self.resampler_quality_combo.findText(values.resampler_quality.title())
        )

        self.demod_type_combo.setCurrentText(values.demod_type)
        self.demod_type_combo.setCurrentIndex(
            self.demod_type_combo.findText(values.demod_type)
        )

        self.threads_spinbox.setValue(max(1, int(values.threads)))

        self.input_sample_rate = values.input_sample_rate / 1e6
        found_rate = False
        for i, rate in enumerate(self._input_combo_rates):
            if abs((rate * 1e6) - values.input_sample_rate) < 5000:
                self.input_samplerate_combo.setCurrentIndex(i)
                found_rate = True
                break
        if not found_rate:
            new_other_text = f"Other ({(values.input_sample_rate / 1e6):g})"
            self.input_samplerate_combo.setPlaceholderText(new_other_text)

        self.update_afe_values(
            values.format,
            values.standard,
            values.afe_left_carrier_deviation,
            values.afe_right_carrier_deviation,
            values.afe_left_carrier,
            values.afe_right_carrier,
        )

        self.update_deemphasis_expander_values(values.format)

        self.input_file = values.input_file
        self.output_file = values.output_file
        # Map preview flags to combo index
        if values.preview_only:
            self.preview_mode_combo.setCurrentIndex(2)
        elif values.preview_real_time:
            self.preview_mode_combo.setCurrentIndex(1)
        else:
            self.preview_mode_combo.setCurrentIndex(0)

    def getValues(self) -> MainUIParameters:
        values = MainUIParameters()
        values.volume = self.volume_dial_control.value()
        values.expander_gain = self.expander_gain_dial_control.value()
        values.expander_ratio = self.expander_ratio_dial_control.value()
        values.expander_env_detection = self.expander_env_detection_combo.currentText()
        values.expander_attack_tau = self.expander_attack_tau_dial_control.value()
        values.expander_hold_tau = self.expander_hold_tau_dial_control.value()
        values.expander_release_tau = self.expander_release_tau_dial_control.value()
        values.expander_weighting_low_tau = (
            self.expander_weighting_low_tau_dial_control.value()
        )
        values.expander_weighting_high_tau = (
            self.expander_weighting_high_tau_dial_control.value()
        )
        values.expander_weighting_low_pass = (
            self.expander_weighting_low_pass_dial_control.value()
        )
        values.expander_weighting_low_pass_transition = (
            self.expander_weighting_low_pass_transition_dial_control.value()
        )
        values.deemphasis_low_tau = self.deemphasis_low_tau_dial_control.value()
        values.deemphasis_high_tau = self.deemphasis_high_tau_dial_control.value()
        values.nr_deemphasis_low_tau = self.nr_deemphasis_low_tau_dial_control.value()
        values.nr_deemphasis_high_tau = self.nr_deemphasis_high_tau_dial_control.value()
        values.afe_left_carrier_deviation = self.afe_left_carrier_deviation_spinbox.value()
        values.afe_right_deviation = self.afe_right_carrier_deviation_spinbox.value()
        values.afe_left_carrier = self.afe_left_carrier_spinbox.value()
        values.afe_right_carrier = self.afe_right_carrier_spinbox.value()
        values.spectral_nr_amount = float(
            self.spectral_nr_amount_dial_control.textbox.text()
        )
        values.normalize = self.normalize_checkbox.isChecked()
        values.doc = self.doc_combo.currentText()
        values.enable_expander = self.enable_expander_checkbox.isChecked()
        values.enable_deemphasis = self.enable_deemphasis_checkbox.isChecked()
        values.head_switching_interpolation = (
            self.head_switching_interpolation_checkbox.isChecked()
        )
        values.automatic_fine_tuning = self.automatic_fine_tuning_checkbox.isChecked()
        values.bias_guess = self.bias_guess_checkbox.isChecked()
        values.audio_sample_rate = int(self.sample_rate_combo.currentText())
        values.standard = self.standard_combo.currentText()
        values.format = self.format_combo.currentText()
        values.audio_mode = self.audio_mode_combo.currentText()
        values.resampler_quality = self.resampler_quality_combo.currentText().lower()
        values.demod_type = self.demod_type_combo.currentText()
        values.input_sample_rate = self.input_sample_rate
        values.input_file = self.input_file
        values.output_file = self.output_file
        values.threads = self.threads_spinbox.value()
        # Map combo index to preview flags
        preview_mode_index = self.preview_mode_combo.currentIndex()
        values.preview_real_time = preview_mode_index in (1, 2)
        values.preview_only = preview_mode_index == 2
        return values

    def update_afe_values(
        self,
        format,
        standard,
        afe_left_carrier_deviation=0,
        afe_right_carrier_deviation=0,
        afe_left_carrier=0,
        afe_right_carrier=0,
    ):
        standard, _ = get_standard(
            "vhs" if format == "VHS" else "8mm",
            "p" if standard == "PAL" else "n",
            afe_left_carrier_deviation,
            afe_right_carrier_deviation,
            afe_left_carrier,
            afe_right_carrier,
        )
        self.afe_left_carrier_deviation_spinbox.setValue(int(standard.LCarrierDeviation))
        self.afe_right_carrier_deviation_spinbox.setValue(int(standard.RCarrierDeviation))
        self.afe_left_carrier_spinbox.setValue(int(standard.LCarrierRef))
        self.afe_right_carrier_spinbox.setValue(int(standard.RCarrierRef))

    def update_deemphasis_expander_values(self, format):
        if format == "VHS":
            self.deemphasis_low_tau_dial_control.setValue(DEFAULT_VHS_DEEMPHASIS_TAU_1)
            self.deemphasis_high_tau_dial_control.setValue(DEFAULT_VHS_DEEMPHASIS_TAU_2)
            self.nr_deemphasis_low_tau_dial_control.setValue(DEFAULT_VHS_NR_DEEMPHASIS_TAU_1)
            self.nr_deemphasis_high_tau_dial_control.setValue(DEFAULT_VHS_NR_DEEMPHASIS_TAU_2)
            self.expander_gain_dial_control.setValue(DEFAULT_VHS_EXPANDER_GAIN)
            self.expander_ratio_dial_control.setValue(DEFAULT_VHS_EXPANDER_RATIO)
            self.expander_attack_tau_dial_control.setValue(DEFAULT_VHS_EXPANDER_ATTACK_TAU)
            self.expander_hold_tau_dial_control.setValue(DEFAULT_VHS_EXPANDER_HOLD_TAU)
            self.expander_release_tau_dial_control.setValue(DEFAULT_VHS_EXPANDER_RELEASE_TAU)
            self.expander_weighting_low_tau_dial_control.setValue(DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_1)
            self.expander_weighting_high_tau_dial_control.setValue(DEFAULT_VHS_EXPANDER_WEIGHTING_TAU_2)
            self.expander_weighting_low_pass_dial_control.setValue(DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS)
            self.expander_weighting_low_pass_transition_dial_control.setValue(DEFAULT_VHS_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION)
            self.audio_mode_combo.setCurrentText(audio_mode_to_ui[DEFAULT_VHS_AUDIO_MODE])
        else:
            self.deemphasis_low_tau_dial_control.setValue(DEFAULT_8MM_DEEMPHASIS_TAU_1)
            self.deemphasis_high_tau_dial_control.setValue(DEFAULT_8MM_DEEMPHASIS_TAU_2)
            self.nr_deemphasis_low_tau_dial_control.setValue(DEFAULT_8MM_NR_DEEMPHASIS_TAU_1)
            self.nr_deemphasis_high_tau_dial_control.setValue(DEFAULT_8MM_NR_DEEMPHASIS_TAU_2)
            self.expander_gain_dial_control.setValue(DEFAULT_8MM_EXPANDER_GAIN)
            self.expander_ratio_dial_control.setValue(DEFAULT_8MM_EXPANDER_RATIO)
            self.expander_attack_tau_dial_control.setValue(DEFAULT_8MM_EXPANDER_ATTACK_TAU)
            self.expander_hold_tau_dial_control.setValue(DEFAULT_8MM_EXPANDER_HOLD_TAU)
            self.expander_release_tau_dial_control.setValue(DEFAULT_8MM_EXPANDER_RELEASE_TAU)
            self.expander_weighting_low_tau_dial_control.setValue(DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_1)
            self.expander_weighting_high_tau_dial_control.setValue(DEFAULT_8MM_EXPANDER_WEIGHTING_TAU_2)
            self.expander_weighting_low_pass_dial_control.setValue(DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS)
            self.expander_weighting_low_pass_transition_dial_control.setValue(DEFAULT_8MM_EXPANDER_WEIGHTING_LOW_PASS_TRANSITION)
            self.audio_mode_combo.setCurrentText(audio_mode_to_ui[DEFAULT_8MM_AUDIO_MODE])

    def on_standard_change(self):
        self.update_afe_values(
            format=self.format_combo.currentText(),
            standard=self.standard_combo.currentText(),
        )

    def on_format_change(self):
        self.update_afe_values(
            format=self.format_combo.currentText(),
            standard=self.standard_combo.currentText(),
        )
        self.update_deemphasis_expander_values(self.format_combo.currentText())

    def change_button_color(self, button, color):
        button.setStyleSheet(
            """
            QPushButton {
                background-color: %s;
                color: #000;
            }
        """
            % color
        )

    def default_button_color(self, button):
        button.setStyleSheet(
            """
            QPushButton {
                background-color: #555;
                color: #eee;
            }
        """
        )

    def confirm_overwrite(self):
        # checks if destination file exists and prompts user to overwrite
        if os.path.exists(self.output_file) and self.transport_state == 0:
            message_box = QMessageBox(
                QMessageBox.Icon.Question,
                "File Exists",
                "Overwrite existing file?",
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                parent=self,
            )
            message_box.setIcon(QMessageBox.Icon.Warning)
            message_box.setDefaultButton(QMessageBox.StandardButton.No)
            # dark mode
            message_box.setStyleSheet(
                """
                QMessageBox {
                    background-color: #333;
                    color: #eee;
                }
                QPushButton {
                    background-color: #555;
                    color: #eee;
                }
            """
            )
            message_box.exec()
            overwrite = message_box.standardButton(message_box.clickedButton())

            if overwrite == QMessageBox.StandardButton.No:
                print("Overwrite cancelled.")
                return True

        return False

    def on_play_clicked(self):
        print("[PLAY] Play command issued.")
        if self.confirm_overwrite():
            return

        self._transport_state = PLAY_STATE
        self.change_button_color(self.play_button, "#0f0")
        self.default_button_color(self.preview_button)
        self.default_button_color(self.pause_button)
        self.default_button_color(self.stop_button)
        self._set_status_window_icon("document-save")

    def on_preview_clicked(self):
        print("[PREVIEW] Preview command issued.")
        if self.confirm_overwrite():
            return

        self._transport_state = PREVIEW_STATE
        self.change_button_color(self.preview_button, "#0f0")
        self.default_button_color(self.play_button)
        self.default_button_color(self.pause_button)
        self.default_button_color(self.stop_button)
        self._set_status_window_icon("document-save")

    def on_pause_clicked(self):
        self.change_button_color(self.pause_button, "#eee")
        self.default_button_color(self.preview_button)
        self.default_button_color(self.play_button)
        self.default_button_color(self.stop_button)
        self._set_status_window_icon("document-open")
        print("|| Pause command issued.")
        self._transport_state = PAUSE_STATE

    def on_stop_clicked(self):
        self.change_button_color(self.stop_button, "#eee")
        self.default_button_color(self.preview_button)
        self.default_button_color(self.play_button)
        self.default_button_color(self.pause_button)
        print("[STOP] Stop command issued.")
        self._transport_state = STOP_STATE

    def set_input_sample_rate_mhz(self, mhz: float):
        """Set the input sample rate, selecting a matching preset when one
        exists, otherwise showing it as "Other (X)"."""
        self.input_sample_rate = mhz
        for i, rate in enumerate(self._input_combo_rates):
            if abs((rate - mhz) * 1e6) < 5000:
                self.input_samplerate_combo.setCurrentIndex(i)
                # use the preset value so the displayed selection and the
                # rate used for decoding always match
                self.input_sample_rate = rate
                return
        self.input_samplerate_combo.setPlaceholderText(f"Other ({mhz:g})")
        self.input_samplerate_combo.setCurrentIndex(-1)

    def on_input_samplerate_changed(self):
        if self.input_samplerate_combo.currentIndex() < 0:
            # placeholder ("Other (X)") is showing, rate was set directly
            return
        print("Input sample rate changed.")
        if "Other" in self.input_samplerate_combo.currentText():
            input_dialog = InputDialog(
                title="Input Sample Rate",
                label="MHz",
                value=self.input_sample_rate,
                validator=QtGui.QDoubleValidator(),
            )
            value: float = input_dialog.get_input_value()
            if value is not None:
                new_other_text = f"Other ({float(value):g})"
                self.input_samplerate_combo.setPlaceholderText(new_other_text)
                self.input_samplerate_combo.setCurrentIndex(-1)
                self.input_sample_rate = value
        else:
            self.input_sample_rate = self._input_combo_rates[
                self.input_samplerate_combo.currentIndex()
            ]

    def generic_message_box(
        self,
        title: str,
        message: str,
        type: QMessageBox.Icon = QMessageBox.Icon.Information,
    ):
        message_box = QMessageBox(type, title, message, parent=self)
        message_box.setIcon(type)
        message_box.setDefaultButton(QMessageBox.StandardButton.Ok)
        # dark mode
        message_box.setStyleSheet(
            """
            QMessageBox {
                background-color: #333;
                color: #eee;
            }
            QPushButton {
                background-color: #555;
                color: #eee;
            }
        """
        )
        message_box.exec()

    def on_decode_finished(self, decoded_filename: str = "input stream"):
        self._set_status_window_icon("document-open")
        # alerts user that decode is finished
        self.generic_message_box(
            "Decode Finished", f"Decode of {decoded_filename} finished."
        )


class FileOutputDialogUI(HifiUi):
    def __init__(
        self,
        params: MainUIParameters,
        title: str = "HiFi Decode",
        main_layout_callback=None,
    ):
        super(FileOutputDialogUI, self).__init__(params, title, self._layout_callback)

    def _layout_callback(self, main_layout):
        # Add file output widgets
        self.file_output_label = QLabel("Output file")
        self.file_output_textbox = QLineEdit(self)
        self.file_output_button = QPushButton("Browse", self)
        self.file_output_button.clicked.connect(self.on_file_output_button_clicked)
        self.file_output_layout = QHBoxLayout()
        self.file_output_layout.setContentsMargins(0, 0, 0, 0)
        self.file_output_layout.setSpacing(4)
        self.file_output_layout.addWidget(self.file_output_label)
        self.file_output_layout.addWidget(self.file_output_textbox)
        self.file_output_layout.addWidget(self.file_output_button)
        self.file_output_layout.setStretch(0, 0)
        self.file_output_layout.setStretch(1, 1)
        self.file_output_layout.setStretch(2, 0)
        self.main_layout.addLayout(self.file_output_layout)
        # sets browse button height to match text box height
        self.file_output_button.setFixedHeight(
            self.file_output_textbox.sizeHint().height()
        )
        self.file_output_textbox.setFixedHeight(
            self.file_output_textbox.sizeHint().height()
        )

    def on_file_output_button_clicked(self):
        qdialog = QFileDialog(self)
        qdialog.setFileMode(QFileDialog.FileMode.AnyFile)
        if os.path.isdir(os.path.dirname(self.file_output_textbox.text())):
            qdialog.setDirectory(os.path.dirname(self.file_output_textbox.text()))

        file_name, _ = qdialog.getOpenFileName(
            self, "Open File", "", "All Files (*);;FLAC (*.flac)"
        )
        if file_name:
            self.file_output_textbox.setText(file_name)
        print("Output file browse button clicked.")

    def getValues(self) -> MainUIParameters:
        values = super(FileOutputDialogUI, self).getValues()
        values.output_file = self.file_output_textbox.text()
        return values

    def setValues(self, values: MainUIParameters):
        super(FileOutputDialogUI, self).setValues(values)
        self.file_output_textbox.setText(values.output_file)

    def on_decode_finished(self, decoded_filename: str = "input stream"):
        decoded_filename = os.path.basename(self.file_output_textbox.text())
        super(FileOutputDialogUI, self).on_decode_finished(decoded_filename)


class DialControl(QWidget):
    valueChanged = QtCore.pyqtSignal(float)

    def __init__(
        self,
        main_window,
        label_text,
        validator,
        scale=1,
        min_value=0,
        max_value=1,
        scroll_step_size=1,
        width=None
    ):
        super(QWidget, self).__init__()
        layout = QHBoxLayout(self)

        # Volume dial and numeric textbox
        label_and_textbox = QVBoxLayout()

        label = QLabel(label_text)
        self.scale = scale

        textbox_character_width = max(int(math.log10(self.scale)) + 1, 3) if width is None else width
        self.textbox = QLineEdit(main_window)
        self.textbox.setValidator(validator)
        metrics = QtGui.QFontMetrics(self.textbox.font())
        width = metrics.horizontalAdvance("M") * textbox_character_width
        self.textbox.setFixedWidth(width)
        self.textbox.show()

        self.dial = QDial(main_window)
        self.dial.setRange(int(min_value * scale), int(max_value * scale))
        self.dial.setPageStep(1)
        self.dial.setSingleStep(scroll_step_size)

        label_and_textbox.addWidget(label)
        label_and_textbox.addWidget(self.textbox)
        layout.addWidget(self.dial)
        layout.addLayout(label_and_textbox)

        self.dial.valueChanged.connect(self.on_dial_change)
        self.textbox.editingFinished.connect(self.on_textbox_change)

    def setValue(self, value):
        self.dial.setValue(int(value * self.scale))
        self.textbox.setText(f"{value:g}")

    def value(self):
        return float(self.textbox.text())

    def on_dial_change(self, value):
        scaled_value = value / self.scale
        self.textbox.setText(f"{scaled_value:g}")
        self.valueChanged.emit(self.value())

    def on_textbox_change(self):
        text = self.textbox.text()
        try:
            value = float(text)
            self.dial.setValue(int(value * self.scale))
            self.valueChanged.emit(value)
        except ValueError:
            pass


class CollapsableSection(QVBoxLayout):
    def __init__(self, main_window, label_text, default_collapsed=False, bg_color = "#333"):
        super(CollapsableSection, self).__init__()
        self.setContentsMargins(0, 0, 0, 0)
        self.setSpacing(1)

        self.main_window = main_window
        self.main_window.collapsableSections.append(self)
        self.default_collapsed = default_collapsed

        # Create toggle button
        self.collapsed_arrow = QLabel()
        self.collapsed_arrow.setStyleSheet("border: none;")
        self.collapsed_arrow.setFixedSize(
            self.collapsed_arrow.sizeHint().width(),
            self.collapsed_arrow.sizeHint().width(),
        )
        collapsed_arrow_font = self.collapsed_arrow.font()
        collapsed_arrow_font.setBold(True)
        collapsed_arrow_font.setPointSize(10)
        collapsed_arrow_font.setStyleHint(QtGui.QFont.StyleHint.Monospace)
        self.collapsed_arrow.setFont(collapsed_arrow_font)

        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        line.setStyleSheet("color: #666;")
        line.setFixedHeight(2)

        label = QLabel(label_text)
        label.setStyleSheet(
            f"background-color: {bg_color}; padding: 1 0px; font-size: 0.8rem;"
        )
        label_font = label.font()
        label_font.setItalic(True)
        label.setFont(label_font)

        # Wrapper widget to hold label + padding
        label_layout = QHBoxLayout()
        label_layout.setContentsMargins(0, 0, 0, 0)
        label_layout.addWidget(label)
        label_widget = QWidget()
        label_widget.setLayout(label_layout)

        # Add both label and line to the same grid cell
        self.header = QWidget()

        horizontal_spacer = QGridLayout(self.header)
        horizontal_spacer.setContentsMargins(0, 0, 0, 0)
        horizontal_spacer.setSpacing(0)
        horizontal_spacer.addWidget(
            self.collapsed_arrow, 0, 0, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        horizontal_spacer.addWidget(
            label_widget, 0, 1, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter
        )
        horizontal_spacer.addWidget(
            line, 0, 2, alignment=QtCore.Qt.AlignmentFlag.AlignVCenter
        )

        self.body = QFrame()
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setObjectName("LabeledFrame")
        self.body.setStyleSheet(
            "QFrame#LabeledFrame { border: 1px solid #aaa; border-radius: 15px; }"
        )
        self.addWidget(self.header)
        self.addWidget(self.body)

        self.header.mousePressEvent = self._toggle
        self.inner_layout = QVBoxLayout(self.body)

    def setDefaultCollapseState(self):
        self._setCollapsed(self.default_collapsed)

    def _setCollapsed(self, value):
        self._collapsed = value
        self.body.setVisible(not self._collapsed)
        self.collapsed_arrow.setText(
            "▶" if self._collapsed else "▼"
        )  # Right arrow if collapsed
        self.update()
        self.main_window.resize_window(axis="v")

    def _toggle(self, _):
        self._setCollapsed(not self._collapsed)


class FileIODialogUI(HifiUi):
    OUTPUT_FILE_SUFFIX = "_HiFi_Decoded.flac"

    def __init__(
        self,
        params: MainUIParameters,
        title: str = "HiFi Decode",
        main_layout_callback=None,
    ):
        super(FileIODialogUI, self).__init__(params, title, self._layout_callback)
        # accept files dragged from a file manager anywhere on the window
        self.setAcceptDrops(True)

    def _layout_callback(self, main_layout):
        # Add file input widgets
        self.file_input_label = QLabel("Input file")
        self.file_input_textbox = QLineEdit(self)
        self.file_input_textbox.setPlaceholderText(
            "Select, type or drag & drop the RF capture file here"
        )
        # let drops fall through to the window-level drop handler instead of
        # QLineEdit pasting a file:// url as plain text
        self.file_input_textbox.setAcceptDrops(False)
        self.file_input_textbox.editingFinished.connect(
            self.on_input_editing_finished
        )
        self._last_input_path = ""
        self.file_input_button = QPushButton("Browse", self)
        self.file_input_button.clicked.connect(self.on_file_input_button_clicked)
        self.file_input_layout = QHBoxLayout()
        self.file_input_layout.setContentsMargins(0, 0, 0, 0)
        self.file_input_layout.setSpacing(4)
        self.file_input_layout.addWidget(self.file_input_label)
        self.file_input_layout.addWidget(self.file_input_textbox)
        self.file_input_layout.addWidget(self.file_input_button)
        self.file_input_layout.setStretch(0, 0)
        self.file_input_layout.setStretch(1, 1)
        self.file_input_layout.setStretch(2, 0)
        self.main_layout.addLayout(self.file_input_layout)

        # Add file output widgets
        self.file_output_label = QLabel("Output file")
        self.file_output_textbox = QLineEdit(self)
        self.file_output_button = QPushButton("Browse", self)
        self.file_output_button.clicked.connect(self.on_file_output_button_clicked)
        self.file_output_layout = QHBoxLayout()
        self.file_output_layout.setContentsMargins(0, 0, 0, 0)
        self.file_output_layout.setSpacing(4)
        self.file_output_layout.addWidget(self.file_output_label)
        self.file_output_layout.addWidget(self.file_output_textbox)
        self.file_output_layout.addWidget(self.file_output_button)
        self.file_output_layout.setStretch(0, 0)
        self.file_output_layout.setStretch(1, 1)
        self.file_output_layout.setStretch(2, 0)
        self.main_layout.addLayout(self.file_output_layout)
        # Keep both file rows visually uniform, but compact.
        # Use textbox baseline height and trim a little so the controls don't
        # grow larger than needed under Qt6/macOS style metrics.
        base_height = max(
            self.file_input_textbox.sizeHint().height(),
            self.file_output_textbox.sizeHint().height(),
        )
        row_height = max(20, base_height - 2)
        self.file_input_textbox.setFixedHeight(row_height)
        self.file_output_textbox.setFixedHeight(row_height)
        self.file_input_button.setFixedHeight(row_height)
        self.file_output_button.setFixedHeight(row_height)

        button_width = max(
            self.file_input_button.sizeHint().width(),
            self.file_output_button.sizeHint().width(),
        )
        self.file_input_button.setFixedWidth(button_width)
        self.file_output_button.setFixedWidth(button_width)

        # Keep labels exactly the same width for row alignment.
        max_label_width = max(
            self.file_input_label.sizeHint().width(),
            self.file_output_label.sizeHint().width(),
        )
        self.file_input_label.setFixedWidth(max_label_width)
        self.file_output_label.setFixedWidth(max_label_width)

    @staticmethod
    def derive_output_file(input_path: str) -> str:
        """Build <input dir>/<input basename>_HiFi_Decoded.flac"""
        root, _ = os.path.splitext(input_path)
        return f"{root}{FileIODialogUI.OUTPUT_FILE_SUFFIX}"

    def auto_populate_output_file(self, input_path: str):
        if not input_path:
            return
        derived = FileIODialogUI.derive_output_file(input_path)
        if self.file_output_textbox.text() != derived:
            self.file_output_textbox.setText(derived)
            print(f"Output file auto set to: {derived}")

    def auto_set_input_frequency(self, input_path: str):
        """Initialize the input sample rate from a FLAC capture's header.

        RF FLAC captures store the capture rate in kHz (e.g. MISRC writes
        40000 for 40 MSps) because the STREAMINFO field cannot hold MHz
        rates.
        """
        if not input_path.lower().endswith(".flac"):
            return
        streaminfo = parse_flac_streaminfo(input_path)
        if streaminfo is None or streaminfo["sample_rate"] <= 0:
            return
        sample_rate = streaminfo["sample_rate"]
        if sample_rate in (44100, 48000):
            # Standard audio rates, not an FM RF capture. RF FLAC stores the
            # rate in kHz (e.g. 40 MSps = 40000), so 44100/48000 are audio.
            print(
                "WARNING: Current loaded file could be audio, not FM RF data! "
                f"(sample rate {sample_rate} Hz)"
            )
            # only show the modal for interactive loads (drop/browse/type);
            # during launcher pre-pop the window is not visible yet, so a
            # modal here would appear before the main window is shown.
            if self.isVisible():
                self.generic_message_box(
                    "Warning",
                    "WARNING: Current loaded file could be audio, not FM RF data!",
                    QMessageBox.Icon.Warning,
                )
            return
        mhz = sample_rate / 1000.0
        self.set_input_sample_rate_mhz(mhz)
        print(f"Input sample rate auto set to {mhz:g} MHz from the FLAC header")

    def auto_detect_format_system_from_filename(self, input_path: str):
        """Auto-set Format and Standard from keywords in the input filename.

        Recognizes VHS/SVHS, Video8/Hi8, Betamax, Betacam, NTSC and PAL. Only
        updates a control when a matching keyword is found; the existing
        selection is left untouched otherwise.
        """
        if not input_path:
            return
        name_lower = os.path.basename(input_path).lower()

        # Check rarer/more-specific formats first so SVHS/VHS don't shadow them.
        if "betamax" in name_lower:
            self.format_combo.setCurrentText("Betamax")
        elif "betacam" in name_lower:
            self.format_combo.setCurrentText("Betacam")
        elif "video8" in name_lower or "hi8" in name_lower or "hi-8" in name_lower:
            self.format_combo.setCurrentText("Video8/Hi8")
        elif (
            "svhs" in name_lower
            or "s-vhs" in name_lower
            or "supervhs" in name_lower
            or "super-vhs" in name_lower
            or "vhs" in name_lower
        ):
            self.format_combo.setCurrentText("VHS")

        if "ntsc" in name_lower:
            self.standard_combo.setCurrentText("NTSC")
        elif "pal" in name_lower:
            self.standard_combo.setCurrentText("PAL")

    def set_input_file(self, file_name: str):
        self.file_input_textbox.setText(file_name)
        self._last_input_path = file_name
        self.auto_populate_output_file(file_name)
        self.auto_set_input_frequency(file_name)
        self.auto_detect_format_system_from_filename(file_name)

    def on_input_editing_finished(self):
        # auto fill the output when the user typed/pasted a new input path
        text = self.file_input_textbox.text()
        if text and text != self._last_input_path:
            self._last_input_path = text
            self.auto_populate_output_file(text)
            self.auto_set_input_frequency(text)
            self.auto_detect_format_system_from_filename(text)

    def dragEnterEvent(self, event):
        paths = extract_dropped_file_paths(event.mimeData())
        if any(os.path.isfile(p) for p in paths):
            event.acceptProposedAction()
            return
        if paths:
            event.ignore()
            return
        super().dragEnterEvent(event)

    # Required on macOS (Cocoa): without dragMoveEvent, move events default to
    # ignored and the drop target can be invalidated mid-drag, causing the
    # cursor to show "no drop" and dropEvent to never fire.
    def dragMoveEvent(self, event):
        paths = extract_dropped_file_paths(event.mimeData())
        if any(os.path.isfile(p) for p in paths):
            event.acceptProposedAction()
            return
        if paths:
            event.ignore()
            return
        super().dragMoveEvent(event)

    def dropEvent(self, event):
        for path in extract_dropped_file_paths(event.mimeData()):
            if os.path.isfile(path):
                self.set_input_file(path)
                print(f"Input file set by drag and drop: {path}")
                event.acceptProposedAction()
                return
        event.ignore()

    def on_file_input_button_clicked(self):
        qdialog = QFileDialog(self)
        qdialog.setFileMode(QFileDialog.FileMode.AnyFile)
        if os.path.isdir(os.path.dirname(self.file_input_textbox.text())):
            qdialog.setDirectory(os.path.dirname(self.file_input_textbox.text()))
        file_name, _ = qdialog.getOpenFileName(
            self, "Open File", "", "All Files (*);;FLAC (*.flac)"
        )

        if os.path.exists(file_name):
            self.set_input_file(file_name)
        print("Input browse button clicked.")

    def on_file_output_button_clicked(self):
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Save File",
            "",
            "All Files (*);;FLAC (*.flac);; WAV (*.wav)",
        )

        if os.path.isdir(os.path.dirname(file_name)):
            self.file_output_textbox.setText(file_name)
        print("Output browse button clicked.")

    def getValues(self) -> MainUIParameters:
        values = super(FileIODialogUI, self).getValues()
        values.input_file = self.file_input_textbox.text()
        values.output_file = self.file_output_textbox.text()
        return values

    def setValues(self, values: MainUIParameters):
        super(FileIODialogUI, self).setValues(values)
        self.file_input_textbox.setText(values.input_file)
        self._last_input_path = values.input_file
        self.file_output_textbox.setText(values.output_file)
        # an input was provided without an output: derive the output name
        if values.input_file and not values.output_file:
            self.auto_populate_output_file(values.input_file)
        # auto-detect format/standard from the filename when pre-populated
        # (e.g. when launched from the decode-launcher with an input file).
        if values.input_file:
            self.auto_detect_format_system_from_filename(values.input_file)

    def on_decode_finished(self, decoded_filename: str = "input stream"):
        decoded_filename = os.path.basename(self.file_input_textbox.text())
        super(FileIODialogUI, self).on_decode_finished(decoded_filename)

class PlotWindow(QWidget):
    def __init__(self, title, getValues):
        super(QWidget, self).__init__()

        self.setWindowTitle(title)
        import matplotlib

        matplotlib.use("Qt5Agg")

        from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
        from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
        from matplotlib.figure import Figure

        fig = Figure(figsize=(12, 8))
        self.ax = fig.subplots()
        self.canvas = FigureCanvas(fig)

        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        layout.addWidget(self.toolbar)
        self.setLayout(layout)
        self.getValues = getValues
    
    @staticmethod
    def tau_as_freq(tau):
        return 1 / (2 * np.pi * tau)

    def plot_response(self, label, color, freq, mag_db, t1, t2):
        # Compute the frequency response
        # Convert taus to frequencies
        t1_f = round(PlotWindow.tau_as_freq(t1))
        t2_f = round(PlotWindow.tau_as_freq(t2))
        center_f = round(np.sqrt(PlotWindow.tau_as_freq(t1) * PlotWindow.tau_as_freq(t2)))
    
        # Plot the full frequency response
        self.ax.semilogx(freq, mag_db, label=label, color=color)
    
        # Helper: get magnitude at closest frequency index
        def point_at(freq_target):
            idx = np.argmin(np.abs(freq - freq_target))
            return freq[idx], mag_db[idx]
    
        # Compute (x, y) for points
        t1_x, t1_y = point_at(t1_f)
        t2_x, t2_y = point_at(t2_f)
        cf_x, cf_y = point_at(center_f)
    
        # Plot dot markers instead of axvlines
        self.ax.plot(t1_x, t1_y, marker='|', color=color, markersize=15, 
                 label=f"({t1_f} Hz)")
        self.ax.plot(cf_x,  cf_y, marker='o', color=color, markersize=5,
                 label=f"({center_f} Hz)")
        self.ax.plot(t2_x, t2_y, marker='|', color=color, markersize=15,
                 label=f"({t2_f} Hz)")

    def update_plot(self):
        self.ax.clear()
        ui_values = self.getValues()
        Deemphasis, Expander = _ensure_plot_helpers()

        deemphasis = Deemphasis(
            ui_values.audio_sample_rate,
            ui_values.deemphasis_low_tau,
            ui_values.deemphasis_high_tau,
        )
        deemphasis_freqs, deemphasis_mag_db = deemphasis.get_response()

        nr_deemphasis = Deemphasis(
            ui_values.audio_sample_rate,
            ui_values.nr_deemphasis_low_tau,
            ui_values.nr_deemphasis_high_tau,
        )
        nr_deemphasis_freqs, nr_deemphasis_mag_db = nr_deemphasis.get_response()

        expander = Expander(
            ui_values.audio_sample_rate,
            ui_values.expander_gain,
            ui_values.expander_ratio,
            ui_values.expander_env_detection,
            ui_values.expander_attack_tau,
            ui_values.expander_hold_tau,
            ui_values.expander_release_tau,
            ui_values.expander_weighting_low_tau,
            ui_values.expander_weighting_high_tau,
            ui_values.expander_weighting_low_pass,
            ui_values.expander_weighting_low_pass_transition,
        )
        expander_freqs, expander_mag_db = expander.get_response()

        self.plot_response(
            "Sideband",
            "green",
            expander_freqs,
            expander_mag_db,
            ui_values.expander_weighting_low_tau,
            ui_values.expander_weighting_high_tau,
        )
        self.plot_response(
            "Deemphasis",
            "red",
            deemphasis_freqs,
            deemphasis_mag_db,
            ui_values.deemphasis_low_tau,
            ui_values.deemphasis_high_tau,
        )
        self.plot_response(
            "NR Deemphasis",
            "blue",
            nr_deemphasis_freqs,
            nr_deemphasis_mag_db,
            ui_values.nr_deemphasis_low_tau,
            ui_values.nr_deemphasis_high_tau,
        )
        self.ax.grid(True, which="both")
        self.ax.set_xlabel("Frequency [Hz]")
        self.ax.set_ylabel("Amplitude [dB]")
        self.ax.set_xlim([1, ui_values.audio_sample_rate/2])
        self.ax.set_ylim([-20, 3])
        
        self.ax.legend()
        self.canvas.draw()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    params = MainUIParameters()
    window = FileIODialogUI(params)
    window.show()
    sys.exit(app.exec())
