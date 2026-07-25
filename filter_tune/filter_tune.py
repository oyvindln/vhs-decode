#!/usr/bin/env python3

# released under GNU GPL v3 or later

import math
import numpy as np
import numpy.fft as npfft
import os
import sys
import logging
import json
from pathlib import Path

try:
    QT_VERSION = 6

    from PyQt6.QtCore import (
        Qt,
        QObject,
        QSize,
        QTimer,
        QThread,
        pyqtSignal,
    )
    from PyQt6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDialog,
        QFileDialog,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QLabel,
        QPushButton,
        QScrollArea,
        QSlider,
        QSpinBox,
        QStyleFactory,
        QSplitter,
        QSizePolicy,
        QVBoxLayout,
        QWidget,
    )
    from PyQt6.QtGui import (
        QGuiApplication,
        QImage,
        QPixmap,
        QColor,
        QPalette,
    )
except ImportError:
    try:
        QT_VERSION = 5

        from PyQt5.QtCore import (
            Qt,
            QObject,
            QSize,
            QTimer,
            QThread,
            pyqtSignal,
        )
        from PyQt5.QtWidgets import (
            QApplication,
            QCheckBox,
            QComboBox,
            QDialog,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QPushButton,
            QScrollArea,
            QSlider,
            QSpinBox,
            QStyleFactory,
            QSplitter,
            QSizePolicy,
            QVBoxLayout,
            QWidget,
        )
        from PyQt5.QtGui import (
            QImage,
            QPixmap,
            QColor,
            QPalette,
        )
    except ImportError:
        print("Neither PyQt5 and PyQt6 not found! can't start filter-tune")
        sys.exit(1)

SUPPORTED_TAPE_FORMATS = (
    "VHS",
    "VHSHQ",
    "SVHS",
    "SVHS_ET",
    "UMATIC",
    "UMATIC_HI",
    "BETAMAX",
    "BETAMAX_HIFI",
    "SUPERBETA",
    "VIDEO8",
    "HI8",
    "EIAJ",
    "QUADRUPLEX",
    "VCR",
    "VCR_LP",
    "TYPEC",
    "TYPEB",
    "VHD",
    "VIDEO2000",
)

_compute_video_filters = None
_sub_deemphasis_inner = None

BLOCK_LEN = 32768
SAMPLE_RATE = (((1 / 64) * 283.75) + (25 / 1000000)) * 4e6
# fs = 40000000
# 2560
# FRAME_WIDTH = 1135
OUT_SCALE_DIVIDEND = np.double(0xC800 - 0x0400)


def _configure_matplotlib_cache_dir() -> None:
    if os.environ.get("MPLCONFIGDIR", "").strip():
        return

    if os.name == "nt":
        base = os.environ.get("LOCALAPPDATA", "").strip()
        if not base:
            base = str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Caches")
    else:
        base = os.environ.get("XDG_CACHE_HOME", "").strip()
        if not base:
            base = str(Path.home() / ".cache")

    matplotlib_cache_dir = Path(base) / "vhs-decode" / "matplotlib"
    try:
        matplotlib_cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ["MPLCONFIGDIR"] = str(matplotlib_cache_dir)
    except Exception:
        # Fall back to matplotlib defaults if cache directory setup fails.
        pass


_configure_matplotlib_cache_dir()


def multicall(l=[]):
    for c in l:
        c()


### copied from vhs-decode


def _hz_to_output(inp, sys_params, phase=0):
    reduced = (
        inp - sys_params["ire0"] - sys_params["track_ire0_offset"][phase]
    ) / sys_params["hz_ire"]
    reduced -= sys_params["vsync_ire"]
    out_scale = OUT_SCALE_DIVIDEND / (100 - sys_params["vsync_ire"])
    return np.uint16(
        np.clip((reduced * out_scale) + sys_params["outputZero"], 0, 65535) + 0.5
    )
def _get_compute_video_filters():
    global _compute_video_filters

    if _compute_video_filters is None:
        from vhsdecode import compute_video_filters as imported_compute_video_filters

        _compute_video_filters = imported_compute_video_filters

    return _compute_video_filters


def _get_format_params(system, tape_format, logger):
    from vhsdecode.formats import get_format_params

    return get_format_params(system, tape_format, 0, logger)


def _get_sub_deemphasis_inner():
    global _sub_deemphasis_inner

    if _sub_deemphasis_inner is None:
        from vhsdecode.nonlinear_filter import (
            sub_deemphasis_inner as imported_sub_deemphasis_inner,
        )

        _sub_deemphasis_inner = imported_sub_deemphasis_inner

    return _sub_deemphasis_inner


#####


def readVHSFrame(f, nr, w, h):
    out = None
    offset = (nr * (2 * w * h) - 1024) * 4
    # offset=0
    try:
        # 1638400
        out = np.fromfile(f, dtype=np.single, count=1638400, offset=offset)
    except:
        print("Failed to load VHS frame from tbc file")
    return out


def readRefFrame(f, nr, w, h, offset=0):
    outa = None
    outb = None
    if not f:
        return None, None
    try:
        out_video = np.fromfile(
            f,
            dtype=np.uint16,
            count=(2 * w * h),
            offset=(nr - offset) * (2 * w * h) * 2,
        )
        outa = out_video[0 : (w * h)]
        outb = out_video[(w * h) : (2 * w * h)]
    except:
        print("Failed to load reference frame from tbc file")
    return outa, outb


def read_json_params(file):
    f = open(file)
    json_data = json.load(f)

    return json_data


def metadata_from_json(json_data):
    video_parameters = json_data["videoParameters"]
    system = video_parameters["system"]
    fs = video_parameters["sampleRate"]
    field_width = video_parameters["fieldWidth"]
    field_height = video_parameters["fieldHeight"]
    if system == "PAL" and field_height < 312:
        system = "MPAL"
    return (system, fs, field_width, field_height)


def _gen_sub_emphasis_params_from_sliders(format_params, filter_params):
    compute_video_filters = _get_compute_video_filters()
    rf_params = {
        "nonlinear_exp_scaling": filter_params["nonlinear_exponential_scale"]["value"],
        "nonlinear_scaling_1": filter_params["nonlinear_linear_scale"]["value"],
        "nonlinear_scaling_2": filter_params["nonlinear_linear_scale_b"]["value"],
        "nonlinear_logistic_mid": filter_params["nonlinear_logistic_mid"]["value"],
        "nonlinear_logistic_rate": filter_params["nonlinear_logistic_rate"]["value"],
        "static_factor": filter_params["nonlinear_static_factor"]["value"],
        "nonlinear_deviation": 2,
    }

    return compute_video_filters.create_sub_emphasis_params(
        rf_params,
        format_params.sys_params,
        format_params.sys_params["hz_ire"],
        format_params.sys_params["vsync_ire"],
    )


class FilterPlot:
    emphasis_reference = {
        "SVHS": {
            "levels": [0, -10, -20, -30],
            "apply_main_deemphasis": False,
            "apply_custom_filters": True,
            "x": np.array([200000, 500000, 1000000, 2000000, 3000000, 5000000]),
            "y": np.array(
                [
                    [
                        1.73,
                        1.60,
                        1.04,
                        0.37,
                        0.07,
                        0.06,
                    ],  # Values directly from the spec
                    [1.30, 0.73, -0.69, -1.75, -2.10, -2.02],
                    [0.65, -1.09, -2.86, -4.16, -4.60, -4.43],
                    [0.49, -2.35, -5.30, -7.14, -7.64, -7.34],
                ]
            ),
        },
        "SVHS_total": {
            "levels": [0, -10, -20, -30],
            "apply_main_deemphasis": True,
            "apply_custom_filters": True,
            "x": np.array([200000, 500000, 1000000, 2000000, 3000000, 5000000]),
            "y": np.array(
                [
                    [
                        -3.47426288,
                        -8.65657528,
                        -11.62615626,
                        -13.24248314,
                        -13.74555096,
                        -13.86360561,
                    ],  # Warning! These values could be wrong
                    [
                        -3.90426288,
                        -9.52657528,
                        -13.35615626,
                        -15.36248314,
                        -15.91555096,
                        -15.94360561,
                    ],
                    [
                        -4.55426288,
                        -11.34657528,
                        -15.52615626,
                        -17.77248314,
                        -18.41555096,
                        -18.35360561,
                    ],
                    [
                        -4.71426288,
                        -12.60657528,
                        -17.96615626,
                        -20.75248314,
                        -21.45555096,
                        -21.26360561,
                    ],
                ]
            ),
        },
    }

    def __init__(self, filters, filter_params, format_params, layout, parent):
        from matplotlib.backends.backend_qtagg import FigureCanvas
        from matplotlib.backends.backend_qtagg import (
            NavigationToolbar2QT as NavigationToolbar,
        )
        from matplotlib.figure import Figure
        from vhsdecode.filter_plot import SubEmphPlotter, plot_filters
        self._canvas = None
        self._canvas = FigureCanvas(Figure(figsize=(5, 3)))
        layout.addWidget(self._canvas)
        self._nav_toolbar = None
        self._nav_toolbar = NavigationToolbar(self._canvas, parent)
        layout.addWidget(self._nav_toolbar)

        self.sub_emph_plotter = SubEmphPlotter(
            filters.block_len, format_params.fs, filters.filters
        )
        self._plot_filters = plot_filters

        self.update(filters, filter_params, format_params)

    def update(self, filters, filter_params, format_params, system=""):
        # self._static_ax = self._canvas.figure.subplots()
        # t = np.linspace(0, 10, 501)
        # self._static_ax.plot(t, np.tan(t), ".")
        self._canvas.figure.clear()
        sub_emphasis_params = _gen_sub_emphasis_params_from_sliders(
            format_params, filter_params
        )
        reference = self.emphasis_reference.get(system, None)

        if reference is not None:
            signal_filters = 1
            if reference["apply_main_deemphasis"]:
                signal_filters *= filters.filters["FDeemp"]
            if reference["apply_custom_filters"]:
                signal_filters *= filters.filters["FCustomVideo"]
            self.sub_emph_plotter.update_signal_filters(signal_filters)
        self._plot_filters(
            filters.filters,
            filters.block_len,
            format_params.fs,
            self._canvas.figure,
            self.sub_emph_plotter,
            sub_emphasis_params,
            reference,
        )
        self._canvas.draw()


class FormatParams:
    def __init__(self, system, tape_format, fs, logger):
        self.change_format(system, tape_format, fs, logger)

    def update_from_json(self, tape_format, json_data, logger):
        (system, fs, field_width, field_height) = metadata_from_json(json_data)
        self.change_format(system, tape_format, fs, logger)
        self.field_lines = field_height
        self.field_width = field_width

    def change_format(self, system, tape_format, fs, logger):
        compute_video_filters = _get_compute_video_filters()
        self.fs = fs
        self.system = system
        # TODO: tape speed
        self.sys_params, self.rf_params = _get_format_params(system, tape_format, logger)
        self.field_lines = max(self.sys_params["field_lines"])
        self.field_width = int(np.round(self.sys_params["line_period"] * (fs / 1e6)))

        self.tape_format = tape_format
        # TODO: THis default should be set elsewhere
        self.sys_params["track_ire0_offset"] = self.sys_params.get(
            "track_ire0_offset", [0, 0]
        )
        self.rf_params["deemph_q"] = self.rf_params.get("deemph_q", 0.5)
        self.sub_emphasis_params = compute_video_filters.create_sub_emphasis_params(
            self.rf_params,
            self.sys_params,
            self.sys_params["hz_ire"],
            self.sys_params["vsync_ire"],
        )


class DeemphasisFilters:
    def __init__(self):
        self._filters = {}

    @property
    def filters(self):
        return self._filters

    @property
    def block_len(self):
        return self._block_len

    def update_filters(self, filter_params, rf_params, fs, block_len):
        self._block_len = block_len
        self.update_deemphasis(filter_params, rf_params, fs, block_len)
        self.update_nonlinear_deemphasis(filter_params, fs, block_len)

    def update_deemphasis(self, filter_params, rf_params, fs, block_len):
        compute_video_filters = _get_compute_video_filters()
        if filter_params["video_lpf_supergauss"]["value"]:
            lpf = compute_video_filters.gen_video_lpf_supergauss(
                filter_params["video_lpf_freq"]["value"],
                filter_params["video_lpf_order"]["value"],
                fs / 2,
                block_len,
            )
        else:
            (_, lpf) = compute_video_filters.gen_video_lpf(
                filter_params["video_lpf_freq"]["value"],
                filter_params["video_lpf_order"]["value"],
                fs / 2.0,
                block_len,
            )

        self._filters["FDeemp"] = compute_video_filters.gen_video_main_deemp_fft(
            filter_params["deemph_gain"]["value"],
            filter_params["deemph_mid"]["value"],
            filter_params["deemph_q"]["value"],
            fs,
            block_len,
        )

        self._filters["FVideo"] = (
            compute_video_filters.gen_video_main_deemp_fft(
                filter_params["deemph_gain"]["value"],
                filter_params["deemph_mid"]["value"],
                filter_params["deemph_q"]["value"],
                fs,
                block_len,
            )
            * lpf
        )

        if (
            filter_params["custom_video_filters"]["value"]
            and rf_params.get("video_custom_luma_filters", None) is not None
        ):
            self._filters["FCustomVideo"] = (
                compute_video_filters.gen_custom_video_filters(
                    rf_params["video_custom_luma_filters"],
                    fs,
                    block_len,
                )
            )
            self._filters["FVideo"] *= self._filters["FCustomVideo"]
        else:
            self._filters["FCustomVideo"] = 1

    def update_nonlinear_deemphasis(self, filter_params, fs, block_len):
        compute_video_filters = _get_compute_video_filters()
        bandpass = None
        if filter_params["nonlinear_bandpass_upper"]["value"] != 0:
            bandpass = filter_params["nonlinear_bandpass_upper"]["value"]
        self.filters["NLHighPassF"] = compute_video_filters.gen_nonlinear_bandpass(
            bandpass,
            filter_params["nonlinear_highpass_freq"].get("value", 1000),
            filter_params["nonlinear_bandpass_order"].get("value", 1),
            fs / 2.0,
            block_len,
        )
        self.filters["NLAmplitudeLPF"] = (
            compute_video_filters.gen_nonlinear_amplitude_lpf(
                filter_params["nonlinear_amplitude_lpf"]["value"], fs / 2.0
            )
        )


class WriteTBCWorker(QObject):
    finished = pyqtSignal()
    nextframe = pyqtSignal(int)
    totalFrames = 0
    abortExport = False
    abortExportSig = pyqtSignal()

    def __init__(self, totalFrames):
        super(WriteTBCWorker, self).__init__()
        self.totalFrames = totalFrames
        self.abortExportSig.connect(self.abortExp)

    def abortExp(self):
        self.abortExport = True

    def run(self):
        self.nextframe.emit(1)
        for i in range(1, self.totalFrames - 1):
            if self.abortExport is True:
                self.finished.emit()
                return
            self.nextframe.emit(i)
        self.nextframe.emit(self.totalFrames - 2)
        self.nextframe.emit(self.totalFrames - 2)
        self.finished.emit()


class VHStune(QDialog):
    refTBCFilename = ""
    proTBCFilename = ""
    saveTBCFilename = ""
    title = "VHStune"

    curFrameNr = 1
    totalFrames = 2

    exportWorker = None
    exportThread = None
    outfile_video = None
    saveParamsFilename = ""

    def __init__(self, tape_format, logger, parent=None):
        super(VHStune, self).__init__(
            parent,
            Qt.WindowType.Window
            | Qt.WindowType.WindowMinimizeButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowCloseButtonHint,
        )
        self._logger = logger

        self.filter_params = None
        self._format_params = FormatParams("PAL", tape_format, SAMPLE_RATE, logger)
        self._deemphasis = DeemphasisFilters()
        self._filters_ready = False
        self._filter_plot = None
        self._finish_initialization_scheduled = False

        self.originalPalette = QApplication.palette()

        self.makeFilterParams()
        self.createControlsGroupBox()
        self.createFilterGroupBox()
        self.createRightArea()

        main_layout = QGridLayout()
        left_layout = QVBoxLayout()
        left_layout.addStrut(370)
        # controls_area = QScrollArea(self.controlsGroupBox)
        left_layout.addWidget(self.controlsGroupBox)
        self.filters_area = QScrollArea()
        self.filters_area.setWidget(self.filterGroupBox)
        left_layout.addWidget(self.filters_area)
        main_layout.addLayout(left_layout, 0, 0)
        img_plot_splitter = QSplitter(Qt.Orientation.Horizontal)
        img_widget = QWidget()
        plot_widget = QWidget()
        # main_layout.addLayout(self._right_layout, 0, 1)
        img_widget.setLayout(self._right_layout)
        self.plot_layout = QGridLayout()
        self._plot_placeholder = QLabel("Loading filter plot...")
        self._plot_placeholder.setMinimumSize(400, 240)
        self.plot_layout.addWidget(self._plot_placeholder, 0, 0)
        # main_layout.addLayout(self.plot_layout, 0, 2)
        plot_widget.setLayout(self.plot_layout)
        img_plot_splitter.addWidget(img_widget)
        img_plot_splitter.addWidget(plot_widget)
        main_layout.addWidget(img_plot_splitter, 0, 1)
        main_layout.setRowStretch(0, 1)
        # main_layout.setRowStretch(1, 1)
        main_layout.setColumnStretch(1, 1)
        # main_layout.setColumnStretch(1, 1)
        self.setLayout(main_layout)

        self.setWindowTitle(self.title)

        self.fftData = []

    def showEvent(self, event):
        super().showEvent(event)
        if not self._finish_initialization_scheduled:
            self._finish_initialization_scheduled = True
            QTimer.singleShot(50, self._finish_initialization)

    def _finish_initialization(self):
        self.initFilters()

        if self._plot_placeholder is not None:
            self.plot_layout.removeWidget(self._plot_placeholder)
            self._plot_placeholder.deleteLater()
            self._plot_placeholder = None

        self._filter_plot = FilterPlot(
            self._deemphasis,
            self.filter_params,
            self._format_params,
            self.plot_layout,
            self,
        )
        self._filters_ready = True
        self.adjustSize()

    def _update_format(self):
        self.makeFilterParams()
        self.createFilterGroupBox()
        self.filters_area.setWidget(self.filterGroupBox)
        self.initFilters()
        self.updateImage()

    def makeFilterParams(self):
        compute_video_filters = _get_compute_video_filters()
        rf_params = self._format_params.rf_params

        if self.filter_params is not None:
            # Remove old controls
            for k in self.filter_params.keys():
                if self.filter_params[k]["step"] is None:
                    self.filter_params[k]["ctrl"].disconnect()
                    self.filter_params[k]["ctrl"].deleteLater()
                elif self.filter_params[k]["value"] is not None:
                    self.filter_params[k]["ctrl"].disconnect()
                    self.filter_params[k]["ctrl"].deleteLater()
                    self.filter_params[k]["label"].deleteLater()

        self.filter_params = {
            "video_lpf_freq": {
                "value": rf_params["video_lpf_freq"],
                "step": 5000,
                "min": 1000000,
                "max": 8000000,
                "desc": "Video low pass filter corner freq ({:.0f} Hz):",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "video_lpf_order": {
                "value": rf_params["video_lpf_order"],
                "step": 1,
                "min": 1,
                "max": 20,
                "desc": "Video low pass filter order ({:.2f}):",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "video_lpf_supergauss": {
                "value": rf_params.get("video_lpf_supergauss", False),
                "step": None,
                "desc": "Use supergauss lowpass filter",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "deemph_mid": {
                "value": rf_params["deemph_mid"],
                "step": 5000,
                "min": 10000,
                "max": 4000000,
                "desc": "Deemphasis mid freq ({:.0f} Hz):",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "deemph_gain": {
                "value": rf_params["deemph_gain"],
                "step": 0.5,
                "min": -30,
                "max": 30,
                "desc": "Deemphasis gain ({:.1f} dB):",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "deemph_q": {
                "value": rf_params["deemph_q"],
                "step": 0.05,
                "min": 0.05,
                "max": 5,
                "desc": "Deemphasis Q ({:.2f}):",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "custom_video_filters": {
                "value": rf_params.get("video_custom_luma_filters", None) != None,
                "step": None,
                "desc": "Enable custom linear filters",
                "onchange": [
                    self.update_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_deemph_enable": {
                "value": rf_params.get("use_sub_deemphasis", False),
                "step": None,
                "desc": "Enable non-linear deemphasis",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_deemph_showonly": {
                "value": False,
                "step": None,
                "desc": "Show only subtracted NL deemphasis part",
                "onchange": [self.drawImage],
            },
            "nonlinear_amplitude_showonly": {
                "value": False,
                "step": None,
                "desc": "Show only subtracted NL detected amplitude part",
                "onchange": [self.drawImage],
            },
            "nonlinear_highpass_freq": {
                "value": rf_params.get("nonlinear_highpass_freq", 10000),
                "step": 5000,
                "min": 10000,
                "max": 4000000,
                "desc": "Non-linear high-pass freq ({} Hz):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_bandpass_order": {
                "value": rf_params.get("nonlinear_bandpass_order", 1),
                "step": 1,
                "min": 1,
                "max": 6,
                "desc": "Non-linear high-pass order ({}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_bandpass_upper": {
                "value": rf_params.get("nonlinear_bandpass_upper", 0),
                "step": 5000,
                "min": 0,
                "max": 8000000,
                "desc": "Non-linear bandpass upper freq (0=lowpass) ({} Hz):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_linear_scale": {
                "value": rf_params.get("nonlinear_scaling_1", 1.0),
                "step": 0.01,
                "min": 0.05,
                "max": 10.0,
                "desc": "Non-linear linear scale ({:.2f}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_linear_scale_b": {
                "value": rf_params.get("nonlinear_scaling_2", 1.0),
                "step": 0.01,
                "min": 0.01,
                "max": 5.0,
                "desc": "Non-linear linear scale B ({:.2f}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_exponential_scale": {
                "value": rf_params.get("nonlinear_exp_scaling", 0.25),
                "step": 0.01,
                "min": 0.05,
                "max": 2,
                "desc": "Non-linear exponential scale ({:.2f}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_static_factor": {
                "value": rf_params.get("nonlinear_static_factor", 0.0),
                "step": 0.001,
                "min": 0.0,
                "max": 1.0,
                "desc": "Non-linear static factor ({:.2f}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_amplitude_lpf": {
                "value": rf_params.get(
                    "nonlinear_amp_lpf_freq",
                    compute_video_filters.NONLINEAR_AMP_LPF_FREQ_DEFAULT,
                ),
                "step": 5000,
                "min": 50000,
                "max": 6000000,
                "desc": (
                    "NL deemp Amplitude detector low pass filter corner freq ({:.1f}):"
                ),
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_logistic_mid": {
                "value": rf_params.get("nonlinear_logistic_mid", 0.1),
                "step": 0.001,
                "min": 0.0,
                "max": 0.25,
                "desc": "Non-linear logistic mid-point ({:.3f}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
            "nonlinear_logistic_rate": {
                "value": rf_params.get("nonlinear_logistic_rate", 0.0),
                "step": 0.1,
                "min": 0,
                "max": 100.0,
                "desc": "Non-linear logistic rate (0=off) ({:.1f}):",
                "onchange": [
                    self.update_nl_deemphasis,
                    self.apply_both_deemph_filters,
                    self.drawImage,
                ],
            },
        }

    def initFilters(self):
        self._deemphasis.update_filters(
            self.filter_params,
            self._format_params.rf_params,
            self._format_params.fs,
            BLOCK_LEN,
        )

    def updateFrameNr(self, s):
        if s == 1:
            self.curFrameNr = self.frameSlider.value()
            self.frameSpin.blockSignals(True)
            self.frameSpin.setValue(self.curFrameNr)
            self.frameSpin.blockSignals(False)
        if s == 2:
            self.curFrameNr = self.frameSpin.value()
            self.frameSlider.blockSignals(True)
            self.frameSlider.setValue(self.curFrameNr)
            self.frameSlider.blockSignals(False)
        else:
            if s == 3:
                self.curFrameNr -= 25
            elif s == 4:
                self.curFrameNr -= 1
            elif s == 5:
                self.curFrameNr += 1
            elif s == 6:
                self.curFrameNr += 25
            if self.curFrameNr <= 0:
                self.curFrameNr = 1
            elif self.curFrameNr >= self.totalFrames - 1:
                self.curFrameNr = self.totalFrames - 2
            self.frameSpin.blockSignals(True)
            self.frameSpin.setValue(self.curFrameNr)
            self.frameSpin.blockSignals(False)
            self.frameSlider.blockSignals(True)
            self.frameSlider.setValue(self.curFrameNr)
            self.frameSlider.blockSignals(False)
        self.updateImage()

    def updateImage(self, f=False, k=None):
        if f is True:
            if self.filter_params[k]["step"] is None:
                self.filter_params[k]["value"] = self.filter_params[k][
                    "ctrl"
                ].isChecked()
            else:
                self.filter_params[k]["value"] = (
                    self.filter_params[k]["step"]
                    * self.filter_params[k]["ctrl"].value()
                )
                self.filter_params[k]["label"].setText(
                    self.filter_params[k]["desc"].format(self.filter_params[k]["value"])
                )
            # print(k)
            for p in self.filter_params[k]["onchange"]:
                p()
        else:
            if not self._filters_ready:
                return
            self.loadImage()
            self.apply_both_deemph_filters()
            self.drawImage()
            self.update_filter_plot()

    def openRefTBCFile(self):
        fileName = QFileDialog.getOpenFileName(
            self, "Open reference tbc file", self.refTBCFilename, "tbc files (*.tbc)"
        )
        if fileName is not None and fileName[0] != "":
            self.refTBCFilename = fileName[0]
            self.updateImage()

    def openProTBCFile(self):
        fileName = QFileDialog.getOpenFileName(
            self, "Open to processed tbc file", self.proTBCFilename, "tbc files (*.tbc)"
        )
        if fileName is not None and fileName[0] != "":
            self.proTBCFilename = fileName[0]
            fSize = os.path.getsize(self.proTBCFilename)
            self.totalFrames = math.floor(
                fSize
                / (
                    4
                    * 2
                    * self._format_params.field_width
                    * self._format_params.field_lines
                )
            )
            self.frameSlider.setRange(1, self.totalFrames - 2)
            self.frameSpin.setRange(1, self.totalFrames - 2)
            json_params = read_json_params(self.proTBCFilename + ".json")
            self._format_params.update_from_json(
                self.systemComboBox.currentText(), json_params, self._logger
            )
            self._update_format()

    def saveProTBCFrame(self, frameNr):
        self.curFrameNr = frameNr
        self.frameSpin.setValue(self.curFrameNr)
        self.frameSlider.setValue(self.curFrameNr)
        self.loadImage()
        self.apply_both_deemph_filters()
        self.drawImage()

    def saveProTBCFinished(self):
        self.outfile_video.close()
        self.outfile_video = None
        self.frameSpin.blockSignals(False)
        self.frameSlider.blockSignals(False)
        self.saveProTBCFileButton.setText("Process and save TBC")

    def saveProTBCFile(self):
        if self.outfile_video is None:
            fileName = QFileDialog.getSaveFileName(
                self,
                "Save processed tbc file",
                self.saveTBCFilename,
                "tbc files (*.tbc)",
            )
            if fileName is not None and fileName[0] != "":
                self.outfile_video = open(fileName[0], "wb")
                self.frameSpin.blockSignals(True)
                self.frameSlider.blockSignals(True)
                self.exportThread = QThread()
                self.exportWorker = WriteTBCWorker(self.totalFrames)
                self.abortExport = pyqtSignal()
                self.exportWorker.moveToThread(self.exportThread)
                self.exportThread.started.connect(self.exportWorker.run)
                self.exportWorker.finished.connect(self.exportThread.quit)
                self.exportWorker.finished.connect(self.exportWorker.deleteLater)
                self.exportThread.finished.connect(self.exportThread.deleteLater)
                self.exportWorker.nextframe.connect(
                    self.saveProTBCFrame, Qt.ConnectionType.BlockingQueuedConnection
                )
                self.exportWorker.finished.connect(
                    self.saveProTBCFinished, Qt.ConnectionType.BlockingQueuedConnection
                )
                self.exportThread.start()
                self.saveProTBCFileButton.setText("Cancel TBC export")
        else:
            self.exportWorker.abortExportSig.emit()

    def _build_params_file_payload(self):
        rf_params = {
            "video_lpf_freq": self.filter_params["video_lpf_freq"]["value"],
            "video_lpf_order": self.filter_params["video_lpf_order"]["value"],
            "video_lpf_supergauss": self.filter_params["video_lpf_supergauss"]["value"],
            "deemph_mid": self.filter_params["deemph_mid"]["value"],
            "deemph_gain": self.filter_params["deemph_gain"]["value"],
            "deemph_q": self.filter_params["deemph_q"]["value"],
            "use_sub_deemphasis": self.filter_params["nonlinear_deemph_enable"][
                "value"
            ],
            "nonlinear_highpass_freq": self.filter_params["nonlinear_highpass_freq"][
                "value"
            ],
            "nonlinear_bandpass_order": self.filter_params[
                "nonlinear_bandpass_order"
            ]["value"],
            "nonlinear_bandpass_upper": self.filter_params[
                "nonlinear_bandpass_upper"
            ]["value"],
            "nonlinear_scaling_1": self.filter_params["nonlinear_linear_scale"][
                "value"
            ],
            "nonlinear_scaling_2": self.filter_params["nonlinear_linear_scale_b"][
                "value"
            ],
            "nonlinear_exp_scaling": self.filter_params[
                "nonlinear_exponential_scale"
            ]["value"],
            "nonlinear_static_factor": self.filter_params[
                "nonlinear_static_factor"
            ]["value"],
            "nonlinear_amp_lpf_freq": self.filter_params[
                "nonlinear_amplitude_lpf"
            ]["value"],
            "nonlinear_logistic_mid": self.filter_params["nonlinear_logistic_mid"][
                "value"
            ],
            "nonlinear_logistic_rate": self.filter_params["nonlinear_logistic_rate"][
                "value"
            ],
        }

        if self.filter_params["custom_video_filters"]["value"]:
            rf_params["video_custom_luma_filters"] = self._format_params.rf_params.get(
                "video_custom_luma_filters", None
            )
        else:
            rf_params["video_custom_luma_filters"] = None

        return {"sys_params": {}, "rf_params": rf_params}

    def exportParamsJSONFile(self):
        fileName = QFileDialog.getSaveFileName(
            self,
            "Export params JSON for vhs-decode --params_file",
            self.saveParamsFilename,
            "JSON files (*.json)",
        )
        if fileName is None or fileName[0] == "":
            return

        target_path = fileName[0]
        if not target_path.lower().endswith(".json"):
            target_path += ".json"

        payload = self._build_params_file_payload()
        with open(target_path, "w", encoding="utf-8") as output:
            json.dump(payload, output, indent=2, sort_keys=True)
            output.write("\n")

        self.saveParamsFilename = target_path

    def loadImage(self):
        self.refFieldA, self.refFieldB = readRefFrame(
            self.refTBCFilename,
            self.curFrameNr,
            self._format_params.field_width,
            self._format_params.field_lines,
            self.refOffset.value(),
        )
        self.vhsFrameData = readVHSFrame(
            self.proTBCFilename,
            self.curFrameNr,
            self._format_params.field_width,
            self._format_params.field_lines,
        )
        pos = 0
        self.fftData = []
        if self.vhsFrameData is None:
            return
        while pos < (len(self.vhsFrameData) - BLOCK_LEN):
            self.fftData.append(npfft.rfft(self.vhsFrameData[pos : pos + BLOCK_LEN]))
            pos += BLOCK_LEN - 2048

    def update_deemphasis(self):
        self._deemphasis.update_deemphasis(
            self.filter_params,
            self._format_params.rf_params,
            self._format_params.fs,
            BLOCK_LEN,
        )
        self.update_filter_plot()

    def update_nl_deemphasis(self):
        self._deemphasis.update_nonlinear_deemphasis(
            self.filter_params, self._format_params.fs, BLOCK_LEN
        )
        self.update_filter_plot()

    def update_filter_plot(self):
        if self._filter_plot is None:
            return
        self._filter_plot.update(
            self._deemphasis,
            self.filter_params,
            self._format_params,
            self.systemComboBox.currentText(),
        )

    def apply_both_deemph_filters(self):
        self.applyDeemphFilter()
        self.applyNLDeemphFilterA()

    def applyNLDeemphFilterA(self):
        sub_deemphasis_inner = _get_sub_deemphasis_inner()
        self.NLDeemphAmplitude = []
        self.NLDeemphPart = []
        self.NLProcessed = []

        for b in self.fftData:
            b_deemp = b * self._deemphasis.filters["FVideo"]
            processed, extracted, amplitude = sub_deemphasis_inner(
                npfft.irfft(b_deemp),
                b_deemp,
                self._deemphasis.filters,
                self._format_params.sub_emphasis_params.deviation,
                self.filter_params["nonlinear_exponential_scale"]["value"],
                self.filter_params["nonlinear_linear_scale"]["value"],
                self.filter_params["nonlinear_linear_scale_b"]["value"],
                self.filter_params["nonlinear_static_factor"]["value"],
            )
            self.NLDeemphAmplitude.append(amplitude)
            self.NLDeemphPart.append(extracted)
            self.NLProcessed.append(processed)

    def applyDeemphFilter(self):
        self.ProcessedFrame = []
        for b in self.fftData:
            self.ProcessedFrame.append(
                npfft.irfft(b * self._deemphasis.filters["FVideo"]).real
            )

    def drawImage(self):
        outImage = None
        for i in range(len(self.ProcessedFrame)):
            img = self.ProcessedFrame[i].copy()
            if self.filter_params["nonlinear_deemph_enable"]["value"] is True:
                if self.filter_params["nonlinear_amplitude_showonly"]["value"] is True:
                    img.fill(
                        self._format_params.sys_params["ire0"]
                        - (self._format_params.sys_params["hz_ire"] * 40)
                    )
                    img += (
                        self.NLDeemphAmplitude[i]
                        * self._format_params.sub_emphasis_params.deviation
                    )
                elif self.filter_params["nonlinear_deemph_showonly"]["value"] is True:
                    img.fill(self._format_params.sys_params["ire0"])
                    img += self.NLDeemphPart[i]
                else:
                    img = self.NLProcessed[i]
            if outImage is None:
                outImage = img[1024:31744]
            else:
                outImage = np.append(outImage, img[1024:31744])
        if outImage is None:
            return
        height = self._format_params.field_lines
        width = self._format_params.field_width
        field_samples = height * width

        vhsFieldA = _hz_to_output(
            outImage[0:field_samples],
            self._format_params.sys_params,
            0 ^ int(self.swapTrackFieldChkBox.isChecked()),
            # self.systemComboBox.currentText(),
        )
        vhsFieldB = _hz_to_output(
            outImage[field_samples : field_samples * 2],
            self._format_params.sys_params,
            1 ^ int(self.swapTrackFieldChkBox.isChecked()),
        )
        if self.outfile_video is not None:
            self.outfile_video.write(vhsFieldA)
            self.outfile_video.write(vhsFieldB)
        outFieldA = None
        outFieldB = None
        if self.refFieldA is not None and self.refFieldB is not None:
            height *= 2
            outFieldA = np.append(self.refFieldA, vhsFieldA)
            outFieldB = np.append(self.refFieldB, vhsFieldB)
        else:
            outFieldA = vhsFieldA
            outFieldB = vhsFieldB

        if self.showFieldsChkBox.isChecked() is True:
            outFrame = np.append(outFieldA, outFieldB)
        else:
            outFrame = np.empty((height * 2, width), dtype=np.uint16)
            outFrame[::2, :] = outFieldA.reshape(height, width)
            outFrame[1::2, :] = outFieldB.reshape(height, width)
        pixmap = QPixmap(
            QImage(
                outFrame.data,
                self._format_params.field_width,
                height * 2,
                self._format_params.field_width * 2,
                QImage.Format.Format_Grayscale16,
            )
        )
        if self.displayWidthSlider.value() != 100:
            pixmap = pixmap.scaled(
                QSize(
                    int(
                        self._format_params.field_width
                        * (self.displayWidthSlider.value() / 100.0)
                    ),
                    height * 2,
                ),
                transformMode=Qt.TransformationMode.SmoothTransformation,
            )
        if self.displayHeightSlider.value() != 1:
            pixmap = pixmap.scaled(
                QSize(pixmap.width(), height * 2 * self.displayHeightSlider.value()),
                transformMode=Qt.TransformationMode.FastTransformation,
            )
        self.dispLabel.setPixmap(pixmap)
        self.dispLabel.resize(pixmap.width(), pixmap.height())

    def createRightArea(self):
        self._right_layout = QGridLayout()
        # scroll = QScrollArea()
        # scroll.setAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter)
        self.dispLabel = QLabel()
        self.dispLabel.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored
        )
        # self._disp_label_layout = QVBoxLayout()
        # self.dispLabel.setLayout(self._disp_label_layout)
        # scroll.setWidget(self.dispLabel)
        self.frameSlider = QSlider(Qt.Orientation.Horizontal, self)

        self.frameSlider.setRange(1, 1)
        self.frameSlider.setValue(1)
        self.frameSpin = QSpinBox()

        self.frameSpin.setRange(1, 1)
        self.frameSpin.setValue(1)
        self.frameSlider.valueChanged.connect(lambda: self.updateFrameNr(1))
        self.frameSpin.valueChanged.connect(lambda: self.updateFrameNr(2))
        frameSpinLabel = QLabel("Frame #:")
        frameSpinLabel.setBuddy(self.frameSpin)
        skipBackBtn = QPushButton("<<")
        skipBackBtn.clicked.connect(lambda: self.updateFrameNr(3))
        stepBackBtn = QPushButton("<")
        stepBackBtn.clicked.connect(lambda: self.updateFrameNr(4))
        stepFwdBtn = QPushButton(">")
        stepFwdBtn.clicked.connect(lambda: self.updateFrameNr(5))
        skipFwdBtn = QPushButton(">>")
        skipFwdBtn.clicked.connect(lambda: self.updateFrameNr(6))
        btnLayout = QHBoxLayout()
        btnLayout.addWidget(frameSpinLabel)
        btnLayout.addWidget(self.frameSpin)
        btnLayout.addWidget(skipBackBtn)
        btnLayout.addWidget(stepBackBtn)
        btnLayout.addWidget(stepFwdBtn)
        btnLayout.addWidget(skipFwdBtn)
        # self._right_layout.addWidget(scroll, 0, 0)
        self._disp_scroll_area = QScrollArea()
        self._disp_scroll_area.setWidget(self.dispLabel)

        self._right_layout.addWidget(self._disp_scroll_area, 0, 0)
        self._right_layout.addWidget(self.frameSlider, 1, 0)
        self._right_layout.addLayout(btnLayout, 2, 0)
        self._right_layout.setRowStretch(0, 1)
        self._right_layout.setColumnStretch(0, 1)

    def _change_format(self):
        self._format_params.change_format(
            self._format_params.system,
            self.systemComboBox.currentText(),
            self._format_params.fs,
            self._logger,
        )
        self._update_format()

    def createControlsGroupBox(self):
        self.controlsGroupBox = QGroupBox("Controls")
        layout = QVBoxLayout()
        openRefTBCFileButton = QPushButton("Open reference TBC")
        openRefTBCFileButton.clicked.connect(self.openRefTBCFile)
        openProTBCFileButton = QPushButton("Open TBC to process")
        openProTBCFileButton.clicked.connect(self.openProTBCFile)
        self.saveProTBCFileButton = QPushButton("Process and save TBC")
        self.saveProTBCFileButton.clicked.connect(self.saveProTBCFile)
        exportParamsJSONButton = QPushButton("Export params JSON")
        exportParamsJSONButton.clicked.connect(self.exportParamsJSONFile)

        layout.addWidget(openRefTBCFileButton)
        layout.addWidget(openProTBCFileButton)
        layout.addWidget(self.saveProTBCFileButton)
        layout.addWidget(exportParamsJSONButton)

        self.systemComboBox = QComboBox()
        supported_tape_formats_list = list(SUPPORTED_TAPE_FORMATS)
        index_of_vhs = supported_tape_formats_list.index(
            self._format_params.tape_format
        )
        self.systemComboBox.addItems(supported_tape_formats_list)
        self.systemComboBox.setCurrentIndex(index_of_vhs)
        self.systemComboBox.currentIndexChanged.connect(self._change_format)
        layout.addWidget(self.systemComboBox)

        self.refOffset = QSpinBox()
        self.refOffset.setRange(-10000, 10000)
        self.refOffset.setValue(0)
        refOffsetLabel = QLabel("Offset of reference TBC:")
        refOffsetLabel.setBuddy(self.refOffset)
        self.refOffset.valueChanged.connect(self.updateImage)

        layout.addWidget(refOffsetLabel)
        layout.addWidget(self.refOffset)

        #        self.tbcWidth = QSpinBox()
        #        self.tbcWidth.setRange(1,10000)
        #        self.tbcWidth.setValue(self._format_params.field_width)
        #        tbcWidthLabel = QLabel("TBC width:")
        #        tbcWidthLabel.setBuddy(self.tbcWidth)

        #        layout.addWidget(tbcWidthLabel)
        #        layout.addWidget(self.tbcWidth)

        #        self.tbcHeight = QSpinBox()
        #        self.tbcHeight.setRange(1,10000)
        #        self.tbcHeight.setValue(self._format_params.field_lines)
        #        tbcHeightLabel = QLabel("TBC height (field lines):")
        #        tbcHeightLabel.setBuddy(self.tbcHeight)

        #        layout.addWidget(tbcHeightLabel)
        #        layout.addWidget(self.tbcHeight)

        self.swapTrackFieldChkBox = QCheckBox("Swap track / field (invert track phase)")
        self.swapTrackFieldChkBox.setCheckState(Qt.CheckState.Unchecked)
        self.swapTrackFieldChkBox.stateChanged.connect(self.drawImage)

        layout.addWidget(self.swapTrackFieldChkBox)

        self.showFieldsChkBox = QCheckBox("Show fields separated")
        self.showFieldsChkBox.setCheckState(Qt.CheckState.Unchecked)
        self.showFieldsChkBox.stateChanged.connect(self.updateImage)

        layout.addWidget(self.showFieldsChkBox)

        self.displayWidthSlider = QSlider(
            Qt.Orientation.Horizontal, self.controlsGroupBox
        )
        self.displayWidthSlider.setRange(1, 200)
        self.displayWidthSlider.setValue(100)
        self.displayWidthSlider.setTracking(False)
        self.displayWidthSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.displayWidthSlider.valueChanged.connect(self.drawImage)

        displayWidthLabel = QLabel("Display width (in percent):")
        displayWidthLabel.setBuddy(self.displayWidthSlider)

        layout.addWidget(displayWidthLabel)
        layout.addWidget(self.displayWidthSlider)

        self.displayHeightSlider = QSlider(
            Qt.Orientation.Horizontal, self.controlsGroupBox
        )
        self.displayHeightSlider.setRange(1, 5)
        self.displayHeightSlider.setValue(1)
        self.displayHeightSlider.setTracking(False)
        self.displayHeightSlider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.displayHeightSlider.valueChanged.connect(self.drawImage)

        displayHeightLabel = QLabel("Display height (factor):")
        displayHeightLabel.setBuddy(self.displayHeightSlider)

        layout.addWidget(displayHeightLabel)
        layout.addWidget(self.displayHeightSlider)

        self.controlsGroupBox.setLayout(layout)

    def createFilterGroupBox(self):
        self.filterGroupBox = QGroupBox("Filter settings")
        layout = QVBoxLayout()

        for k in self.filter_params:
            if self.filter_params[k]["step"] is None:
                self.filter_params[k]["ctrl"] = QCheckBox(self.filter_params[k]["desc"])
                self.filter_params[k]["ctrl"].setChecked(self.filter_params[k]["value"])
                self.filter_params[k]["ctrl"].stateChanged.connect(
                    lambda state, x=k: self.updateImage(True, x)
                )
                layout.addWidget(self.filter_params[k]["ctrl"])
            elif self.filter_params[k]["value"] is not None:
                self.filter_params[k]["ctrl"] = QSlider(
                    Qt.Orientation.Horizontal, self.filterGroupBox
                )
                self.filter_params[k]["ctrl"].setTracking(False)
                self.filter_params[k]["ctrl"].setRange(
                    int(self.filter_params[k]["min"] / self.filter_params[k]["step"]),
                    int(self.filter_params[k]["max"] / self.filter_params[k]["step"]),
                )
                self.filter_params[k]["ctrl"].setValue(
                    int(self.filter_params[k]["value"] / self.filter_params[k]["step"])
                )
                self.filter_params[k]["ctrl"].valueChanged.connect(
                    lambda state, x=k: self.updateImage(True, x)
                )
                self.filter_params[k]["label"] = QLabel(
                    self.filter_params[k]["desc"].format(self.filter_params[k]["value"])
                )
                self.filter_params[k]["label"].setBuddy(self.filter_params[k]["ctrl"])
                layout.addWidget(self.filter_params[k]["label"])
                layout.addWidget(self.filter_params[k]["ctrl"])

        layout.addStretch(1)
        self.filterGroupBox.setLayout(layout)

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


def main():
    if QT_VERSION == 5:
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    os.environ["QT_STYLE_OVERRIDE"] = "Fusion"
    app = QApplication(sys.argv)
    _apply_fusion_dark_mode(app)
    logger = logging.getLogger("vhstune")
    tape_format = "VHS"
    if len(sys.argv) > 1:
        tape_format = sys.argv[1]
    vhsTune = VHStune(tape_format, logger)
    vhsTune.show()
    pos = vhsTune.pos()
    if pos.x() < 0 or pos.y() < 0:
        vhsTune.move(0, 0)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
