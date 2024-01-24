import os.path
import sys
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QDial, QCheckBox, \
    QComboBox, QPushButton, QLineEdit, QFileDialog, QDesktopWidget, QDialog, QMessageBox
from PyQt5 import QtGui, QtCore
from vhsdecode.hifi.HiFiDecode import DEFAULT_NR_GAIN_


class MainUIParameters:
    def __init__(self):
        self.volume: float = 1.0
        self.sidechain_gain: float = DEFAULT_NR_GAIN_ / 100.0
        self.noise_reduction: bool = True
        self.automatic_fine_tuning: bool = True
        self.grc = False
        self.preview: bool = False
        self.audio_sample_rate: int = 44100
        self.standard: str = "PAL"
        self.format: str = "VHS"
        self.audio_mode: str = "Stereo"
        self.input_sample_rate: float = 40.0
        self.input_file: str = ""
        self.output_file: str = ""


def decode_options_to_ui_parameters(decode_options):
    values = MainUIParameters()
    values.volume = decode_options["gain"]
    values.sidechain_gain = decode_options["nr_side_gain"] / 100.0
    values.noise_reduction = decode_options["noise_reduction"]
    values.automatic_fine_tuning = decode_options["auto_fine_tune"]
    values.audio_sample_rate = decode_options["audio_rate"]
    values.standard = "PAL" if decode_options["standard"] == "p" else "NTSC"
    values.format = "VHS" if decode_options["format"] == "vhs" else "Video8/Hi8"
    values.audio_mode = "Stereo"
    values.input_sample_rate = decode_options["input_rate"]
    values.input_file = decode_options["input_file"]
    values.output_file = decode_options["output_file"]
    return values


def ui_parameters_to_decode_options(values: MainUIParameters):
    decode_options = {
        "input_rate": float(values.input_sample_rate) * 1e6,
        "standard": "p" if values.standard == "PAL" else "n",
        "format": "vhs" if values.format == "VHS" else "h8",
        "preview": values.preview,
        "original": False,
        "noise_reduction": values.noise_reduction,
        "auto_fine_tune": values.automatic_fine_tuning,
        "nr_side_gain": values.sidechain_gain * 100.0,
        "grc": values.grc,
        "audio_rate": values.audio_sample_rate,
        "gain": values.volume,
        "input_file": values.input_file,
        "output_file": values.output_file,
        "mode": "s" if values.audio_mode == "Stereo"
        else "l" if values.audio_mode == "L"
        else "r" if values.audio_mode == "R"
        else "mpx" if values.audio_mode == "Stereo MPX"
        else "sum"
    }
    return decode_options


class InputDialog(QDialog):
    def __init__(self, prompt: str, value=None, validator=None):
        super().__init__()
        self.validator = validator
        self.prompt = prompt
        self.value = value
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle('Input Dialog')

        layout = QVBoxLayout()

        label = QLabel(self.prompt, self)
        self.input_line = QLineEdit(self)
        self.input_line.setText(str(self.value))
        if self.validator is not None:
            self.input_line.setValidator(self.validator)
        ok_button = QPushButton('Accept', self)
        cancel_button = QPushButton('Cancel', self)
        button_layout = QHBoxLayout()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        ok_button.clicked.connect(self.accept)
        cancel_button.clicked.connect(self.reject)

        layout.addWidget(label)
        layout.addWidget(self.input_line)
        layout.addLayout(button_layout)

        self.setLayout(layout)

        self.setStyleSheet("""
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
        """)

        # Set dialog size
        self.setFixedWidth(int(self.sizeHint().width()))
        self.setFixedHeight(self.sizeHint().height())

    def get_input_value(self):
        result = self.exec_()  # Ejecutar el diálogo de entrada
        if result == QDialog.Accepted:
            return self.input_line.text()
        else:
            return None


class HifiUi(QMainWindow):
    def __init__(self, params: MainUIParameters, title: str = "HiFi Main Controls",
                 main_layout_callback=None):
        super(HifiUi, self).__init__()

        # 0 stop, 1 play, 2 pause
        self._transport_state = 0

        # Set up the main window
        self.setWindowTitle(title)
        self.setGeometry(100, 100, 400, 200)

        # Create central widget and layout
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        self.main_layout: QHBoxLayout = QVBoxLayout(central_widget)
        main_layout_callback(self.main_layout)

        # Upper partition layout (horizontal)
        upper_layout = QHBoxLayout()
        self.main_layout.addLayout(upper_layout)

        # Volume dial and numeric textbox
        volume_label = QLabel("Volume:")
        self.volume_dial = QDial(self)
        self.volume_dial.setRange(0, 100)
        self.volume_textbox = QLineEdit(self)
        self.volume_textbox.setValidator(QtGui.QDoubleValidator())  # Only allow float input
        self.volume_textbox.setMaxLength(5)  # Set a reasonable maximum length for display

        # Sidechain gain dial and numeric textbox
        sidechain_label = QLabel("NR Gain:")
        self.sidechain_dial = QDial(self)
        self.sidechain_dial.setRange(0, 100)
        self.sidechain_textbox = QLineEdit(self)
        self.sidechain_textbox.setValidator(QtGui.QDoubleValidator())  # Only allow float input
        self.sidechain_textbox.setMaxLength(5)  # Set a reasonable maximum length for display

        # Add widgets to the upper layout
        upper_layout.addWidget(volume_label)
        upper_layout.addWidget(self.volume_dial)
        upper_layout.addWidget(self.volume_textbox)
        upper_layout.addWidget(sidechain_label)
        upper_layout.addWidget(self.sidechain_dial)
        upper_layout.addWidget(self.sidechain_textbox)

        # Middle partition layout (vertical)
        middle_layout = QVBoxLayout()
        self.main_layout.addLayout(middle_layout)

        # Checkboxes
        self.noise_reduction_checkbox = QCheckBox("Noise reduction")
        self.automatic_fine_tuning_checkbox = QCheckBox("Automatic fine tuning")
        self.preview_checkbox = QCheckBox("Preview")
        middle_layout.addWidget(self.noise_reduction_checkbox)
        middle_layout.addWidget(self.automatic_fine_tuning_checkbox)
        middle_layout.addWidget(self.preview_checkbox)

        samplerate_layout = QHBoxLayout()
        # Sample rate options dropdown
        sample_rate_label = QLabel("Audio Sample Rate Hz:")
        self.sample_rate_combo = QComboBox(self)
        self.sample_rate_combo.addItems(["44100", "48000", "96000", "192000"])
        samplerate_layout.addWidget(sample_rate_label)
        samplerate_layout.addWidget(self.sample_rate_combo)

        # Adds standard (pal/ntsc) layout
        standard_layout = QHBoxLayout()
        standard_label = QLabel("Standard:")
        self.standard_combo = QComboBox(self)
        self.standard_combo.addItems(["PAL", "NTSC"])
        standard_layout.addWidget(standard_label)
        standard_layout.addWidget(self.standard_combo)

        # Adds format (vhs/video8) layout
        format_layout = QHBoxLayout()
        format_label = QLabel("Format:")
        self.format_combo = QComboBox(self)
        self.format_combo.addItems(["VHS", "Video8/Hi8"])
        format_layout.addWidget(format_label)
        format_layout.addWidget(self.format_combo)

        # Adds audio mode (mono L/mono R/stereo/stereo mpx) layout
        audio_mode_layout = QHBoxLayout()
        audio_mode_label = QLabel("Audio Mode:")
        self.audio_mode_combo = QComboBox(self)
        self.audio_mode_combo.addItems(["Stereo", "L", "R", "Stereo MPX", "Sum"])
        audio_mode_layout.addWidget(audio_mode_label)
        audio_mode_layout.addWidget(self.audio_mode_combo)

        # adds input sample rate layout
        input_samplerate_layout = QHBoxLayout()
        input_samplerate_label = QLabel("Input Sample Rate MHz:")
        self.input_samplerate_combo = QComboBox(self)
        self.input_samplerate_combo.addItems(
            ["DdD (40.0)",
             "cxadc (28.64)",
             "cxadc3 (35.8)",
             "10cxadc (14.32)",
             "10cxadc3 (17.9)",
             "Other"]
        )
        self._input_combo_rates = [
            40.0,
            28.64,
            35.8,
            14.32,
            17.9,
        ]
        self.input_sample_rate = self._input_combo_rates[0]
        input_samplerate_layout.addWidget(input_samplerate_label)
        input_samplerate_layout.addWidget(self.input_samplerate_combo)
        self.input_samplerate_combo.currentIndexChanged.connect(self.on_input_samplerate_changed)

        middle_layout.addLayout(input_samplerate_layout)
        middle_layout.addLayout(standard_layout)
        middle_layout.addLayout(format_layout)
        middle_layout.addLayout(audio_mode_layout)
        middle_layout.addLayout(samplerate_layout)

        # Bottom partition layout (horizontal)
        bottom_layout = QHBoxLayout()
        self.main_layout.addLayout(bottom_layout)

        # Playback controls
        self.play_button = QPushButton("▶", self)  # Play symbol
        self.pause_button = QPushButton("||", self)  # Pause symbol
        self.stop_button = QPushButton("■", self)  # Stop symbol
        max_button_height = max(self.play_button.sizeHint().height(), self.pause_button.sizeHint().height(), self.stop_button.sizeHint().height())
        self.play_button.setFixedHeight(max_button_height)
        self.pause_button.setFixedHeight(max_button_height)
        self.stop_button.setFixedHeight(max_button_height)
        bottom_layout.addWidget(self.play_button)
        bottom_layout.addWidget(self.pause_button)
        bottom_layout.addWidget(self.stop_button)

        # Connect events to functions
        self.volume_dial.valueChanged.connect(self.on_volume_changed)
        self.volume_textbox.editingFinished.connect(self.on_volume_textbox_changed)
        self.sidechain_dial.valueChanged.connect(self.on_sidechain_gain_changed)
        self.sidechain_textbox.editingFinished.connect(self.on_sidechain_textbox_changed)
        self.play_button.clicked.connect(self.on_play_clicked)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        self.stop_button.clicked.connect(self.on_stop_clicked)

        # Apply dark theme with improved text visibility
        self.setStyleSheet("""
            QMainWindow {
                background-color: #333;
                color: #eee;
            }
            QDial, QLineEdit, QCheckBox, QComboBox, QPushButton {
                background-color: #555;
                color: #eee;
                border: 1px solid #777;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox::item {
                background-color: #555;
                color: #eee;
            }
            QCheckBox::indicator {
                width: 15px;
                height: 15px;
            }
            QLabel {
                color: #eee;
            }
        """)
        self.change_button_color(self.stop_button, "#eee")
        # disables maximize button
        self.setWindowFlags(self.windowFlags() & ~QtCore.Qt.WindowMaximizeButtonHint)
        # disables resize
        self.setFixedWidth(int(self.sizeHint().width() * 3 / 4))
        # sets fixed height
        self.setFixedHeight(self.sizeHint().height())
        # sets defaut window icon
        self.setWindowIcon(QIcon.fromTheme('document-open'))
        self.center_on_screen()
        self.setValues(params)

    @property
    def transport_state(self):
        return self._transport_state if self.isVisible() else 0

    @transport_state.setter
    def transport_state(self, value):
        if value == 0:
            self.on_stop_clicked()
        elif value == 1:
            self.on_play_clicked()
        elif value == 2:
            self.on_pause_clicked()
        else:
            raise ValueError("Invalid transport state value")
        self._transport_state = value

    def center_on_screen(self):
        # Get the screen geometry
        screen_geometry = QDesktopWidget().screenGeometry()

        # Calculate the center of the screen
        screen_center = screen_geometry.center()

        # Set the window position to the center of the screen
        self.move(
            int(screen_center.x() - self.width() / 2),
            int((screen_center.y() - self.height()) * 3 / 4)
        )

    def setValues(self, values: MainUIParameters):
        self.volume_dial.setValue(int(values.volume * 100 / 2))
        self.volume_textbox.setText(str(values.volume))
        self.sidechain_dial.setValue(int(values.sidechain_gain * 100))
        self.sidechain_textbox.setText(str(values.sidechain_gain))
        self.noise_reduction_checkbox.setChecked(values.noise_reduction)
        self.automatic_fine_tuning_checkbox.setChecked(values.automatic_fine_tuning)
        self.sample_rate_combo.setCurrentText(str(values.audio_sample_rate))
        self.sample_rate_combo.setCurrentIndex(self.sample_rate_combo.findText(str(values.audio_sample_rate)))
        self.standard_combo.setCurrentText(values.standard)
        self.standard_combo.setCurrentIndex(self.standard_combo.findText(values.standard))
        self.format_combo.setCurrentText(values.format)
        self.format_combo.setCurrentIndex(self.format_combo.findText(values.format))
        self.audio_mode_combo.setCurrentText(values.audio_mode)
        self.audio_mode_combo.setCurrentIndex(self.audio_mode_combo.findText(values.audio_mode))

        self.input_sample_rate = values.input_sample_rate / 1e6
        found_rate = False
        for i, rate in enumerate(self._input_combo_rates):
            if abs((rate * 1e6) - values.input_sample_rate) < 5000:
                self.input_samplerate_combo.setCurrentIndex(i)
                found_rate = True
                break
        if not found_rate:
            self.input_samplerate_combo.setCurrentText(f'Other ({values.input_sample_rate / 1e6:.2f})')

        self.input_file = values.input_file
        self.output_file = values.output_file
        self.preview_checkbox.setChecked(values.preview)

    def getValues(self) -> MainUIParameters:
        values = MainUIParameters()
        values.volume = float(self.volume_textbox.text())
        values.sidechain_gain = float(self.sidechain_textbox.text())
        values.noise_reduction = self.noise_reduction_checkbox.isChecked()
        values.automatic_fine_tuning = self.automatic_fine_tuning_checkbox.isChecked()
        values.audio_sample_rate = int(self.sample_rate_combo.currentText())
        values.standard = self.standard_combo.currentText()
        values.format = self.format_combo.currentText()
        values.audio_mode = self.audio_mode_combo.currentText()
        values.input_sample_rate = self.input_sample_rate
        values.input_file = self.input_file
        values.output_file = self.output_file
        values.preview = self.preview_checkbox.isChecked()
        return values

    def on_volume_changed(self, value):
        self.volume_textbox.setText(str(value * 2 / 100.0))

    def on_volume_textbox_changed(self):
        text = self.volume_textbox.text()
        try:
            value = float(text)
            self.volume_dial.setValue(int(value * 100 / 2))
        except ValueError:
            pass

    def on_sidechain_gain_changed(self, value):
        self.sidechain_textbox.setText(str(value / 100.0))

    def on_sidechain_textbox_changed(self):
        text = self.sidechain_textbox.text()
        try:
            value = float(text)
            self.sidechain_dial.setValue(int(value * 100))
        except ValueError:
            pass

    def change_button_color(self, button, color):
        button.setStyleSheet("""
            QPushButton {
                background-color: %s;
                color: #000;
            }
        """ % color)

    def default_button_color(self, button):
        button.setStyleSheet("""
            QPushButton {
                background-color: #555;
                color: #eee;
            }
        """)

    def on_play_clicked(self):
        print("▶ Play command issued.")
        # checks if destination file exists and prompts user to overwrite
        if os.path.exists(self.output_file) and self.transport_state == 0:
            message_box = QMessageBox(
                QMessageBox.Question,
                'File Exists',
                'Overwrite existing file?',
                buttons=QMessageBox.Yes | QMessageBox.No,
                parent=self
            )
            message_box.setIcon(QMessageBox.Warning)
            message_box.setDefaultButton(QMessageBox.No)
            # dark mode
            message_box.setStyleSheet("""
                QMessageBox {
                    background-color: #333;
                    color: #eee;
                }
                QPushButton {
                    background-color: #555;
                    color: #eee;
                }
            """)
            message_box.exec_()
            overwrite = message_box.standardButton(message_box.clickedButton())

            if overwrite == QMessageBox.No:
                print("Overwrite cancelled.")
                return

        self._transport_state = 1
        self.change_button_color(self.play_button, "#0f0")
        self.default_button_color(self.pause_button)
        self.default_button_color(self.stop_button)
        self.setWindowIcon(QIcon.fromTheme('document-save'))

    def on_pause_clicked(self):
        self.change_button_color(self.pause_button, "#eee")
        self.default_button_color(self.play_button)
        self.default_button_color(self.stop_button)
        self.setWindowIcon(QIcon.fromTheme('document-open'))
        print("|| Pause command issued.")
        self._transport_state = 2

    def on_stop_clicked(self):
        self.change_button_color(self.stop_button, "#eee")
        self.default_button_color(self.play_button)
        self.default_button_color(self.pause_button)
        print("■ Stop command issued.")
        self._transport_state = 0

    def on_input_samplerate_changed(self):
        print("Input sample rate changed.")
        if 'Other' in self.input_samplerate_combo.currentText():
            input_dialog = InputDialog(
                prompt='Input Sample Rate (MHz)',
                value=self.input_sample_rate,
                validator=QtGui.QDoubleValidator()
            )
            value: float = input_dialog.get_input_value()
            if value is not None:
                new_other_text = f'Other ({value})'
                self.input_samplerate_combo.setItemText(self.input_samplerate_combo.currentIndex(), new_other_text)
                self.input_samplerate_combo.setCurrentText(value)
                self.input_sample_rate = value
        else:
            self.input_sample_rate = self._input_combo_rates[self.input_samplerate_combo.currentIndex()]

    def on_decode_finished(self, decoded_filename: str = 'input stream'):
        self.setWindowIcon(QIcon.fromTheme('document-open'))
        # alerts user that decode is finished
        message_box = QMessageBox(
            QMessageBox.Information,
            'Decode Finished',
            f'Decode of {decoded_filename} finished.',
            parent=self
        )
        message_box.setIcon(QMessageBox.Information)
        message_box.setDefaultButton(QMessageBox.Ok)
        # dark mode
        message_box.setStyleSheet("""
            QMessageBox {
                background-color: #333;
                color: #eee;
            }
            QPushButton {
                background-color: #555;
                color: #eee;
            }
        """)
        message_box.exec_()


class FileOutputDialogUI(HifiUi):
    def __init__(self, params: MainUIParameters, title: str = "HiFi File Output", main_layout_callback=None):
        super(FileOutputDialogUI, self).__init__(params, title, self._layout_callback)

    def _layout_callback(self, main_layout):
        # Add file output widgets
        self.file_output_label = QLabel("Output file:")
        self.file_output_textbox = QLineEdit(self)
        self.file_output_button = QPushButton("Browse", self)
        self.file_output_button.clicked.connect(self.on_file_output_button_clicked)
        self.file_output_layout = QHBoxLayout()
        self.file_output_layout.addWidget(self.file_output_label)
        self.file_output_layout.addWidget(self.file_output_textbox)
        self.file_output_layout.addWidget(self.file_output_button)
        self.main_layout.addLayout(self.file_output_layout)
        #sets browse button height to match text box height
        self.file_output_button.setFixedHeight(self.file_output_textbox.sizeHint().height())
        self.file_output_textbox.setFixedHeight(self.file_output_textbox.sizeHint().height())


    def on_file_output_button_clicked(self):
        options = QFileDialog.Options()
        qdialog = QFileDialog(self)
        qdialog.setFileMode(QFileDialog.AnyFile)
        if os.path.isdir(os.path.dirname(self.file_output_textbox.text())):
            qdialog.setDirectory(os.path.dirname(self.file_output_textbox.text()))

        file_name, _ = qdialog.getOpenFileName(self, "Open File", "", "All Files (*);;FLAC (*.flac)",
                                                   options=options)
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

    def on_decode_finished(self, decoded_filename: str = 'input stream'):
        decoded_filename = os.path.basename(self.file_output_textbox.text())
        super(FileOutputDialogUI, self).on_decode_finished(decoded_filename)


class FileIODialogUI(HifiUi):
    def __init__(self, params: MainUIParameters, title: str = "HiFi File Input/Output", main_layout_callback=None):
        super(FileIODialogUI, self).__init__(params, title, self._layout_callback)

    def _layout_callback(self, main_layout):
        # Add file input widgets
        self.file_input_label = QLabel("Input file:")
        self.file_input_textbox = QLineEdit(self)
        self.file_input_button = QPushButton("Browse", self)
        self.file_input_button.clicked.connect(self.on_file_input_button_clicked)
        self.file_input_layout = QHBoxLayout()
        self.file_input_layout.addWidget(self.file_input_label)
        self.file_input_layout.addWidget(self.file_input_textbox)
        self.file_input_layout.addWidget(self.file_input_button)
        self.main_layout.addLayout(self.file_input_layout)
        #sets browse button height to match text box height
        self.file_input_button.setFixedHeight(self.file_input_textbox.sizeHint().height())
        self.file_input_textbox.setFixedHeight(self.file_input_textbox.sizeHint().height())

        # Add file output widgets
        self.file_output_label = QLabel("Output file:")
        self.file_output_textbox = QLineEdit(self)
        self.file_output_button = QPushButton("Browse", self)
        self.file_output_button.clicked.connect(self.on_file_output_button_clicked)
        self.file_output_layout = QHBoxLayout()
        self.file_output_layout.addWidget(self.file_output_label)
        self.file_output_layout.addWidget(self.file_output_textbox)
        self.file_output_layout.addWidget(self.file_output_button)
        self.main_layout.addLayout(self.file_output_layout)
        #sets browse button height to match text box height
        self.file_output_button.setFixedHeight(self.file_output_textbox.sizeHint().height())
        self.file_output_textbox.setFixedHeight(self.file_output_textbox.sizeHint().height())
        # sets file input and output text boxes to same width
        max_label_width = max(self.file_input_label.sizeHint().width(), self.file_output_label.sizeHint().width())
        self.file_input_label.setMinimumWidth(max_label_width)
        self.file_output_label.setMinimumWidth(max_label_width)

    def on_file_input_button_clicked(self):
        options = QFileDialog.Options()
        qdialog = QFileDialog(self)
        qdialog.setFileMode(QFileDialog.AnyFile)
        if os.path.isdir(os.path.dirname(self.file_input_textbox.text())):
            qdialog.setDirectory(os.path.dirname(self.file_input_textbox.text()))
        file_name, _ = qdialog.getOpenFileName(
            self, "Open File", "", "All Files (*);;FLAC (*.flac)",
            options=options)

        if os.path.exists(file_name):
            self.file_input_textbox.setText(file_name)
        print("Input browse button clicked.")

    def on_file_output_button_clicked(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(self, "Save File", "", "All Files (*);;FLAC (*.flac);; WAV (*.wav)",
                                                   options=options)

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
        self.file_output_textbox.setText(values.output_file)

    def on_decode_finished(self, decoded_filename: str = 'input stream'):
        decoded_filename = os.path.basename(self.file_input_textbox.text())
        super(FileIODialogUI, self).on_decode_finished(decoded_filename)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    params = MainUIParameters()
    window = FileIODialogUI(params)
    window.show()
    sys.exit(app.exec_())
