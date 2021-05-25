/************************************************************************

    mainwindow.h

    ld-analyse - TBC output analysis
    Copyright (C) 2018-2021 Simon Inns

    This file is part of ld-decode-tools.

    ld-analyse is free software: you can redistribute it and/or
    modify it under the terms of the GNU General Public License as
    published by the Free Software Foundation, either version 3 of the
    License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

************************************************************************/

#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <QtDebug>
#include <QStandardPaths>
#include <QDir>
#include <QFileDialog>
#include <QMessageBox>
#include <QLabel>
#include <QMouseEvent>

#include "oscilloscopedialog.h"
#include "aboutdialog.h"
#include "vbidialog.h"
#include "dropoutanalysisdialog.h"
#include "blacksnranalysisdialog.h"
#include "whitesnranalysisdialog.h"
#include "busydialog.h"
#include "closedcaptionsdialog.h"
#include "chromadecoderconfigdialog.h"
#include "configuration.h"
#include "tbcsource.h"

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QString inputFilenameParam, QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    // Menu bar handlers
    void on_actionExit_triggered();
    void on_actionOpen_TBC_file_triggered();
    void on_actionReload_TBC_triggered();
    void on_actionLine_scope_triggered();
    void on_actionAbout_ld_analyse_triggered();
    void on_actionVBI_triggered();
    void on_actionDropout_analysis_triggered();
    void on_actionSNR_analysis_triggered();
    void on_actionSave_frame_as_PNG_triggered();
    void on_actionZoom_In_triggered();
    void on_actionZoom_Out_triggered();
    void on_actionZoom_1x_triggered();
    void on_actionZoom_2x_triggered();
    void on_actionZoom_3x_triggered();
    void on_actionClosed_Captions_triggered();
    void on_actionChroma_decoder_configuration_triggered();

    // Media control frame handlers
    void on_previousPushButton_clicked();
    void on_nextPushButton_clicked();
    void on_endFramePushButton_clicked();
    void on_startFramePushButton_clicked();
    void on_frameNumberSpinBox_editingFinished();
    void on_frameHorizontalSlider_valueChanged(int value);
    void on_videoPushButton_clicked();
    void on_dropoutsPushButton_clicked();
    void on_fieldOrderPushButton_clicked();
    void on_zoomInPushButton_clicked();
    void on_zoomOutPushButton_clicked();
    void on_originalSizePushButton_clicked();
    void on_mouseModePushButton_clicked();
    void on_aspectPushButton_clicked();

    // Miscellaneous handlers
    void scanLineChangedSignalHandler(qint32 scanLine, qint32 pictureDot);
    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);
    void chromaDecoderConfigChangedSignalHandler();

    // Tbc Source signal handlers
    void on_busyLoading(QString infoMessage);
    void on_finishedLoading();    

    void on_actionWhite_SNR_analysis_triggered();

private:
    Ui::MainWindow *ui;

    // Dialogues
    OscilloscopeDialog* oscilloscopeDialog;
    AboutDialog* aboutDialog;
    VbiDialog* vbiDialog;
    DropoutAnalysisDialog* dropoutAnalysisDialog;
    BlackSnrAnalysisDialog* blackSnrAnalysisDialog;
    WhiteSnrAnalysisDialog* whiteSnrAnalysisDialog;
    BusyDialog* busyDialog;
    ClosedCaptionsDialog *closedCaptionDialog;
    ChromaDecoderConfigDialog *chromaDecoderConfigDialog;

    // Class globals
    Configuration configuration;
    QLabel sourceVideoStatus;
    QLabel fieldNumberStatus;
    TbcSource tbcSource;
    qint32 aspectRatio;
    qint32 lastScopeLine;
    qint32 lastScopeDot;
    qint32 currentFrameNumber;
    qreal scaleFactor;
    QPalette buttonPalette;
    QString lastFilename;

    // Update GUI methods
    void updateGuiLoaded();
    void updateGuiUnloaded();

    // Frame display methods
    void showFrame();
    void updateFrameViewer();
    void hideFrame();

    // TBC source signal handlers
    void loadTbcFile(QString inputFileName);
    void updateOscilloscopeDialogue(qint32 scanLine, qint32 pictureDot);
    void mouseScanLineSelect(qint32 oX, qint32 oY);
};

#endif // MAINWINDOW_H
