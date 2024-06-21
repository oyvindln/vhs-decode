/************************************************************************

    videoparametersdialog.cpp

    ld-analyse - TBC output analysis
    Copyright (C) 2022 Adam Sampson

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

#include "videoparametersdialog.h"
#include "ui_videoparametersdialog.h"

VideoParametersDialog::VideoParametersDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VideoParametersDialog)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);
    
    // Update the dialogue
    updateDialog();
}

VideoParametersDialog::~VideoParametersDialog()
{
    delete ui;
}

void VideoParametersDialog::setVideoParameters(const LdDecodeMetaData::VideoParameters &_videoParameters)
{
    videoParameters = _videoParameters;
    originalActiveVideoStart = videoParameters.activeVideoStart;
    originalActiveVideoWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;

    // Transfer values to the dialogue
    ui->blackLevelSpinBox->setValue(videoParameters.black16bIre);
    ui->whiteLevelSpinBox->setValue(videoParameters.white16bIre);
    ui->activeVideoWidthSpinBox->setValue(videoParameters.activeVideoEnd - videoParameters.activeVideoStart);
    ui->activeVideoStartSpinBox->setValue(videoParameters.activeVideoStart);
    if (videoParameters.isWidescreen) ui->aspectRatio169RadioButton->setChecked(true);
    else ui->aspectRatio43RadioButton->setChecked(true);

    // Update the dialogue
    updateDialog();
}

void VideoParametersDialog::updateDialog()
{
    // Adjust the black level reset buttons depending on whether the system is NTSC
    if (videoParameters.system == NTSC) {
        ui->blackLevelResetButton->setText("Reset NTSC");
        ui->blackLevelAltResetButton->setText("Reset NTSC-J");
        ui->blackLevelAltResetButton->show();
    } else {
        ui->blackLevelResetButton->setText("Reset");
        ui->blackLevelAltResetButton->hide();
    }
}

// Public slots

// Set either black or white level, depending on which half of the range the value is in
void VideoParametersDialog::levelSelected(qint32 level)
{
    if (level < 0x8000) {
        ui->blackLevelSpinBox->setValue(level);
    } else {
        ui->whiteLevelSpinBox->setValue(level);
    }
}

// Private slots

void VideoParametersDialog::on_blackLevelSpinBox_valueChanged(int value)
{
    videoParameters.black16bIre = value;
    updateDialog();
    emit videoParametersChanged(videoParameters);
}

void VideoParametersDialog::on_whiteLevelSpinBox_valueChanged(int value)
{
    videoParameters.white16bIre = value;
    updateDialog();
    emit videoParametersChanged(videoParameters);
}

void VideoParametersDialog::on_activeVideoStartSpinBox_valueChanged(int value)
{
    int adjustAmount = videoParameters.activeVideoStart - value;
    videoParameters.activeVideoStart = value;
    videoParameters.activeVideoEnd = value + ui->activeVideoWidthSpinBox->value();
    updateDialog();
    emit videoParametersChanged(videoParameters);
}

void VideoParametersDialog::on_activeVideoWidthSpinBox_valueChanged(int value)
{
    videoParameters.activeVideoEnd = videoParameters.activeVideoStart + value;
    updateDialog();
    emit videoParametersChanged(videoParameters);
}
// The reset black and white levels come from EBU Tech 3280 p6 (PAL) and SMPTE
// 244M p2 (NTSC), and match what ld-decode uses by default.

void VideoParametersDialog::on_blackLevelResetButton_clicked()
{
    if (videoParameters.system == NTSC) {
        ui->blackLevelSpinBox->setValue(0x3C00 + 0x0A80); // including setup
    } else {
        ui->blackLevelSpinBox->setValue(0x4000);
    }
}

void VideoParametersDialog::on_blackLevelAltResetButton_clicked()
{
    ui->blackLevelSpinBox->setValue(0x3C00);
}

void VideoParametersDialog::on_whiteLevelResetButton_clicked()
{
    if (videoParameters.system == NTSC) {
        ui->whiteLevelSpinBox->setValue(0xC800);
    } else {
        ui->whiteLevelSpinBox->setValue(0xD300);
    }
}

void VideoParametersDialog::on_activeVideoStartResetButton_clicked()
{
    ui->activeVideoStartSpinBox->setValue(originalActiveVideoStart);
}

void VideoParametersDialog::on_activeVideoWidthResetButton_clicked()
{
    ui->activeVideoWidthSpinBox->setValue(originalActiveVideoWidth);
}

void VideoParametersDialog::on_aspectRatioButtonGroup_buttonClicked(QAbstractButton *button)
{
    videoParameters.isWidescreen = (button == ui->aspectRatio169RadioButton);
    updateDialog();
    emit videoParametersChanged(videoParameters);
}
