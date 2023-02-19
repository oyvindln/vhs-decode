/************************************************************************

    vbidialog.cpp

    ld-analyse - TBC output analysis
    Copyright (C) 2018-2022 Simon Inns

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

#include "vbidialog.h"
#include "ui_vbidialog.h"

VbiDialog::VbiDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VbiDialog)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);
}

VbiDialog::~VbiDialog()
{
    delete ui;
}

void VbiDialog::updateVbi(VbiDecoder::Vbi vbi, bool isVbiValid)
{
    qDebug() << "VbiDialog::updateVbi(): Called";

    if (!isVbiValid) {
        // VBI data is not valid
        ui->discTypeLabel->setText("No metadata");
        ui->leadInLabel->setText("No metadata");
        ui->leadOutLabel->setText("No metadata");
        ui->userCodeLabel->setText("No metadata");
        ui->pictureNumberLabel->setText("No metadata");
        ui->pictureStopCodeLabel->setText("No metadata");
        ui->chapterNumberLabel->setText("No metadata");
        ui->clvTimeCodeLabel->setText("No metadata");

        ui->cxLabel->setText("No metadata");
        ui->discSizeLabel->setText("No metadata");
        ui->discSideLabel->setText("No metadata");
        ui->teletextLabel->setText("No metadata");
        ui->programmeDumpLabel->setText("No metadata");
        ui->fmFmMultiplexLabel->setText("No metadata");
        ui->digitalLabel->setText("No metadata");
        ui->parityCorrectLabel->setText("No metadata");
        ui->soundModeLabel->setText("No metadata");

        ui->cxLabelAm2->setText("No metadata");
        ui->discSizeLabelAm2->setText("No metadata");
        ui->discSideLabelAm2->setText("No metadata");
        ui->teletextLabelAm2->setText("No metadata");
        ui->copyAllowedLabelAm2->setText("No metadata");
        ui->standardVideoLabelAm2->setText("No metadata");
        ui->soundModeLabelAm2->setText("No metadata");

        return;
    }

    if (vbi.type != VbiDecoder::VbiDiscTypes::unknownDiscType) {
        if (vbi.type == VbiDecoder::VbiDiscTypes::cav) ui->discTypeLabel->setText("CAV");
        if (vbi.type == VbiDecoder::VbiDiscTypes::clv) ui->discTypeLabel->setText("CLV");
    } else {
        if (vbi.type == VbiDecoder::VbiDiscTypes::cav) ui->discTypeLabel->setText("CAV");
        if (vbi.type == VbiDecoder::VbiDiscTypes::clv) ui->discTypeLabel->setText("CLV");
        if (vbi.type == VbiDecoder::VbiDiscTypes::unknownDiscType) ui->discTypeLabel->setText("Unknown");
    }

    if (vbi.leadIn) ui->leadInLabel->setText("Yes");
    else ui->leadInLabel->setText("No");

    if (vbi.leadOut) ui->leadOutLabel->setText("Yes");
    else ui->leadOutLabel->setText("No");

    if (vbi.userCode.isEmpty()) ui->userCodeLabel->setText("None");
    else {
        if (!vbi.userCode.isEmpty()) ui->userCodeLabel->setText(vbi.userCode);
        else ui->userCodeLabel->setText(vbi.userCode);
    }

    if (vbi.picNo != -1) ui->pictureNumberLabel->setText(QString::number(vbi.picNo));
    else ui->pictureNumberLabel->setText("Unknown");

    if (vbi.picStop) ui->pictureStopCodeLabel->setText("Yes");
    else ui->pictureStopCodeLabel->setText("No");

    if (vbi.chNo != -1) ui->chapterNumberLabel->setText(QString::number(vbi.chNo));
    else ui->chapterNumberLabel->setText("Unknown");

    QString clvTimecodeString;
    // Hours
    if (vbi.clvHr != -1) clvTimecodeString = QString("%1").arg(vbi.clvHr, 2, 10, QChar('0')) + ":";
    else clvTimecodeString = "xx:";

    // Minutes
    if (vbi.clvMin != -1) clvTimecodeString += QString("%1").arg(vbi.clvMin, 2, 10, QChar('0')) + ":";
    else clvTimecodeString += "xx:";

    // Seconds
    if (vbi.clvSec != -1) clvTimecodeString += QString("%1").arg(vbi.clvSec, 2, 10, QChar('0')) + ".";
    else clvTimecodeString += "xx.";

    // Picture (frame) number
    if (vbi.clvPicNo != -1) clvTimecodeString += QString("%1").arg(vbi.clvPicNo, 2, 10, QChar('0'));
    else clvTimecodeString += "xx";

    if (clvTimecodeString == "xx:xx:xx.xx") clvTimecodeString = "Unknown";
    ui->clvTimeCodeLabel->setText(clvTimecodeString);

    // Display original programme status
    if (vbi.cx) ui->cxLabel->setText("On");
    else ui->cxLabel->setText("Off");

    if (vbi.size) ui->discSizeLabel->setText("12 inch disc");
    else ui->discSizeLabel->setText("8 inch disc");

    if (vbi.side) ui->discSideLabel->setText("Side 1");
    else ui->discSideLabel->setText("Side 2");

    if (vbi.teletext) ui->teletextLabel->setText("Present on disc");
    else ui->teletextLabel->setText("Not present on disc");

    if (vbi.dump) ui->programmeDumpLabel->setText("Yes");
    else ui->programmeDumpLabel->setText("No");

    if (vbi.fm) ui->fmFmMultiplexLabel->setText("Yes");
    else ui->fmFmMultiplexLabel->setText("No");

    if (vbi.digital) ui->digitalLabel->setText("Yes");
    else ui->digitalLabel->setText("No");

    if (vbi.parity) ui->parityCorrectLabel->setText("Yes");
    else ui->parityCorrectLabel->setText("No");

    if (vbi.soundMode == VbiDecoder::VbiSoundModes::futureUse) {
        ui->soundModeLabel->setText("Future use/unknown");
    } else if (vbi.soundMode != VbiDecoder::VbiSoundModes::futureUse) {
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::stereo) ui->soundModeLabel->setText("Stereo");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::mono) ui->soundModeLabel->setText("Mono");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::audioSubCarriersOff) ui->soundModeLabel->setText("Audio sub-carriers off");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::bilingual) ui->soundModeLabel->setText("Bilingual");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::stereo_stereo) ui->soundModeLabel->setText("Stereo_Stereo");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::stereo_bilingual) ui->soundModeLabel->setText("Stereo_Bilingual");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::crossChannelStereo) ui->soundModeLabel->setText("Cross Channel Stereo");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::bilingual_bilingual) ui->soundModeLabel->setText("Bilingual_Bilingual");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::mono_dump) ui->soundModeLabel->setText("Mono dump");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::stereo_dump) ui->soundModeLabel->setText("Stereo dump");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::bilingual_dump) ui->soundModeLabel->setText("Bilingual dump");
        if (vbi.soundMode == VbiDecoder::VbiSoundModes::futureUse) ui->soundModeLabel->setText("Future use/unknown");
    }

    // Display programme status amendment 2
    if (vbi.cx) ui->cxLabelAm2->setText("On");
    else ui->cxLabelAm2->setText("Off");

    if (vbi.size) ui->discSizeLabelAm2->setText("12 inch disc");
    else ui->discSizeLabelAm2->setText("8 inch disc");

    if (vbi.side) ui->discSideLabelAm2->setText("Side 1");
    else ui->discSideLabelAm2->setText("Side 2");

    if (vbi.teletext) ui->teletextLabelAm2->setText("Present on disc");
    else ui->teletextLabelAm2->setText("Not present on disc");

    if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::futureUse) {
        ui->soundModeLabelAm2->setText("Future use/unknown");
    } else if (vbi.soundModeAm2 != VbiDecoder::VbiSoundModes::futureUse) {
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::stereo) ui->soundModeLabelAm2->setText("Stereo");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::mono) ui->soundModeLabelAm2->setText("Mono");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::audioSubCarriersOff) ui->soundModeLabelAm2->setText("Audio sub-carriers off");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::bilingual) ui->soundModeLabelAm2->setText("Bilingual");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::stereo_stereo) ui->soundModeLabelAm2->setText("Stereo_Stereo");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::stereo_bilingual) ui->soundModeLabelAm2->setText("Stereo_Bilingual");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::crossChannelStereo) ui->soundModeLabelAm2->setText("Cross Channel Stereo");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::bilingual_bilingual) ui->soundModeLabelAm2->setText("Bilingual_Bilingual");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::mono_dump) ui->soundModeLabelAm2->setText("Mono dump");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::stereo_dump) ui->soundModeLabelAm2->setText("Stereo dump");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::bilingual_dump) ui->soundModeLabelAm2->setText("Bilingual dump");
        if (vbi.soundModeAm2 == VbiDecoder::VbiSoundModes::futureUse) ui->soundModeLabelAm2->setText("Future use/unknown");
    }

    if (vbi.copyAm2) ui->copyAllowedLabelAm2->setText("Yes");
    else ui->copyAllowedLabelAm2->setText("No");

    if (vbi.standardAm2) ui->standardVideoLabelAm2->setText("Yes");
    else ui->standardVideoLabelAm2->setText("No");

}
