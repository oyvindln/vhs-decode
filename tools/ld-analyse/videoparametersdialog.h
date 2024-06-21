/************************************************************************

    videoparametersdialog.h

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

#ifndef VIDEOPARAMETERSDIALOG_H
#define VIDEOPARAMETERSDIALOG_H

#include <QAbstractButton>
#include <QDialog>

#include "lddecodemetadata.h"

namespace Ui {
class VideoParametersDialog;
}

class VideoParametersDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VideoParametersDialog(QWidget *parent = nullptr);
    ~VideoParametersDialog();

    void setVideoParameters(const LdDecodeMetaData::VideoParameters &videoParameters);

signals:
    void videoParametersChanged(const LdDecodeMetaData::VideoParameters &videoParameters);

public slots:
    void levelSelected(qint32 level);

private slots:
    void on_blackLevelSpinBox_valueChanged(int value);
    void on_whiteLevelSpinBox_valueChanged(int value);
    void on_activeVideoStartSpinBox_valueChanged(int value);
    void on_activeVideoWidthSpinBox_valueChanged(int value);

    void on_blackLevelResetButton_clicked();
    void on_blackLevelAltResetButton_clicked();
    void on_whiteLevelResetButton_clicked();
    void on_activeVideoStartResetButton_clicked();
    void on_activeVideoWidthResetButton_clicked();

    void on_aspectRatioButtonGroup_buttonClicked(QAbstractButton *button);

private:
    Ui::VideoParametersDialog *ui;
    LdDecodeMetaData::VideoParameters videoParameters;
    qint32 originalActiveVideoStart = -1;
    qint32 originalActiveVideoWidth = -1;

    void updateDialog();
};

#endif // VIDEOPARAMETERSDIALOG_H
