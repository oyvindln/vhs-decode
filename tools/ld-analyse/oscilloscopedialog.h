/************************************************************************

    oscilloscopedialog.cpp

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

#ifndef OSCILLOSCOPEDIALOG_H
#define OSCILLOSCOPEDIALOG_H

#include <QDialog>
#include <QGraphicsPixmapItem>
#include <QPainter>
#include <QDebug>
#include <QMouseEvent>

#include "sourcevideo.h"
#include "lddecodemetadata.h"
#include "tbcsource.h"

namespace Ui {
class OscilloscopeDialog;
}

class OscilloscopeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit OscilloscopeDialog(QWidget *parent = nullptr);
    ~OscilloscopeDialog();

    void showTraceImage(TbcSource::ScanLineData scanLineData, qint32 xCoord, qint32 yCoord, qint32 frameWidth, qint32 frameHeight);

signals:
    void scopeCoordsChanged(qint32 xCoord, qint32 yCoord);
    void scopeLevelSelect(qint32 value);

private slots:
    void on_previousPushButton_clicked();
    void on_nextPushButton_clicked();
    void on_xCoordSpinBox_valueChanged(int arg1);
    void on_yCoordSpinBox_valueChanged(int arg1);
    void on_YCcheckBox_clicked();
    void on_YcheckBox_clicked();
    void on_CcheckBox_clicked();
    void on_dropoutsCheckBox_clicked();

    void mousePressEvent(QMouseEvent *event);
    void mouseMoveEvent(QMouseEvent *event);

private:
    Ui::OscilloscopeDialog *ui;
    qint32 maximumX;
    qint32 maximumY;
    qint32 scopeWidth;
    qint32 lastScopeX;
    qint32 lastScopeY;

    QImage getFieldLineTraceImage(TbcSource::ScanLineData scanLineData, qint32 pictureDot);
    void mouseLevelSelect(qint32 oY);
    void mousePictureDotSelect(qint32 oX);
};

#endif // OSCILLOSCOPEDIALOG_H
