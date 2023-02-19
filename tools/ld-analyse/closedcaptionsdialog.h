/************************************************************************

    closedcaptionsdialog.h

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

#ifndef CLOSEDCAPTIONSDIALOG_H
#define CLOSEDCAPTIONSDIALOG_H

#include <QDialog>
#include <QDebug>

namespace Ui {
class ClosedCaptionsDialog;
}

class ClosedCaptionsDialog : public QDialog
{
    Q_OBJECT

public:
    explicit ClosedCaptionsDialog(QWidget *parent = nullptr);
    ~ClosedCaptionsDialog();

    void addData(qint32 frameNumber, qint32 data0, qint32 data1);

private:
    Ui::ClosedCaptionsDialog *ui;
    bool waitingForPreamble;
    qint32 lastFrameNumber;

    qint32 lastNonDisplayCommand;
    qint32 lastDisplayCommand;

    void processCommand(qint32 data0, qint32 data1);
    void resetCaptions();
};

#endif // CLOSEDCAPTIONSDIALOG_H
