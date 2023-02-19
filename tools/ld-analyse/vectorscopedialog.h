/************************************************************************

    vectorscopedialog.cpp

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

#ifndef VECTORSCOPEDIALOG_H
#define VECTORSCOPEDIALOG_H

#include <QAbstractButton>
#include <QGraphicsPixmapItem>
#include <QDialog>

#include "componentframe.h"
#include "lddecodemetadata.h"

namespace Ui {
class VectorscopeDialog;
}

class VectorscopeDialog : public QDialog
{
    Q_OBJECT

public:
    explicit VectorscopeDialog(QWidget *parent = nullptr);
    ~VectorscopeDialog();

    void showTraceImage(const ComponentFrame &componentFrame, const LdDecodeMetaData::VideoParameters &videoParameters);

signals:
    void scopeChanged();

private slots:
    void on_defocusCheckBox_clicked();
    void on_graticuleButtonGroup_buttonClicked(QAbstractButton *button);

private:
    Ui::VectorscopeDialog *ui;

    QImage getTraceImage(const ComponentFrame &componentFrame, const LdDecodeMetaData::VideoParameters &videoParameters);
};

#endif // VECTORSCOPEDIALOG_H
