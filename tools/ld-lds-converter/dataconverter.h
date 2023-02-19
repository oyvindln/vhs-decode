/************************************************************************

    dataconverter.h

    ld-lds-converter - 10-bit to 16-bit .lds converter for ld-decode
    Copyright (C) 2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-lds-converter is free software: you can redistribute it and/or
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

#ifndef DATACONVERTER_H
#define DATACONVERTER_H

#include <QObject>
#include <QDebug>
#include <QFile>

class DataConverter : public QObject
{
    Q_OBJECT
public:
    explicit DataConverter(QString inputFileNameParam, QString outputFileNameParam, bool isPackingParam, bool isRIFFParam, QObject *parent = nullptr);
    bool process(void);

signals:

public slots:

private:
    QString inputFileName;
    QString outputFileName;
    bool isPacking;
    bool isRIFF;

    QFile *inputFileHandle;
    QFile *outputFileHandle;

    // Private methods
    bool openInputFile(void);
    void closeInputFile(void);
    bool openOutputFile(void);
    void closeOutputFile(void);
    void packFile(void);
    void unpackFile(void);
};

#endif // DATACONVERTER_H
