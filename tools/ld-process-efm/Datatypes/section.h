/************************************************************************

    section.h

    ld-process-efm - EFM data decoder
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-efm is free software: you can redistribute it and/or
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

#ifndef SECTION_H
#define SECTION_H

#include <QCoreApplication>
#include <QDebug>

#include "Datatypes/tracktime.h"

class Section
{
public:
    Section();

    // Structure of the Q Control flags
    struct QControl {
        bool isStereoNotQuad;
        bool isAudioNotData;
        bool isCopyProtectedNotUnprotected;
        bool isNoPreempNotPreemp;
    };

    // Structure of the Q mode 1 and 4 metadata
    struct QMode1And4 {
        bool isLeadIn;
        bool isLeadOut;
        qint32 trackNumber;
        qint32 x;
        qint32 point;
        TrackTime trackTime;
        TrackTime discTime;
        bool isEncoderRunning;
    };

    // Structure of the Q mode 2 metadata
    struct QMode2 {
        QString catalogueNumber;
        qint32 aFrame;
    };

    struct QMetadata {
        QControl qControl;
        QMode1And4 qMode1And4;
        QMode2 qMode2;
    };

    bool setData(const uchar *dataIn);
    qint32 getQMode() const;
    const QMetadata &getQMetadata() const;

private:
    // Q channel specific data
    QMetadata qMetadata;
    qint32 qMode;

    // Subcode channels
    uchar pSubcode[12];
    uchar qSubcode[12];
    uchar rSubcode[12];
    uchar sSubcode[12];
    uchar tSubcode[12];
    uchar uSubcode[12];
    uchar vSubcode[12];
    uchar wSubcode[12];

    bool verifyQ();
    static quint16 crc16(const uchar *addr, quint16 num);
    qint32 decodeQAddress();
    void decodeQControl();
    void decodeQDataMode1And4();
    void decodeQDataMode2();
    static qint32 bcdToInteger(uchar bcd);
};

#endif // SECTION_H
