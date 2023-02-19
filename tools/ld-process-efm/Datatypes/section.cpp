/************************************************************************

    section.cpp

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

#include "section.h"

// Note: Class for storing 'sections' as defined by clause 18 of ECMA-130

Section::Section()
{
    qMode = -1;

    // Default the Q Metadata
    qMetadata.qControl.isAudioNotData = false;
    qMetadata.qControl.isStereoNotQuad = false;
    qMetadata.qControl.isNoPreempNotPreemp = false;
    qMetadata.qControl.isCopyProtectedNotUnprotected = false;
    qMetadata.qMode1And4.x = 0;
    qMetadata.qMode1And4.point = 0;
    qMetadata.qMode1And4.discTime = TrackTime();
    qMetadata.qMode1And4.trackTime = TrackTime();
    qMetadata.qMode1And4.isLeadIn = false;
    qMetadata.qMode1And4.isLeadOut = false;
    qMetadata.qMode1And4.trackNumber = 0;
    qMetadata.qMode1And4.isEncoderRunning = true;
}

bool Section::setData(const uchar *dataIn)
{
    // Interpret the section data
    qint32 symbolNumber = 2;
    for (qint32 byteC = 0; byteC < 12; byteC++) {
        // Initialise the channel bytes
        pSubcode[byteC] = 0;
        qSubcode[byteC] = 0;
        rSubcode[byteC] = 0;
        sSubcode[byteC] = 0;
        tSubcode[byteC] = 0;
        uSubcode[byteC] = 0;
        vSubcode[byteC] = 0;
        wSubcode[byteC] = 0;

        // Copy in the channel data from the section data
        for (qint32 bitC = 7; bitC >= 0; bitC--) {
            qint32 subcode = static_cast<qint32>(dataIn[symbolNumber]);

            if (subcode & 0x80) pSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x40) qSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x20) rSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x10) sSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x08) tSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x04) uSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x02) vSubcode[byteC] |= (1 << bitC);
            if (subcode & 0x01) wSubcode[byteC] |= (1 << bitC);
            symbolNumber++;
        }
    }

    // The Q channel specifies how the blocks frame data should be used
    // so we decode that here

    // Firstly we CRC the Q channel to ensure it contains valid data
    if (verifyQ()) {
        // Decode the Q channel mode
        qMode = decodeQAddress();

        // Decode the Q control
        decodeQControl();

        // If mode 0 (Custom DATA-Q), flag as unsupported
        if (qMode == 0) qDebug() << "Section::setData(): Unsupported Q Mode 0 (Custom DATA-Q)";

        // If mode 1 (CD), decode the metadata
        if (qMode == 1) decodeQDataMode1And4();

        // If mode 2 (Catalogue number), decode the metadata
        if (qMode == 2) decodeQDataMode2();

        // Mode 3 (Track ID), is currently unsupported
        if (qMode == 3) qDebug() << "Section::setData(): Unsupported Q Mode 3 (track ID)";

        // If mode 4 (LD), decode the metadata
        if (qMode == 4) decodeQDataMode1And4();

        // If mode is unsupported, flag in debug
        if (qMode < 0 || qMode > 4) qDebug() << "Section::setData(): Unsupported Q Mode" << qMode;
    } else {
        // Q channel mode is invalid
        qMode = -1;
        return false;
    }

    return true;
}

// Method to determine the Q mode
qint32 Section::getQMode() const
{
    return qMode;
}

// Method to get Q channel metadata
const Section::QMetadata &Section::getQMetadata() const
{
    return qMetadata;
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to CRC verify the Q subcode channel
bool Section::verifyQ()
{
    // CRC check the Q-subcode - CRC is on control+mode+data 4+4+72 = 80 bits with 16-bit CRC (96 bits total)
    quint16 crcChecksum = static_cast<quint16>(~((qSubcode[10] << 8) + qSubcode[11])); // Inverted on disc
    quint16 calcChecksum = crc16(qSubcode, 10);

    // Is the Q subcode valid?
    if (crcChecksum != calcChecksum) {
        //qDebug() << "SubcodeBlock::verifyQ(): Q Subcode CRC failed - Q subcode payload is invalid";
        return false;
    }

    return true;
}

// Method to perform CRC16 (XMODEM)
// Adapted from http://mdfs.net/Info/Comp/Comms/CRC16.htm
quint16 Section::crc16(const uchar *addr, quint16 num)
{
    qint32 i;
    quint32 crc = 0;

    for (; num > 0; num--) {
        crc = crc ^ static_cast<quint32>(*addr++ << 8);
        for (i = 0; i < 8; i++) {
            crc = crc << 1;
            if (crc & 0x10000) crc = (crc ^ 0x1021) & 0xFFFF;
        }
    }

    return static_cast<quint16>(crc);
}

// Method to decode the Q subcode ADR field
qint32 Section::decodeQAddress()
{
    // Get the Q Mode value
    qint32 qMode = (qSubcode[0] & 0x0F);

    // Range check
    if (qMode < 0 || qMode > 4) qMode = -1;

    return qMode;
}

// Method to decode the Q subcode CONTROL field
void Section::decodeQControl()
{
    // Get the control payload
    qint32 qControlField = (qSubcode[0] & 0xF0) >> 4;

    // Control field values can be:
    //
    // x000 = 2-Channel/4-Channel
    // 0x00 = audio/data
    // 00x0 = Copy not permitted/copy permitted
    // 000x = pre-emphasis off/pre-emphasis on

    if ((qControlField & 0x08) == 0x08) qMetadata.qControl.isStereoNotQuad = false;
    else qMetadata.qControl.isStereoNotQuad = true;

    if ((qControlField & 0x04) == 0x04) qMetadata.qControl.isAudioNotData = false;
    else qMetadata.qControl.isAudioNotData = true;

    if ((qControlField & 0x02) == 0x02) qMetadata.qControl.isCopyProtectedNotUnprotected = false;
    else qMetadata.qControl.isCopyProtectedNotUnprotected = true;

    if ((qControlField & 0x01) == 0x01) qMetadata.qControl.isNoPreempNotPreemp = false;
    else qMetadata.qControl.isNoPreempNotPreemp = true;
}

// Method to decode Q subcode Mode 1 and Mode 4 DATA-Q
void Section::decodeQDataMode1And4()
{
    // Get the track number (TNO) field
    qint32 tno = bcdToInteger(qSubcode[1]);

    // Use TNO to detect lead-in, audio or lead-out
    if (qSubcode[1] == 0xAA) {
        // Lead out
        qMetadata.qMode1And4.isLeadOut = true;
        qMetadata.qMode1And4.isLeadIn = false;
        qMetadata.qMode1And4.trackNumber = bcdToInteger(qSubcode[1]);
        qMetadata.qMode1And4.x = bcdToInteger(qSubcode[2]);
        qMetadata.qMode1And4.point = -1;
        qMetadata.qMode1And4.trackTime = TrackTime(bcdToInteger(qSubcode[3]), bcdToInteger(qSubcode[4]), bcdToInteger(qSubcode[5]));
        qMetadata.qMode1And4.discTime = TrackTime(bcdToInteger(qSubcode[7]), bcdToInteger(qSubcode[8]), bcdToInteger(qSubcode[9]));

    } else if (tno == 0) {
        // Lead in
        qMetadata.qMode1And4.isLeadOut = false;
        qMetadata.qMode1And4.isLeadIn = true;
        qMetadata.qMode1And4.trackNumber = bcdToInteger(qSubcode[1]);
        qMetadata.qMode1And4.x = -1;
        qMetadata.qMode1And4.point = bcdToInteger(qSubcode[2]);
        qMetadata.qMode1And4.trackTime = TrackTime(bcdToInteger(qSubcode[3]), bcdToInteger(qSubcode[4]), bcdToInteger(qSubcode[5]));
        qMetadata.qMode1And4.discTime = TrackTime(bcdToInteger(qSubcode[7]), bcdToInteger(qSubcode[8]), bcdToInteger(qSubcode[9]));
    } else {
        // Audio
        qMetadata.qMode1And4.isLeadOut = false;
        qMetadata.qMode1And4.isLeadIn = false;
        qMetadata.qMode1And4.trackNumber = bcdToInteger(qSubcode[1]);
        qMetadata.qMode1And4.x = bcdToInteger(qSubcode[2]);
        qMetadata.qMode1And4.point = -1;
        qMetadata.qMode1And4.trackTime = TrackTime(bcdToInteger(qSubcode[3]), bcdToInteger(qSubcode[4]), bcdToInteger(qSubcode[5]));
        qMetadata.qMode1And4.discTime = TrackTime(bcdToInteger(qSubcode[7]), bcdToInteger(qSubcode[8]), bcdToInteger(qSubcode[9]));
    }

    // Determine if the encoder is running or not
    // (logic is translated from the specification)
    if (qMetadata.qMode1And4.isLeadIn) {
        // Q Mode 1/4 - Lead in section
        qMetadata.qMode1And4.isEncoderRunning = false;
    } else if (qMetadata.qMode1And4.isLeadOut) {
        // Q Mode 1/4 - Lead out section
        if (qMetadata.qMode1And4.x == 0) {
            // Encoding paused
            qMetadata.qMode1And4.isEncoderRunning = false;
        } else {
            // Encoding running
            qMetadata.qMode1And4.isEncoderRunning = true;
        }
    } else {
        // Q Mode 1/4 - Audio section
        if (qMetadata.qMode1And4.x == 0) {
            // Encoding paused
            qMetadata.qMode1And4.isEncoderRunning = false;
        } else {
            // Encoding running
            qMetadata.qMode1And4.isEncoderRunning = true;
        }
    }
}

// Method to decode Q subcode Mode 2 DATA-Q
void Section::decodeQDataMode2()
{
    // Get the 13 digit catalogue number
    QString catalogueNumber;
    catalogueNumber  = QString("%1").arg(bcdToInteger(qSubcode[1]), 2, 10, QChar('0')); // n1 and n2
    catalogueNumber += QString("%1").arg(bcdToInteger(qSubcode[2]), 2, 10, QChar('0')); // n3 and n4
    catalogueNumber += QString("%1").arg(bcdToInteger(qSubcode[3]), 2, 10, QChar('0')); // n5 and n6
    catalogueNumber += QString("%1").arg(bcdToInteger(qSubcode[4]), 2, 10, QChar('0')); // n7 and n8
    catalogueNumber += QString("%1").arg(bcdToInteger(qSubcode[5]), 2, 10, QChar('0')); // n9 and n10
    catalogueNumber += QString("%1").arg(bcdToInteger(qSubcode[6]), 2, 10, QChar('0')); // n11 and n12
    catalogueNumber += QString("%1").arg(bcdToInteger(qSubcode[7]), 2, 10, QChar('0')); // n13 and n14 (n14 is always 0)
    qMetadata.qMode2.catalogueNumber = catalogueNumber.left(13);

    // Get the AFRAME number
    qMetadata.qMode2.aFrame = bcdToInteger(qSubcode[9]);
}

// Method to convert 2 digit BCD byte to an integer
qint32 Section::bcdToInteger(uchar bcd)
{
   return (((bcd>>4)*10) + (bcd & 0xF));
}
