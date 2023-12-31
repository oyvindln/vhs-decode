/************************************************************************

    videoid.cpp

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2023 Phillip Blucas

    This file is part of ld-decode-tools.

    ld-process-vbi is free software: you can redistribute it and/or
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

#include "videoid.h"
#include "vbiutilities.h"

/*!
    \class VideoID

    Decoder for VIDEO ID as defined in IEC 61880.  This data on lines 20
    and 283 can contain aspect ratio, CGMS-A, and APS information.

    References:

    [IEC] "Video systems (525/60) - Video and accompanied data using the
    vertical blanking interval - Analogue interface",
    (https://webstore.iec.ch/publication/6057)
    IEC 61880:1998.
*/

// Public method to read IEC 61880 data.
// Return true if data was decoded successfully, false otherwise.
bool VideoID::decodeLine(const SourceVideo::Data& lineData,
                      const LdDecodeMetaData::VideoParameters& videoParameters,
                      LdDecodeMetaData::Field& fieldMetadata)
{
    // Reset data to invalid
    fieldMetadata.ntsc.isVideoIdDataValid = false;
    fieldMetadata.ntsc.videoIdData = -1;

    quint32 message = 0;
    quint32 word0 = 0;
    quint32 word1 = 0;
    quint32 word2 = 0;
    quint32 crcc = 0;

    quint32 codeWord = 0;
    qint32 decodeCount = 0;

    // The zero-crossing point is 35 IRE [IEC p9]
    qint32 zcPoint = ((videoParameters.white16bIre - videoParameters.black16bIre) * 35 / 100 ) + videoParameters.black16bIre;

    // Get the transition map for the line
    QVector<bool> transitionMap = getTransitionMap(lineData, zcPoint);

    // Bit clock is fSC / 8, i.e. 455/16 * fH [IEC p9]
    double samplesPerBit = static_cast<double>(videoParameters.fieldWidth) * 16 / 455;

    // Each line contains start a reference bit, a blank bit, and then a 20-bit
    // codeword that uses a 6-bit CRC. [IEC p9]

    // Find the first transition
    double x = static_cast<double>(videoParameters.colourBurstEnd);
    double xLimit = static_cast<double>(videoParameters.fieldWidth) - 22.0 * samplesPerBit;

    // Find the start bits (10)
    if (!findTransition(transitionMap, true, x, xLimit)) {
        qDebug() << "VideoID::decodeLine(): No reference bit found (1)";
        return false;
    }
    x += samplesPerBit * 1.5;
    if (transitionMap[static_cast<qint32>(x)]){
        qDebug() << "VideoID::decodeLine(): No start code found (10)";
        return false;
    }

    // Get the rest of the bits
    x += samplesPerBit;
    while (x < transitionMap.size() && decodeCount < 20) {
        codeWord = (codeWord << 1) + transitionMap[static_cast<qint32>(x)];
        decodeCount++;
        x += samplesPerBit;
    }

    // Show the 20-bit codeword
    qDebug() << "VideoID::decodeLine(): 20-bit code is" << QStringLiteral("%1").arg(codeWord, 20, 2, QLatin1Char('0'));

    // Split the result into the required fields
    word0 = (codeWord & 0xC0000) >> 18;
    word1 = (codeWord & 0x3C000) >> 14;
    word2 = (codeWord & 0x07F00) >> 6;
    crcc =  codeWord & 0x3F;
    message = codeWord >> 6;

    qDebug() << "VideoID::decodeLine(): word0 =" << QStringLiteral("%1").arg(word0, 2, 2, QLatin1Char('0'));
    qDebug() << "VideoID::decodeLine(): word1 =" << QStringLiteral("%1").arg(word1, 4, 2, QLatin1Char('0'));
    qDebug() << "VideoID::decodeLine(): word2 =" << QStringLiteral("%1").arg(word2, 8, 2, QLatin1Char('0'));
    qDebug() << "VideoID::decodeLine(): crcc  =" << QStringLiteral("%1").arg(crcc, 6, 2, QLatin1Char('0'));

    // Calculate the CRC [IEC p11]
    // x^6 + x + 1, initialized with all-ones
    uint32_t crc = 0b111111;
    for (int i = 0; i < 14; i++)
    {
        int invert = ((message >> i) & 1) ^ ((crc >> 5) & 1);
        crc ^= invert;
        crc <<= 1;
        crc += invert;
    }
    crc &= 0x3F;

    // Quit if the calculated CRC doesn't match
    if (crc != crcc) {
        qDebug() << "VideoID::decodeLine(): Invalid CRC" << QStringLiteral("%1").arg(crc, 6, 2, QLatin1Char('0'));
        return false;
    }

    // Everything looks good -- update the metadata
    fieldMetadata.ntsc.isVideoIdDataValid = true;
    fieldMetadata.ntsc.videoIdData = message;
    return true;
}
