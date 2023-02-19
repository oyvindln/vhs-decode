/************************************************************************

    closedcaption.cpp

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2023 Adam Sampson

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

#include "closedcaption.h"
#include "vbiutilities.h"

/*!
    \class ClosedCaption

    Decoder for EIA/CEA-608 data lines, widely used for closed
    captioning in NTSC, and occasionally in other standards.

    References:

    [CTA] "Line 21 Data Services", (https://shop.cta.tech/products/line-21-data-services)
    ANSI/CTA-608-E S-2019, April 2008.
*/

// Public method to read CEA-608 Closed Captioning data.
// Return true if CC data was decoded successfully, false otherwise.
bool ClosedCaption::decodeLine(const SourceVideo::Data& lineData,
                               const LdDecodeMetaData::VideoParameters& videoParameters,
                               LdDecodeMetaData::Field& fieldMetadata)
{
    // Reset data to invalid
    fieldMetadata.closedCaption.inUse = false;
    fieldMetadata.closedCaption.data0 = -1;
    fieldMetadata.closedCaption.data1 = -1;

    // The zero-crossing point is 25 IRE [CTA p13]
    qint32 zcPoint = ((videoParameters.white16bIre - videoParameters.black16bIre) / 4) + videoParameters.black16bIre;

    // Get the transition map for the line
    QVector<bool> transitionMap = getTransitionMap(lineData, zcPoint);

    // Bit clock is 32 x fH [CTA p14, note 1]
    double samplesPerBit = static_cast<double>(videoParameters.fieldWidth) / 32.0;

    // Following the colourburst, the line starts with 2-7 (but usually 7)
    // cycles of sine wave at the bit clock rate, then start bits 001, then 16
    // bits of data. [CTA p14] ("21.4 D" in the standard is a typo; it should
    // be "2.14 D" from the time given.)

    // Find the 00 by looking for a 1.5-bit low period
    double x = static_cast<double>(videoParameters.colourBurstEnd) + (2.0 * samplesPerBit);
    double xLimit = static_cast<double>(videoParameters.fieldWidth) - (17.0 * samplesPerBit);
    double lastOne = x;
    while ((x - lastOne) < (1.5 * samplesPerBit)) {
        if (x >= xLimit) {
            qDebug() << "ClosedCaption::decodeLine(): No start bits found (00)";
            return false;
        }
        if (transitionMap[static_cast<qint32>(x)] == true) lastOne = x;
        x += 1.0;
    }

    // Resynchronise on the 1 transition
    if (!findTransition(transitionMap, true, x, xLimit)) {
        qDebug() << "ClosedCaption::decodeLine(): No start bits found (1)";
        return false;
    }

    qDebug() << "ClosedCaption::decodeLine(): Found start bit transition at" << x;

    // Skip the the start bit and move to the centre of the first payload bit
    x += 1.5 * samplesPerBit;

    // Get the first 7 bit code
    uchar byte0 = 0;
    for (qint32 i = 0; i < 7; i++)
    {
        byte0 >>= 1;
        if (transitionMap[static_cast<qint32>(x)]) byte0 += 64;
        x += samplesPerBit;
    }

    // Get the first 7 bit parity
    uchar byte0Parity = 0;
    if (transitionMap[static_cast<qint32>(x)]) byte0Parity = 1;
    x += samplesPerBit;

    // Get the second byte
    uchar byte1 = 0;
    for (qint32 i = 0; i < 7; i++)
    {
        byte1 >>= 1;
        if (transitionMap[static_cast<qint32>(x)]) byte1 += 64;
        x += samplesPerBit;
    }

    // Get the second 7 bit parity
    uchar byte1Parity = 0;
    if (transitionMap[static_cast<qint32>(x)]) byte1Parity = 1;
    x += samplesPerBit;

    qDebug().nospace() << "ClosedCaption::decodeLine(): Bytes are: " << byte0 << " (" << byte0Parity << ") - "
                       << byte1 << " (" << byte1Parity << ")";

    if (isEvenParity(byte0) && byte0Parity != 1) {
        qDebug() << "ClosedCaption::decodeLine(): First byte failed parity check!";
    } else {
        fieldMetadata.closedCaption.data0 = byte0;
        fieldMetadata.closedCaption.inUse = true;
    }

    if (isEvenParity(byte1) && byte1Parity != 1) {
        qDebug() << "ClosedCaption::decodeLine(): Second byte failed parity check!";
    } else {
        fieldMetadata.closedCaption.data1 = byte1;
        fieldMetadata.closedCaption.inUse = true;
    }

    return true;
}
