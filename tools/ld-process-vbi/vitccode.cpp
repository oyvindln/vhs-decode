/************************************************************************

    vitccode.cpp

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2022 Adam Sampson

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

#include "vitccode.h"
#include "vbiutilities.h"

#include <algorithm>
#include <array>

/*!
    \class VitcCode

    Decoder for vertical interval timecode (VITC) lines.

    References:

    [ITU] "Time and control code standards, for production applications in
    order to facilitate the international exchange of television programmes on
    magnetic tapes", (https://www.itu.int/rec/R-REC-BR.780/en) Rec. ITU-R BR.780-2.

    [SMPTE] "Standard for Television - Time and Control Code"
    (https://ieeexplore.ieee.org/document/7289820 - not open-access), ST 12-1:2008.
*/

// Read a VITC signal from a scanline.
// Return true if a signal was found and successfully decoded, false otherwise.
bool VitcCode::decodeLine(const SourceVideo::Data &lineData,
                          const LdDecodeMetaData::VideoParameters& videoParameters,
                          LdDecodeMetaData::Field& fieldMetadata)
{
    // Reset data to invalid
    fieldMetadata.vitc.inUse = false;
    std::fill(fieldMetadata.vitc.vitcData.begin(), fieldMetadata.vitc.vitcData.end(), 0);

    // Convert line data to binary values.
    // For NTSC, 40 IRE is halfway between the 0 and 1 limits; PAL is very close to this. [ITU 6.18.1]
    const qint32 zcPoint = videoParameters.black16bIre
                           + ((40 * (videoParameters.white16bIre - videoParameters.black16bIre)) / 100);
    QVector<bool> dataBits = getTransitionMap(lineData, zcPoint);

    // Number of samples per bit [ITU 6.18]
    const double bitSamples = videoParameters.fieldWidth / 115.0;

    // VITC encodes 8 x 8-bit bytes of real data, plus an 8-bit CRC. Each byte
    // is preceded with 10 for synchronisation, making 90 bits overall. [ITU 6.15]
    std::array<qint32, 9> vitcData;
    std::fill(vitcData.begin(), vitcData.end(), 0);

    // The CRC is computed over the raw data including the synchronisation
    // bits, and has the generator x^8 + 1. [ITU 6.16.6] This is equivalent to
    // just breaking the raw data into bytes and XORing them together.
    std::array<qint32, 12> crcData;
    std::fill(crcData.begin(), crcData.end(), 0);

    // Find the leading edge of the first byte. As per [ITU 6.19], there should
    // be (625/525-line) 11.2/10.0 usec between the leading edge of the sync
    // pulse and the leading edge of the first byte, and 1.9/2.1 usec between
    // the trailing edge and the next sync pulse, but in practice signals that
    // don't meet these specs are common. So start searching from the end of
    // the colourburst, and just make sure there's space for 90 bits before the
    // next sync pulse.
    double byteStart = videoParameters.colourBurstEnd;
    double byteStartLimit = static_cast<double>(lineData.size()) - (90 * bitSamples);
    if (!findTransition(dataBits, false, byteStart, byteStartLimit)) {
        qDebug() << "VitcCode::decodeLine(): No leading zero found";
        return false;
    }
    if (!findTransition(dataBits, true, byteStart, byteStartLimit)) {
        qDebug() << "VitcCode::decodeLine(): No leading edge found";
        return false;
    }

    // Sample each of the 9 bytes
    qint32 bitCount = 0;
    for (qint32 byteNum = 0; byteNum < 9; byteNum++) {
        // Resynchronise by finding the 1-0 transition in the synchronisation sequence
        byteStart += bitSamples * 0.5;
        byteStartLimit += 10 * bitSamples;
        if (!findTransition(dataBits, false, byteStart, byteStartLimit)) {
            qDebug() << "VitcCode::decodeLine(): No transition found for byte" << byteNum;
            return false;
        }
        byteStart -= bitSamples;

        // Extract 10 bits by sampling the centre of each bit, LSB first
        for (qint32 i = 0; i < 10; i++) {
            const qint32 bit = dataBits[static_cast<qint32>(byteStart + ((i + 0.5) * bitSamples))] ? 1 : 0;
            vitcData[byteNum] |= bit << i;

            // Accumulate bits for the CRC as well
            crcData[(bitCount / 8)] |= bit << (bitCount % 8);
            bitCount++;
        }

        // Check for, and remove, the synchronisation sequence
        if ((vitcData[byteNum] & 3) != 1) {
            qDebug() << "VitcCode::decodeLine(): No synchronisation sequence found for byte" << byteNum;
            return false;
        }
        vitcData[byteNum] >>= 2;

        // Advance to the next byte
        byteStart += 10.0 * bitSamples;
    }

    // Check the CRC is valid
    qint32 crcTotal = 0;
    for (qint32 crcValue: crcData) crcTotal ^= crcValue;
    if (crcTotal != 0) {
        qDebug() << "VitcCode::decodeLine(): Invalid CRC" << crcTotal;
        return false;
    }

    // Everything looks good -- update the metadata
    fieldMetadata.vitc.inUse = true;
    std::copy(vitcData.begin(), vitcData.begin() + 8, fieldMetadata.vitc.vitcData.begin());
    qDebug() << "VitcCode::decodeLine(): Found VITC";

    return true;
}

// Return the 1-based frame line numbers that are likely to contain VITC signals
std::vector<qint32> VitcCode::getLineNumbers(const LdDecodeMetaData::VideoParameters& videoParameters)
{
    // VITC can be on any line between 10-20 (525-line) or 6-22 (625-line), but
    // the standards [ITU 6.20, SMPTE 10.6] recommend lines to use. Try the
    // recommended lines first (prioritising those that don't clash with
    // LaserDisc VBI), then the others.
    if (videoParameters.system == PAL) {
        // 625-line
        return {
            21, 19, 18, 20,
            6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 22,
        };
    } else {
        // 525-line
        return {
            14, 12, 16, 18,
            10, 11, 13, 15, 17, 19, 20,
        };
    }
}
