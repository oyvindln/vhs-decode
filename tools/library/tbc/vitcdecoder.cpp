/************************************************************************

    vitcdecoder.cpp

    ld-decode-tools TBC library
    Copyright (C) 2022 Adam Sampson

    This file is part of ld-decode-tools.

    ld-decode-tools is free software: you can redistribute it and/or
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

#include "vitcdecoder.h"

// Decode raw VITC data, for a given system, into a Vitc struct.
// Sets isValid to indicate whether the data seems reasonable.
VitcDecoder::Vitc VitcDecoder::decode(const std::array<qint32, 8>& vitcData, VideoSystem system)
{
    Vitc vitc;
    vitc.isValid = true;

    // Some bit assignments differ between 25-frame and 30-frame systems
    const bool is30Frame = (system != PAL);

    // Decode time
    decodeBCD(vitcData[7] & 0x03, vitcData[6] & 0x0F, vitc.hour, vitc.isValid);
    if (vitc.hour > 23) vitc.isValid = false;
    decodeBCD(vitcData[5] & 0x07, vitcData[4] & 0x0F, vitc.minute, vitc.isValid);
    if (vitc.minute > 59) vitc.isValid = false;
    decodeBCD(vitcData[3] & 0x07, vitcData[2] & 0x0F, vitc.second, vitc.isValid);
    if (vitc.second > 59) vitc.isValid = false;
    decodeBCD(vitcData[1] & 0x03, vitcData[0] & 0x0F, vitc.frame, vitc.isValid);
    if (vitc.frame > (is30Frame ? 29 : 24)) vitc.isValid = false;

    // Decode flags
    if (is30Frame) {
        vitc.isDropFrame = ((vitcData[1] & 0x04) != 0);
        vitc.isColFrame = ((vitcData[1] & 0x08) != 0);
        vitc.isFieldMark = ((vitcData[3] & 0x08) != 0);
        vitc.binaryGroupFlags = (((vitcData[5] & 0x08) != 0) ? 1 : 0)
                                | (((vitcData[7] & 0x04) != 0) ? 2 : 0)
                                | (((vitcData[7] & 0x08) != 0) ? 4 : 0);
    } else {
        vitc.isColFrame = ((vitcData[1] & 0x08) != 0);
        vitc.isFieldMark = ((vitcData[7] & 0x08) != 0);
        vitc.binaryGroupFlags = (((vitcData[3] & 0x08) != 0) ? 1 : 0)
                                | (((vitcData[7] & 0x04) != 0) ? 2 : 0)
                                | (((vitcData[5] & 0x08) != 0) ? 4 : 0);
    }

    // Decode binary groups (without caring about their meaning, for now)
    for (int i = 0; i < 8; i++) {
        vitc.binaryGroups[i] = (vitcData[i] >> 4) & 0x0F;
    }

    return vitc;
}

// Decode a two-digit BCD number, setting isValid to false if it's not reasonable
void VitcDecoder::decodeBCD(qint32 tens, qint32 units, qint32& output, bool& isValid)
{
    if (tens > 9) {
        isValid = false;
        tens = 9;
    }
    if (units > 9) {
        isValid = false;
        units = 9;
    }
    output = (tens * 10) + units;
}
