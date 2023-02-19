/************************************************************************

    whiteflag.h

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2021 Simon Inns

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

#include "whiteflag.h"

// Public method to read the white flag status from a field-line.
// Return true if the flag is detected, false otherwise.
bool WhiteFlag::decodeLine(const SourceVideo::Data& lineData,
                           const LdDecodeMetaData::VideoParameters& videoParameters,
                           LdDecodeMetaData::Field& fieldMetadata)
{
    // Determine the 16-bit zero-crossing point
    qint32 zcPoint = (videoParameters.white16bIre + videoParameters.black16bIre) / 2;

    qint32 whiteCount = 0;
    for (qint32 x = videoParameters.activeVideoStart; x < videoParameters.activeVideoEnd; x++) {
        if (lineData[x] > zcPoint) whiteCount++;
    }

    // Mark the line as a white flag if at least 50% of the data is above the zc point
    if (whiteCount > ((videoParameters.activeVideoEnd - videoParameters.activeVideoStart) / 2)) {
        qDebug() << "WhiteFlag::getWhiteFlag(): White-flag detected: White count was" << whiteCount << "out of" <<
                    (videoParameters.activeVideoEnd - videoParameters.activeVideoStart);
        fieldMetadata.ntsc.whiteFlag = true;
        return true;
    }

    // No white flag detected
    fieldMetadata.ntsc.whiteFlag = false;
    return false;
}
