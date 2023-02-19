/************************************************************************

    biphasecode.cpp

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns

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

#include "biphasecode.h"
#include "vbiutilities.h"

// Decode the three biphase code lines, writing the result into fieldMetadata.
// Return true if any line was decoded successfully, false if none were.
bool BiphaseCode::decodeLines(const SourceVideo::Data& line16Data, const SourceVideo::Data& line17Data,
                              const SourceVideo::Data& line18Data,
                              const LdDecodeMetaData::VideoParameters& videoParameters,
                              LdDecodeMetaData::Field& fieldMetadata)
{
    // Decode all three lines
    bool success = false;
    success |= decodeLine(0, line16Data, videoParameters, fieldMetadata);
    success |= decodeLine(1, line17Data, videoParameters, fieldMetadata);
    success |= decodeLine(2, line18Data, videoParameters, fieldMetadata);
    if (!success) {
        qDebug() << "VbiLineDecoder::process(): No biphase VBI present";
    }

    // Mark VBI as present if any was decoded successfully
    fieldMetadata.vbi.inUse = success;

    return success;
}

// Decode one of the three biphase code lines, writing the result into fieldMetadata.
// Return true if decoding was successful, false otherwise.
bool BiphaseCode::decodeLine(qint32 lineIndex, const SourceVideo::Data& lineData,
                                const LdDecodeMetaData::VideoParameters& videoParameters,
                                LdDecodeMetaData::Field& fieldMetadata)
{
    // Determine the 16-bit zero-crossing point
    qint32 zcPoint = (videoParameters.white16bIre + videoParameters.black16bIre) / 2;

    fieldMetadata.vbi.vbiData[lineIndex] = manchesterDecoder(lineData, zcPoint, videoParameters);

    return (fieldMetadata.vbi.vbiData[lineIndex] != 0);
}

// Private method to read a 24-bit biphase coded signal (manchester code) from a field line
qint32 BiphaseCode::manchesterDecoder(const SourceVideo::Data &lineData, qint32 zcPoint,
                                         LdDecodeMetaData::VideoParameters videoParameters)
{
    qint32 result = 0;
    QVector<bool> manchesterData = getTransitionMap(lineData, zcPoint);

    // Get the number of samples for 1.5us
    double fJumpSamples = (videoParameters.sampleRate / 1000000) * 1.5;
    qint32 jumpSamples = static_cast<qint32>(fJumpSamples);

    // Keep track of the number of bits decoded
    qint32 decodeCount = 0;

    // Find the first transition
    qint32 x = videoParameters.activeVideoStart;
    while (x < manchesterData.size() && manchesterData[x] == false) {
        x++;
    }

    if (x < manchesterData.size()) {
        // Plot the first transition (which is always 01)
        result++;
        decodeCount++;

        // Find the rest of the transitions based on the expected clock rate of 2us per cell window
        while (x < manchesterData.size()) {
            x += jumpSamples;

            // Ensure we don't go out of bounds
            if (x >= manchesterData.size()) break;

            bool startState = manchesterData[x];
            while (x < manchesterData.size() && manchesterData[x] == startState)
            {
                x++;
            }

            if (x < manchesterData.size()) {
                if (manchesterData[x - 1] == false && manchesterData[x] == true) {
                    // 01 transition
                    result = (result << 1) + 1;
                }
                if (manchesterData[x - 1] == true && manchesterData[x] == false) {
                    // 10 transition
                    result = result << 1;
                }
                decodeCount++;
            }
        }
    }

    // We must have 24-bits if the decode was successful
    if (decodeCount != 24) {
        if (decodeCount != 0) qDebug() << "BiphaseCode::manchesterDecoder(): Manchester decode failed!  Got" << decodeCount << "bits, expected 24";
        result = 0;
    }

    return result;
}
