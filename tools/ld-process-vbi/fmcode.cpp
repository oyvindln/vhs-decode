/************************************************************************

    fmcode.cpp

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

#include "fmcode.h"
#include "vbiutilities.h"

// Public method to read a 40-bit FM coded signal from a field line.
// Return true if decoding was successful, false otherwise.
bool FmCode::decodeLine(const SourceVideo::Data &lineData,
                        const LdDecodeMetaData::VideoParameters& videoParameters,
                        LdDecodeMetaData::Field& fieldMetadata)
{
    // Reset data to invalid
    fieldMetadata.ntsc.isFmCodeDataValid = false;
    fieldMetadata.ntsc.fmCodeData = -1;
    fieldMetadata.ntsc.fieldFlag = false;

    quint64 receiverClockSyncBits = 0;
    quint64 videoFieldIndicator = 0;
    quint64 leadingDataRecognitionBits = 0;
    quint64 dataValue = 0;
    quint64 dataParityBit = 0;
    quint64 trailingDataRecognitionBits = 0;

    quint64 decodedBytes = 0;

    // Determine the 16-bit zero-crossing point
    qint32 zcPoint = (videoParameters.white16bIre + videoParameters.black16bIre) / 2;

    QVector<bool> fmData = getTransitionMap(lineData, zcPoint);

    // Get the number of samples for 0.75us
    double fSamples = (videoParameters.sampleRate / 1000000) * 0.75;
    qint32 samples = static_cast<qint32>(fSamples);

    // Keep track of the number of bits decoded
    qint32 decodeCount = 0;

    // Find the first transition
    qint32 x = videoParameters.activeVideoStart;
    while (x < fmData.size() && fmData[x] == false) {
        x++;
    }

    if (x < fmData.size()) {
        qint32 lastTransitionX = x;
        bool lastState = fmData[x];

        // Find the rest of the bits
        while (x < fmData.size() && decodeCount < 40) {
            // Find the next transition
            while (x < fmData.size() && fmData[x] == lastState)
            {
                x++;
            }

            lastState = fmData[x];

            // Was the transition in the middle of the cell?
            if (x - lastTransitionX < samples) {
                decodedBytes = (decodedBytes << 1) + 1;
                lastTransitionX = x;
                decodeCount++;

                // Find the end of the cell
                while (x < fmData.size() && fmData[x] == lastState)
                {
                    x++;
                }
                if (x >= fmData.size()) break; // Check for overflow
                lastState = fmData[x];
                lastTransitionX = x;
            } else {
                decodedBytes = (decodedBytes << 1);
                lastTransitionX = x;
                decodeCount++;
            }

            // Next x
            x++;
        }
    }

    // We must have 40-bits if the decode was successful
    if (decodeCount != 40) {
        if (decodeCount == 0) qDebug() << "FmCode::fmDecoder(): No FM code data found in the field line";
        else qDebug() << "FmCode::fmDecoder(): FM decode failed!  Only got" << decodeCount << "bits";
        return false;
    }

    // Show the 40-bit FM coded data as hexadecimal
    qDebug() << "FmCode::fmDecoder(): 40-bit FM code is" << QString::number(decodedBytes, 16);

    // Split the result into the required fields
    receiverClockSyncBits = (decodedBytes & 0xF000000000) >> 36;
    videoFieldIndicator = (decodedBytes & 0x0800000000) >> 35;
    leadingDataRecognitionBits = (decodedBytes & 0x07F0000000) >> 28;
    dataValue = (decodedBytes & 0x000FFFFF00) >>  8;
    dataParityBit = (decodedBytes & 0x0000000080) >> 7;
    trailingDataRecognitionBits = (decodedBytes & 0x000000007F);

    qDebug() << "FmCode::fmDecoder(): receiverClockSyncBits =" << receiverClockSyncBits;
    qDebug() << "FmCode::fmDecoder(): videoFieldIndicator =" << videoFieldIndicator;
    qDebug() << "FmCode::fmDecoder(): leadingDataRecognitionBits =" << leadingDataRecognitionBits;
    qDebug() << "FmCode::fmDecoder(): dataValue =" << dataValue;
    qDebug() << "FmCode::fmDecoder(): dataParityBit =" << dataParityBit;
    qDebug() << "FmCode::fmDecoder(): trailingDataRecognitionBits =" << trailingDataRecognitionBits;

    // Sanity check the data
    if (receiverClockSyncBits != 3 || leadingDataRecognitionBits != 114 || trailingDataRecognitionBits != 13) {
        qWarning() << "FM code does not appear sane";
        return false;
    }

    // Check parity (dataParityBit = 1 = odd parity)
    if (dataParityBit == 1 && !isEvenParity(dataValue)) {
        qWarning() << "Data fails parity check (expected even, got odd)";
        return false;
    }

    if (dataParityBit == 0 && isEvenParity(dataValue)) {
        qWarning() << "Data fails parity check (expected odd, got even)";
        return false;
    }

    // Everything looks good -- update the metadata
    fieldMetadata.ntsc.isFmCodeDataValid = true;
    fieldMetadata.ntsc.fmCodeData = static_cast<qint32>(dataValue);
    fieldMetadata.ntsc.fieldFlag = (videoFieldIndicator == 1);

    return true;
}
