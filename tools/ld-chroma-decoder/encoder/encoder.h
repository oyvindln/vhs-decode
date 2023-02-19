/************************************************************************

    encoder.h

    ld-chroma-encoder - Composite video encoder
    Copyright (C) 2019-2022 Adam Sampson
    Copyright (C) 2022 Phillip Blucas

    This file is part of ld-decode-tools.

    ld-chroma-encoder is free software: you can redistribute it and/or
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

#ifndef ENCODER_H
#define ENCODER_H

#include <QByteArray>
#include <QFile>
#include <cmath>
#include <vector>

#include "lddecodemetadata.h"

// Limits, zero points and scaling factors (from 0-1) for Y'CbCr colour representations
// [Poynton ch25 p305] [BT.601-7 sec 2.5.3]
static constexpr double ONE_MINUS_Kb = 1.0 - 0.114;
static constexpr double ONE_MINUS_Kr = 1.0 - 0.299;
static constexpr double Y_ZERO = 16.0 * 256.0;
static constexpr double Y_SCALE = 219.0 * 256.0;
static constexpr double C_ZERO = 128.0 * 256.0;
static constexpr double C_SCALE = 112.0 * 256.0;

// kB = sqrt(209556997.0 / 96146491.0) / 3.0
// kR = sqrt(221990474.0 / 288439473.0)
// [Poynton eq 28.1 p336]
static constexpr double kB = 0.49211104112248356308804691718185;
static constexpr double kR = 0.87728321993817866838972487283129;

// [Poynton eq 25.5 p307]
static constexpr double cbScale = ONE_MINUS_Kb * kB / C_SCALE;
static constexpr double crScale = ONE_MINUS_Kr * kR / C_SCALE;

// Types of sync pulse [Poynton p521 for PAL, p502 for NTSC]
enum SyncPulseType {
    NONE = 0,
    NORMAL,
    EQUALIZATION,
    BROAD
};

class Encoder
{
public:
    // Constructor.
    // This only sets the member variables it takes as parameters; subclasses
    // must initialise the VideoParameters, compute the active region and
    // resize inputFrame.
    Encoder(QFile &inputFile, QFile &tbcFile, QFile &chromaFile, LdDecodeMetaData &metaData,
            int fieldOffset, bool isComponent);

    // Encode input RGB/YCbCr stream to TBC.
    // Returns true on success; on failure, prints an error and returns false.
    bool encode();

protected:
    qint32 encodeFrame(qint32 frameNo);
    bool encodeField(qint32 fieldNo);

    // Fill in the metadata for a generated field
    virtual void getFieldMetadata(qint32 fieldNo, LdDecodeMetaData::Field &fieldData) = 0;

    // Encode one line of a field into composite video.
    // outputC includes the chroma signal and burst.
    // outputVBS includes the luma signal, blanking and syncs.
    virtual void encodeLine(qint32 fieldNo, qint32 frameLine, const quint16 *inputData,
                            std::vector<double> &outputC, std::vector<double> &outputVBS) = 0;

    // Scale and write a line of data to one of the output files.
    // Returns true on success; on failure, prints an error and returns false.
    bool writeLine(const std::vector<double> &input, std::vector<quint16> &buffer, bool isChroma, QFile &file);

    QFile &inputFile;
    QFile &tbcFile;
    QFile &chromaFile;
    LdDecodeMetaData &metaData;
    int fieldOffset;
    bool isComponent;

    LdDecodeMetaData::VideoParameters videoParameters;
    qint32 activeWidth;
    qint32 activeHeight;
    qint32 activeLeft;
    qint32 activeTop;

    QByteArray inputFrame;
};

// Generate a gate waveform with raised-cosine transitions, with 50% points at given start and end times
static inline double raisedCosineGate(double t, double startTime, double endTime, double halfRiseTime)
{
    if (t < startTime - halfRiseTime) {
        return 0.0;
    } else if (t < startTime + halfRiseTime) {
        return 0.5 + (0.5 * sin((M_PI / 2.0) * ((t - startTime) / halfRiseTime)));
    } else if (t < endTime - halfRiseTime) {
        return 1.0;
    } else if (t < endTime + halfRiseTime) {
        return 0.5 - (0.5 * sin((M_PI / 2.0) * ((t - endTime) / halfRiseTime)));
    } else {
        return 0.0;
    }
}

#endif
