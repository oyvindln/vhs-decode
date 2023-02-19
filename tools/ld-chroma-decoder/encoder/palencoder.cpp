/************************************************************************

    palencoder.cpp

    ld-chroma-encoder - Composite video encoder
    Copyright (C) 2019-2022 Adam Sampson

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

/*!
    \class PALEncoder

    This is a simplistic PAL encoder for decoder testing. The code aims to be
    accurate rather than fast.

    See \l Encoder for references.
 */

#include "palencoder.h"

#include "firfilter.h"

#include <algorithm>
#include <array>
#include <cmath>

PALEncoder::PALEncoder(QFile &_inputFile, QFile &_tbcFile, QFile &_chromaFile, LdDecodeMetaData &_metaData,
                       int _fieldOffset, bool _isComponent, bool _scLocked)
    : Encoder(_inputFile, _tbcFile, _chromaFile, _metaData, _fieldOffset, _isComponent), scLocked(_scLocked)
{
    // PAL subcarrier frequency [Poynton p529] [EBU p5]
    videoParameters.fSC = 4433618.75;

    videoParameters.sampleRate = 4 * videoParameters.fSC;

    if (scLocked) {
        // Parameters for 4fSC subcarrier-locked sampling:
        //
        // Each frame in the TBC file contains (1135 * 625) + 4 samples,
        // followed by dummy samples to fill out the rest of the "626th" line.
        // For horizontal alignment between the two fields, we treat this as:
        // - field 1: 1135 x 313 lines, plus 2 extra samples
        // - field 2: 1135 x 312 lines, plus 2 extra samples
        // - 1131 padding samples
        //
        // Each 64 usec line is 1135 + (4 / 625) samples long, so everything
        // moves to the right by (4 / 625) samples on each line. The values
        // in this struct represent the sample numbers *on the first line*.
        //
        // Each line in the output TBC consists of a series of blanking samples
        // followed by a series of active samples [EBU p9] -- different from
        // ld-decode, which starts each line with the leading edge of the
        // horizontal sync pulse (0H).
        //
        // The first sample in the TBC frame is the first blanking sample of
        // field 1 line 1, sample 948 of 1135. 0H occurs midway between samples
        // 957 and 958. [EBU p7]
        const double zeroH = 957.5 - 948;

        // Burst gate opens 5.6 usec after 0H, and closes 10 cycles later.
        // [Poynton p530]
        const double burstStartPos = zeroH + (5.6e-6 * videoParameters.sampleRate);
        const double burstEndPos = burstStartPos + (10 * 4);
        videoParameters.colourBurstStart = static_cast<qint32>(lrint(burstStartPos));
        videoParameters.colourBurstEnd = static_cast<qint32>(lrint(burstEndPos));
        // The colourburst is sampled at 0, 90, 180 and 270 degrees, so the
        // sample values are [95.5, 64, 32.5, 64] * 0x100. [Poynton p532]

        // Centre the 922 samples for 4:3 in the 948-sample digital active area
        // [Poynton p532]
        videoParameters.activeVideoStart = (1135 - 948) + ((948 - 922) / 2);
        videoParameters.activeVideoEnd = videoParameters.activeVideoStart + 922;
    } else {
        // Parameters for line-locked sampling, based on ld-decode's usual output:

        videoParameters.colourBurstStart = 98;
        videoParameters.colourBurstEnd = 138;

        videoParameters.activeVideoStart = 185;
        videoParameters.activeVideoEnd = 1107;
    }

    // Parameters that are common for subcarrier- and line-locked output:
    videoParameters.numberOfSequentialFields = 0;
    videoParameters.system = PAL;
    videoParameters.isSubcarrierLocked = scLocked;
    // White level and blanking level, extended to 16 bits [EBU p6]
    videoParameters.white16bIre = 0xD300;
    videoParameters.black16bIre = 0x4000;
    videoParameters.fieldWidth = 1135;
    videoParameters.fieldHeight = 313;
    videoParameters.isMapped = false;

    // Compute the location of the input image within the PAL frame, based on
    // the parameters above. For a 4:3 picture, there should really be 922
    // horizontal samples at 4fSC, but ld-chroma-decoder expands both sides to
    // make the width a multiple of 8 -- so centre the input across the active
    // area.
    activeWidth = 928;
    activeLeft = ((videoParameters.activeVideoStart + videoParameters.activeVideoEnd) / 2) - (activeWidth / 2);
    activeTop = 44;
    activeHeight = 620 - activeTop;

    // Resize the RGB48/YUV444P16 input buffer.
    inputFrame.resize(activeWidth * activeHeight * 3 * 2);

    // Resize the output buffers.
    Y.resize(videoParameters.fieldWidth);
    U.resize(videoParameters.fieldWidth);
    V.resize(videoParameters.fieldWidth);
}

void PALEncoder::getFieldMetadata(qint32 fieldNo, LdDecodeMetaData::Field &fieldData)
{
    fieldData.seqNo = fieldNo;
    fieldData.isFirstField = (fieldNo % 2) == 0;
    fieldData.syncConf = 100;
    // Burst peak-to-peak amplitude is 3/7 of black-white range
    fieldData.medianBurstIRE = 100.0 * (3.0 / 7.0) / 2.0;
    fieldData.fieldPhaseID = 0;
    fieldData.audioSamples = 0;
}

// Generate a gate waveform for a sync pulse in one half of a line
static double syncPulseGate(double t, double startTime, SyncPulseType type)
{
    // Timings from [Poynton p521]
    double length;
    switch (type) {
    case NONE:
        return 0.0;
    case NORMAL:
        length = 4.7e-6;
        break;
    case EQUALIZATION:
        length = 4.7e-6 / 2.0;
        break;
    case BROAD:
        length = (64.0e-6 / 2.0) - 4.7e-6;
        break;
    }

    return raisedCosineGate(t, startTime, startTime + length, 200.0e-9 / 2.0);
}

// 1.3 MHz low-pass Gaussian filter
// Generated by: c = scipy.signal.gaussian(13, 1.52); c / sum(c)
//
// The UV filter should be 0 dB at 0 Hz, >= -3 dB at 1.3 MHz, <= -20 dB at
// 4.0 MHz. [Clarke p8]
static constexpr std::array<double, 13> uvFilterCoeffs {
    0.00010852890120228184, 0.0011732778293138913, 0.008227778710181127, 0.03742748297181873, 0.11043962430879829,
    0.21139051659718247, 0.2624655813630064, 0.21139051659718247, 0.11043962430879829, 0.03742748297181873,
    0.008227778710181127, 0.0011732778293138913, 0.00010852890120228184,
};
static constexpr auto uvFilter = makeFIRFilter(uvFilterCoeffs);

void PALEncoder::encodeLine(qint32 fieldNo, qint32 frameLine, const quint16 *inputData,
                            std::vector<double> &outputC, std::vector<double> &outputVBS)
{
    if (frameLine == 625) {
        // Dummy last line, filled with black
        std::fill(outputC.begin(), outputC.end(), 0.0);
        std::fill(outputVBS.begin(), outputVBS.end(), 0.0);
        return;
    }

    // How many complete lines have gone by since the start of the 4-frame sequence?
    const qint32 fieldID = (fieldNo + fieldOffset) % 8;
    const qint32 prevLines = ((fieldID / 2) * 625) + ((fieldID % 2) * 313) + (frameLine / 2);

    // Compute the time at which 0H occurs within the line (see above)
    double zeroH;
    if (videoParameters.isSubcarrierLocked) {
        zeroH = ((957.5 - 948) + ((prevLines % 625) * (4.0 / 625))) / videoParameters.sampleRate;
    } else {
        zeroH = 0.0;
    }

    // How many cycles of the subcarrier have gone by at 0H? [Poynton p529]
    const double prevCycles = prevLines * 283.7516;

    // Compute the V-switch state and colourburst phase on this line [Poynton p530]
    const double Vsw = (prevLines % 2) == 0 ? 1.0 : -1.0;
    const double burstOffset = Vsw * 135.0 * M_PI / 180.0;

    // Burst peak-to-peak amplitude is 3/7 of black-white range [Poynton p532 eq 44.3]
    double burstAmplitude = 3.0 / 7.0;

    // Compute colourburst gating times, relative to 0H [Poynton p530]
    const double halfBurstRiseTime = 300.0e-9 / 2.0;
    const double burstStartTime = 5.6e-6;
    const double burstEndTime = burstStartTime + (10.0 / videoParameters.fSC);

    // Compute luma/chroma gating times, relative to 0H, to avoid sharp
    // transitions at the edge of the active region. The rise times are as
    // suggested in [Poynton p323], timed so that the video reaches full
    // amplitude at the start/end of the active region.
    const double halfLumaRiseTime = 2.0 / (4.0 * videoParameters.fSC);
    const double halfChromaRiseTime = 3.0 / (4.0 * videoParameters.fSC);
    double activeStartTime = (videoParameters.activeVideoStart / videoParameters.sampleRate) - zeroH - (2.0 * halfChromaRiseTime);
    double activeEndTime = (videoParameters.activeVideoEnd / videoParameters.sampleRate) - zeroH + (2.0 * halfChromaRiseTime);

    // Adjust gating for half-lines [Poynton p525]
    if (frameLine == 44) {
        activeStartTime = 42.5e-6;
    }
    if (frameLine == 619) {
        activeEndTime = 30.35e-6;
    }

    // Compute sync pulse times and pattern, relative to 0H [Poynton p520]
    // Sync level is -300mV, or 0x0100 [EBU p6]
    const double syncLevel = -0.3 / 0.7;
    const double leftSyncStartTime = 0.0;
    const double rightSyncStartTime = 64.0e-6 / 2.0;
    SyncPulseType leftSyncType = NORMAL;
    if (frameLine < 5) {
        leftSyncType = BROAD;
    } else if (frameLine >= 5 && frameLine < 10) {
        leftSyncType = EQUALIZATION;
    } else if (frameLine >= 620) {
        leftSyncType = EQUALIZATION;
    }
    SyncPulseType rightSyncType = NONE;
    if (frameLine < 4) {
        rightSyncType = BROAD;
    } else if (frameLine >= 4 && frameLine < 9) {
        rightSyncType = EQUALIZATION;
    } else if (frameLine >= 619 && frameLine < 624) {
        rightSyncType = EQUALIZATION;
    } else if (frameLine == 624) {
        rightSyncType = BROAD;
    }

    // Burst suppression [Poynton p520]
    if (leftSyncType != NORMAL) {
        burstAmplitude = 0.0;
    } else if (frameLine == 619) {
        burstAmplitude = 0.0;
    } else if (Vsw < 0 && (frameLine == 10 || frameLine == 11 || frameLine == 618)) {
        burstAmplitude = 0.0;
    }

    // Clear Y'UV buffers. Values in these are scaled so that 0.0 is black and
    // 1.0 is white.
    std::fill(Y.begin(), Y.end(), 0.0);
    std::fill(U.begin(), U.end(), 0.0);
    std::fill(V.begin(), V.end(), 0.0);

    if (inputData != nullptr) {
        if (isComponent) {
            // Convert the Y'CbCr data to Y'UV form [Poynton p307 eq 25.5]
            int stride = activeWidth * activeHeight;
            for (qint32 i = 0; i < activeWidth; i++) {
                qint32 x = activeLeft + i;
                if (videoParameters.isSubcarrierLocked && (fieldNo % 2) == 1) {
                    x += 2;
                }
                Y[x] = (inputData[i] - Y_ZERO) / Y_SCALE;
                U[x] = (inputData[i + stride    ] - C_ZERO) * cbScale;
                V[x] = (inputData[i + stride * 2] - C_ZERO) * crScale;
            }
        } else {
            // Convert the R'G'B' data to Y'UV form [Poynton p337 eq 28.5]
            for (qint32 i = 0; i < activeWidth; i++) {
                const double R = inputData[i * 3]       / 65535.0;
                const double G = inputData[(i * 3) + 1] / 65535.0;
                const double B = inputData[(i * 3) + 2] / 65535.0;

                qint32 x = activeLeft + i;
                if (videoParameters.isSubcarrierLocked && (fieldNo % 2) == 1) {
                    x += 2;
                }
                Y[x] = (R * 0.299)     + (G * 0.587)     + (B * 0.114);
                U[x] = (R * -0.147141) + (G * -0.288869) + (B * 0.436010);
                V[x] = (R * 0.614975)  + (G * -0.514965) + (B * -0.100010);
            }
        }
        // Low-pass filter U and V to 1.3 MHz [Poynton p342]
        uvFilter.apply(U);
        uvFilter.apply(V);
    }

    for (qint32 x = 0; x < videoParameters.fieldWidth; x++) {
        // For this sample, compute time relative to 0H, and subcarrier phase
        const double t = (x / videoParameters.sampleRate) - zeroH;
        const double a = 2.0 * M_PI * ((videoParameters.fSC * t) + prevCycles);

        // Generate colourburst
        const double burst = sin(a + burstOffset) * burstAmplitude / 2.0;

        // Encode the chroma signal [Poynton p338]
        const double chroma = (U[x] * sin(a)) + (V[x] * cos(a) * Vsw);

        // Generate C output
        const double burstGate = raisedCosineGate(t, burstStartTime, burstEndTime, halfBurstRiseTime);
        const double chromaGate = raisedCosineGate(t, activeStartTime, activeEndTime, halfChromaRiseTime);
        outputC[x] = (burst * burstGate)
                     + qBound(-chromaGate, chroma, chromaGate);

        // Generate VBS output
        const double lumaGate = raisedCosineGate(t, activeStartTime, activeEndTime, halfLumaRiseTime);
        const double leftSyncGate = syncPulseGate(t, leftSyncStartTime, leftSyncType);
        const double rightSyncGate = syncPulseGate(t, rightSyncStartTime, rightSyncType);
        outputVBS[x] = qBound(-lumaGate, Y[x], lumaGate)
                       + (syncLevel * (leftSyncGate + rightSyncGate));
    }
}
