/************************************************************************

    ntscencoder.cpp

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

/*!
    \class NTSCEncoder

    This is a simplistic NTSC encoder for decoder testing. The code aims to be
    accurate rather than fast.

    See \l Encoder for references.
 */

#include "ntscencoder.h"

#include "firfilter.h"

#include <algorithm>
#include <array>
#include <cmath>

NTSCEncoder::NTSCEncoder(QFile &_inputFile, QFile &_tbcFile, QFile &_chromaFile, LdDecodeMetaData &_metaData,
                         int _fieldOffset, bool _isComponent, ChromaMode _chromaMode, bool _addSetup)
    : Encoder(_inputFile, _tbcFile, _chromaFile, _metaData, _fieldOffset, _isComponent),
      chromaMode(_chromaMode), addSetup(_addSetup)
{
    // NTSC subcarrier frequency [Poynton p511]
    videoParameters.fSC = 315.0e6 / 88.0;

    videoParameters.sampleRate = 4 * videoParameters.fSC;

    // Parameters for 4fSC subcarrier-locked sampling:
    //
    // Each frame in the TBC file contains (910 * 526) samples.
    // The last line is ignored.
    //
    // Each 63.555 usec line is 910 samples long. The values
    // in this struct represent the sample numbers *on the first line*.
    //
    // Each line in the output TBC consists of a series of blanking samples
    // followed by a series of active samples [SMPTE p4] -- different from
    // ld-decode, which starts each line with the leading edge of the
    // horizontal sync pulse (0H).
    //
    // The first sample in the TBC frame is the first blanking sample of
    // field 1 line 1, sample 768 of 910. 0H occurs 33/90 between samples
    // 784 and 785. [SMPTE p4]
    const double zeroH = 784 + 33.0 / 90.0 - 768;

    // Burst gate opens 19 cycles after 0H, and closes 9 cycles later.
    // [Poynton p512]
    const double burstStartPos = zeroH + (19 * 4);
    const double burstEndPos = burstStartPos + (9 * 4);
    videoParameters.colourBurstStart = static_cast<qint32>(lrint(burstStartPos));
    videoParameters.colourBurstEnd = static_cast<qint32>(lrint(burstEndPos));
    // The colorburst is sampled at -33, 57, 123 and 213 degrees, so the
    // sample values are [46, 83, 74, 37] * 0x100. [Poynton p517]

    // Center the 757+ analog active samples in the 768-sample digital active
    // area. [Poynton p517-518]
    videoParameters.activeVideoStart = (910 - 768) + ((768 - 758) / 2);
    videoParameters.activeVideoEnd = videoParameters.activeVideoStart + 758;

    videoParameters.numberOfSequentialFields = 0;
    videoParameters.system = NTSC;
    videoParameters.isSubcarrierLocked = true;

    // White level, black level, and blanking level, extended to 16 bits
    // [SMPTE p2, Poynton p517]
    videoParameters.white16bIre = 0xC800;
    videoParameters.black16bIre = blankingIre + (addSetup ? setupIreOffset : 0);
    videoParameters.fieldWidth = 910;
    videoParameters.fieldHeight = 263;
    videoParameters.isMapped = false;

    // Compute the location of the input image within the NTSC frame, based on
    // the parameters above.
    activeWidth = 758;
    activeLeft = ((videoParameters.activeVideoStart + videoParameters.activeVideoEnd) / 2) - (activeWidth / 2);
    activeTop = 39;
    activeHeight = 525 - activeTop;

    // Resize the RGB48/YUV444P16 input buffer.
    inputFrame.resize(activeWidth * activeHeight * 3 * 2);

    // Resize the output buffers.
    Y.resize(videoParameters.fieldWidth);
    C1.resize(videoParameters.fieldWidth);
    C2.resize(videoParameters.fieldWidth);
}

void NTSCEncoder::getFieldMetadata(qint32 fieldNo, LdDecodeMetaData::Field &fieldData)
{
    fieldData.seqNo = fieldNo;
    fieldData.isFirstField = (fieldNo % 2) == 0;
    fieldData.syncConf = 100;
    fieldData.medianBurstIRE = 20;
    fieldData.fieldPhaseID = ((fieldNo + fieldOffset) % 4) + 1;
    fieldData.audioSamples = 0;
}

// Generate a gate waveform for a sync pulse in one half of a line
static double syncPulseGate(double t, double startTime, SyncPulseType type)
{
    // Timings from [Poynton p502]
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
        length = ((63 + 5.0/9.0) / 2.0) * 1e-6 - 4.7e-6;
        break;
    }

    return raisedCosineGate(t, startTime, startTime + length, 200.0e-9 / 2.0);
}

// 1.3 MHz low-pass filter
//
// The filter should be 0 dB at 0 Hz, >= -2 dB at 1.3 MHz, < -20 dB at
// 3.6 MHz. [Clarke p15]
static constexpr std::array<double, 9> uvFilterCoeffs {
    0.0021, 0.0191, 0.0903, 0.2308, 0.3153,
    0.2308, 0.0903, 0.0191, 0.0021
};
static constexpr auto uvFilter = makeFIRFilter(uvFilterCoeffs);

// 0.6 MHz low-pass filter
//
// The filter should be 0 dB at 0 Hz, >= -2 dB at 0.4 MHz, >= -6 dB at
// 0.5 MHz, <= -6 dB at 0.6 MHz. [Clarke p15]
static constexpr std::array<double, 23> qFilterCoeffs {
    0.0002, 0.0027, 0.0085, 0.0171, 0.0278, 0.0398, 0.0522, 0.0639, 0.0742, 0.0821, 0.0872, 0.0889,
    0.0872, 0.0821, 0.0742, 0.0639, 0.0522, 0.0398, 0.0278, 0.0171, 0.0085, 0.0027, 0.0002
};
static constexpr auto qFilter = makeFIRFilter(qFilterCoeffs);

// Y'IQ [Poynton p367 eq 30.2]
static const double SIN_33 = sin(33.0 * M_PI / 180.0);
static const double COS_33 = cos(33.0 * M_PI / 180.0);

void NTSCEncoder::encodeLine(qint32 fieldNo, qint32 frameLine, const quint16 *inputData,
                             std::vector<double> &outputC, std::vector<double> &outputVBS)
{
    if (frameLine == 525) {
        // Dummy last line, filled with blanking
        std::fill(outputC.begin(), outputC.end(), 0.0);
        const double blanking = (static_cast<double>(blankingIre) - videoParameters.black16bIre)
                                / (videoParameters.white16bIre - videoParameters.black16bIre);
        std::fill(outputVBS.begin(), outputVBS.end(), blanking);

        return;
    }

    // How many complete lines have gone by since the start of the 4-field
    // sequence?
    const qint32 fieldID = (fieldNo + fieldOffset) % 4;
    const qint32 prevLines = ((fieldID / 2) * 525) + ((fieldID % 2) * 263) + (frameLine / 2);

    // Compute the time at which 0H occurs within the line (see above).
    const double zeroH = (784 + 33.0 / 90.0 - 768) / videoParameters.sampleRate;

    // How many cycles of the subcarrier have gone by at 0H?
    // There are 227.5 cycles per line (910/4). [Poynton p511]
    // Subtract 1/4 cycle because the burst is inverted but it should be
    // crossing zero and going positive at the start of the field sequence.
    const double prevCycles = (prevLines * 227.5) - 0.25;

    // The colorburst is inverted from subcarrier [SMPTE p4] [Poynton p512]
    const double burstOffset = 180.0 * M_PI / 180.0;

    // Burst peak-to-peak amplitude is 2/5 of black-white range
    // [Poynton p516 eq 42.6]
    double burstAmplitude = 2.0 / 5.0;

    // Compute colorburst gating times, relative to 0H [Poynton p512]
    const double halfBurstRiseTime = 300.0e-9 / 2.0;
    const double burstStartTime = 19.0 / videoParameters.fSC;
    const double burstEndTime = burstStartTime + (9.0 / videoParameters.fSC);

    // Compute luma/chroma gating times, relative to 0H, to avoid sharp
    // transitions at the edge of the active region. The rise times are as
    // suggested in [Poynton p323], timed so that the video reaches full
    // amplitude at the start/end of the active region.
    const double halfLumaRiseTime = 2.0 / (4.0 * videoParameters.fSC);
    const double halfChromaRiseTime = 3.0 / (4.0 * videoParameters.fSC);
    double activeStartTime = (videoParameters.activeVideoStart / videoParameters.sampleRate) - zeroH - (2.0 * halfChromaRiseTime);
    double activeEndTime = (videoParameters.activeVideoEnd / videoParameters.sampleRate) - zeroH + (2.0 * halfChromaRiseTime);

    // Adjust gating for half-lines [Poynton p506]
    if (frameLine == 39) {
        activeStartTime = 41.259e-6;
    }
    if (frameLine == 524) {
        activeEndTime = 30.593e-6;
    }

    // Compute sync pulse times and pattern, relative to 0H [Poynton p520]
    // Sync level is -285.7mV, or 0x1000 [SMPTE p2]
    const double syncLevel = -285.7 / 714.3;
    const double leftSyncStartTime = 0.0;
    const double rightSyncStartTime = (63 + 5.0/9.0) / 2.0 * 1e-6;
    SyncPulseType leftSyncType = NORMAL;
    if (frameLine < 6) {
        leftSyncType = EQUALIZATION;
    } else if (frameLine >= 6 && frameLine < 12) {
        leftSyncType = BROAD;
    } else if (frameLine < 18) {
        leftSyncType = EQUALIZATION;
    }
    SyncPulseType rightSyncType = NONE;
    if (frameLine < 5) {
        rightSyncType = EQUALIZATION;
    } else if (frameLine >= 5 && frameLine < 11) {
        rightSyncType = BROAD;
    } else if (frameLine < 17) {
        rightSyncType = EQUALIZATION;
    } else if (frameLine == 524) {
        rightSyncType = EQUALIZATION;
    }

    // Burst suppression in VBI [SMPTE 170M p9]
    if (frameLine < 18) {
        burstAmplitude = 0.0;
    }

    // Clear output buffers. Values in these are scaled so that 0.0 is black and
    // 1.0 is white.
    std::fill(Y.begin(), Y.end(), 0.0);
    std::fill(C1.begin(), C1.end(), 0.0);
    std::fill(C2.begin(), C2.end(), 0.0);

    if (inputData != nullptr) {
        if (isComponent) {
            // Convert the Y'CbCr data to Y'UV form [Poynton p307 eq 25.5]
            int stride = activeWidth * activeHeight;
            for (qint32 i = 0; i < activeWidth; i++) {
                qint32 x = activeLeft + i;
                Y[x] = (inputData[i] - Y_ZERO) / Y_SCALE;
                const double U    = (inputData[i + stride    ] - C_ZERO) * cbScale;
                const double V    = (inputData[i + stride * 2] - C_ZERO) * crScale;
                if (chromaMode == WIDEBAND_YUV) {
                    C1[x] = U;
                    C2[x] = V;
                } else {
                    // Rotate 33 degrees to create Y'IQ
                    C1[x] = -SIN_33 * U + COS_33 * V;
                    C2[x] =  COS_33 * U + SIN_33 * V;
                }
            }
        } else {
            // Convert the R'G'B' data to Y'UV/Y'IQ
            for (qint32 i = 0; i < activeWidth; i++) {
                const double R = inputData[i * 3]       / 65535.0;
                const double G = inputData[(i * 3) + 1] / 65535.0;
                const double B = inputData[(i * 3) + 2] / 65535.0;
                qint32 x = activeLeft + i;
                Y[x] = (R * 0.299)    + (G * 0.587)     + (B * 0.114);
                if (chromaMode == WIDEBAND_YUV) {
                    // Y'UV [Poynton p337 eq 28.5]
                    C1[x] = (R * -0.147141) + (G * -0.288869) + (B * 0.436010);
                    C2[x] = (R * 0.614975)  + (G * -0.514965) + (B * -0.100010);
                } else {
                    // Y'IQ [Poynton p367 eq 30.2]
                    C1[x] = (R * 0.595901) + (G * -0.274557) + (B * -0.321344);
                    C2[x] = (R * 0.211537) + (G * -0.522736) + (B * 0.311200);
                }
            }
        }

        // Low-pass filter chroma components to 1.3 MHz [Poynton p342]
        uvFilter.apply(C1);
        if (chromaMode == NARROWBAND_Q) {
            qFilter.apply(C2);
        } else {
            uvFilter.apply(C2);
        }
    }

    for (qint32 x = 0; x < videoParameters.fieldWidth; x++) {
        // For this sample, compute time relative to 0H, and subcarrier phase
        const double t = (x / videoParameters.sampleRate) - zeroH;
        const double a = 2.0 * M_PI * ((videoParameters.fSC * t) + prevCycles);

        // Generate colorburst
        const double burst = sin(a + burstOffset) * burstAmplitude / 2.0;

        // Encode the chroma signal
        double chroma;
        if (chromaMode == WIDEBAND_YUV) {
            // Y'UV [Poynton p338]
            chroma = C1[x] * sin(a) + C2[x] * cos(a);
        } else {
            // Y'IQ [Poynton p368]
            chroma = C2[x] * sin(a + 33.0 * M_PI / 180.0) + C1[x] * cos(a + 33.0 * M_PI / 180.0);
        }

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
