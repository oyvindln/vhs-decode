/************************************************************************

    c2deinterleave.cpp

    ld-process-efm - EFM data decoder
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-efm is free software: you can redistribute it and/or
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

#include "c2deinterleave.h"

C2Deinterleave::C2Deinterleave()
{
    reset();
}

// Method to reset and flush all buffers
void C2Deinterleave::reset()
{
    flush();
    resetStatistics();
}

// Methods to handle statistics
void C2Deinterleave::resetStatistics()
{
    statistics.validDeinterleavedC2s = 0;
    statistics.invalidDeinterleavedC2s = 0;
    statistics.c2flushed = 0;
}

const C2Deinterleave::Statistics &C2Deinterleave::getStatistics() const
{
    return statistics;
}

// Method to write statistics information to qInfo
void C2Deinterleave::reportStatistics() const
{
    qInfo() << "";
    qInfo() << "F3 to F2 frame C2 Deinterleave:";
    qInfo() << "  Total C2s processed:" << statistics.validDeinterleavedC2s + statistics.invalidDeinterleavedC2s;
    qInfo() << "            Valid C2s:" << statistics.validDeinterleavedC2s;
    qInfo() << "          Invalid C2s:" << statistics.invalidDeinterleavedC2s;
    qInfo() << " Delay buffer flushes:" << statistics.c2flushed;
}

void C2Deinterleave::pushC2(const uchar *dataSymbols, const uchar *errorSymbols)
{
    // Create a new C2 element and append it to the C2 delay buffer
    C2Element newC2Element;
    for (qint32 i = 0; i < 28; i++) {
        newC2Element.c2Data[i] = dataSymbols[i];
        newC2Element.c2Error[i] = errorSymbols[i];
    }

    c2DelayBuffer.push_back(newC2Element);

    if (c2DelayBuffer.size() >= 3) {
        // Maintain the C2 delay buffer at 3 elements maximum
        if (c2DelayBuffer.size() > 3) c2DelayBuffer.erase(c2DelayBuffer.begin());

        // Deinterleave the C2 data
        deinterleave();
    }
}

// Return the deinterleaved C2 data symbols if available
const uchar *C2Deinterleave::getDataSymbols() const
{
    if (c2DelayBuffer.size() >= 3) return outputC2Data;
    return nullptr;
}

// Return the deinterleaved C2 error symbols if available
const uchar *C2Deinterleave::getErrorSymbols() const
{
    if (c2DelayBuffer.size() >= 3) return outputC2Errors;
    return nullptr;
}

// Method to flush the C1 buffers
void C2Deinterleave::flush()
{
    c2DelayBuffer.clear();

    for (qint32 i = 0; i < 24; i++) {
        outputC2Data[i] = 0;
        outputC2Errors[i] = 0;
    }

    statistics.c2flushed++;
}

// Deinterleave C2 data as per IEC60908 Figure 13 - CIRC decoder (de-interleaving sequence)
void C2Deinterleave::deinterleave()
{
    // Element 2 is the current C2, element 0 is 2 line delays behind
    qint32 curr = 2; // C2 0-frame delay
    qint32 prev = 0; // C2 2=frame delay

    // Deinterleave data
    outputC2Data[ 0] = c2DelayBuffer[curr].c2Data[ 0];
    outputC2Data[ 1] = c2DelayBuffer[curr].c2Data[ 1];
    outputC2Data[ 2] = c2DelayBuffer[curr].c2Data[ 6];
    outputC2Data[ 3] = c2DelayBuffer[curr].c2Data[ 7];

    outputC2Data[ 8] = c2DelayBuffer[curr].c2Data[ 2];
    outputC2Data[ 9] = c2DelayBuffer[curr].c2Data[ 3];
    outputC2Data[10] = c2DelayBuffer[curr].c2Data[ 8];
    outputC2Data[11] = c2DelayBuffer[curr].c2Data[ 9];

    outputC2Data[16] = c2DelayBuffer[curr].c2Data[ 4];
    outputC2Data[17] = c2DelayBuffer[curr].c2Data[ 5];
    outputC2Data[18] = c2DelayBuffer[curr].c2Data[10];
    outputC2Data[19] = c2DelayBuffer[curr].c2Data[11];

    outputC2Data[ 4] = c2DelayBuffer[prev].c2Data[16];
    outputC2Data[ 5] = c2DelayBuffer[prev].c2Data[17];
    outputC2Data[ 6] = c2DelayBuffer[prev].c2Data[22];
    outputC2Data[ 7] = c2DelayBuffer[prev].c2Data[23];

    outputC2Data[12] = c2DelayBuffer[prev].c2Data[18];
    outputC2Data[13] = c2DelayBuffer[prev].c2Data[19];
    outputC2Data[14] = c2DelayBuffer[prev].c2Data[24];
    outputC2Data[15] = c2DelayBuffer[prev].c2Data[25];

    outputC2Data[20] = c2DelayBuffer[prev].c2Data[20];
    outputC2Data[21] = c2DelayBuffer[prev].c2Data[21];
    outputC2Data[22] = c2DelayBuffer[prev].c2Data[26];
    outputC2Data[23] = c2DelayBuffer[prev].c2Data[27];

    // Deinterleave errors
    outputC2Errors[ 0] = c2DelayBuffer[curr].c2Error[ 0];
    outputC2Errors[ 1] = c2DelayBuffer[curr].c2Error[ 1];
    outputC2Errors[ 2] = c2DelayBuffer[curr].c2Error[ 6];
    outputC2Errors[ 3] = c2DelayBuffer[curr].c2Error[ 7];

    outputC2Errors[ 8] = c2DelayBuffer[curr].c2Error[ 2];
    outputC2Errors[ 9] = c2DelayBuffer[curr].c2Error[ 3];
    outputC2Errors[10] = c2DelayBuffer[curr].c2Error[ 8];
    outputC2Errors[11] = c2DelayBuffer[curr].c2Error[ 9];

    outputC2Errors[16] = c2DelayBuffer[curr].c2Error[ 4];
    outputC2Errors[17] = c2DelayBuffer[curr].c2Error[ 5];
    outputC2Errors[18] = c2DelayBuffer[curr].c2Error[10];
    outputC2Errors[19] = c2DelayBuffer[curr].c2Error[11];

    outputC2Errors[ 4] = c2DelayBuffer[prev].c2Error[16];
    outputC2Errors[ 5] = c2DelayBuffer[prev].c2Error[17];
    outputC2Errors[ 6] = c2DelayBuffer[prev].c2Error[22];
    outputC2Errors[ 7] = c2DelayBuffer[prev].c2Error[23];

    outputC2Errors[12] = c2DelayBuffer[prev].c2Error[18];
    outputC2Errors[13] = c2DelayBuffer[prev].c2Error[19];
    outputC2Errors[14] = c2DelayBuffer[prev].c2Error[24];
    outputC2Errors[15] = c2DelayBuffer[prev].c2Error[25];

    outputC2Errors[20] = c2DelayBuffer[prev].c2Error[20];
    outputC2Errors[21] = c2DelayBuffer[prev].c2Error[21];
    outputC2Errors[22] = c2DelayBuffer[prev].c2Error[26];
    outputC2Errors[23] = c2DelayBuffer[prev].c2Error[27];

    // Check that both required C2 symbols contain valid data
    bool outputC2Valid = true;
    for (qint32 i = 0; i < 23; i++) {
        if (outputC2Errors[i] == static_cast<uchar>(1)) outputC2Valid = false;
    }

    if (outputC2Valid) statistics.validDeinterleavedC2s++;
    else statistics.invalidDeinterleavedC2s++;
}

