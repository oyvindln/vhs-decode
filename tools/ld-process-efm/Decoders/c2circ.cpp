/************************************************************************

    c2circ.cpp

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

#include "c2circ.h"

C2Circ::C2Circ()
{
    reset();
}

// Method to reset and flush all buffers
void C2Circ::reset()
{
    flush();
    resetStatistics();
}

// Methods to handle statistics
void C2Circ::resetStatistics()
{
    statistics.c2Passed = 0;
    statistics.c2Corrected = 0;
    statistics.c2Failed = 0;
    statistics.c2flushed = 0;
}

const C2Circ::Statistics &C2Circ::getStatistics() const
{
    return statistics;
}

// Method to write statistics information to qInfo
void C2Circ::reportStatistics() const
{
    qInfo() << "";
    qInfo() << "F3 to F2 frame C2 Error correction:";
    qInfo() << "  Total C2s processed:" << statistics.c2Passed + statistics.c2Corrected + statistics.c2Failed;
    qInfo() << "            Valid C2s:" << statistics.c2Passed + statistics.c2Corrected;
    qInfo() << "          Invalid C2s:" << statistics.c2Failed;
    qInfo() << "        C2s corrected:" << statistics.c2Corrected;
    qInfo() << " Delay buffer flushes:" << statistics.c2flushed;
}

void C2Circ::pushC1(const uchar *dataSymbols, const uchar *errorSymbols)
{
    // Create a new C1 element and append it to the C1 delay buffer
    C1Element newC1Element;
    for (qint32 i = 0; i < 28; i++) {
        newC1Element.c1Data[i] = dataSymbols[i];
        newC1Element.c1Error[i] = errorSymbols[i];
    }
    c1DelayBuffer.push_back(newC1Element);

    if (c1DelayBuffer.size() >= 109) {
        // Maintain the C1 delay buffer at 109 elements maximum
        if (c1DelayBuffer.size() > 109) c1DelayBuffer.erase(c1DelayBuffer.begin());

        // Interleave the C1 data and perform C2 error correction
        interleave();
        errorCorrect();
    }
}

// Return the C2 data symbols if available
const uchar *C2Circ::getDataSymbols() const
{
    if (c1DelayBuffer.size() >= 109) return outputC2Data;
    return nullptr;
}

// Return the C2 error symbols if available
const uchar *C2Circ::getErrorSymbols() const
{
    if (c1DelayBuffer.size() >= 109) return outputC2Errors;
    return nullptr;
}

// Method to flush the C2 buffers
void C2Circ::flush()
{
    c1DelayBuffer.clear();

    for (qint32 i = 0; i < 28; i++) {
        interleavedC2Data[i] = 0;
        interleavedC2Errors[i] = 0;
        outputC2Data[i] = 0;
        outputC2Errors[i] = 0;
    }

    statistics.c2flushed++;
}

// Interleave the C1 data by applying delay lines of unequal length
// according to fig. 13 in IEC 60908 in order to produce the C2 data
void C2Circ::interleave()
{
    // Longest delay is 27 * 4 = 108
    for (qint32 byteC = 0; byteC < 28; byteC++) {

        qint32 delayC1Line = (108 - ((27 - byteC) * 4));
        interleavedC2Data[byteC] = c1DelayBuffer[delayC1Line].c1Data[byteC];
        interleavedC2Errors[byteC] = c1DelayBuffer[delayC1Line].c1Error[byteC];
    }
}

// Perform a C2 level error check and correction
//
// Note: RS ERC isn't a checksum and, if there are too many error/erasure symbols passed to it,
// it is possible to receive false-positive corrections.  It is essential that the inbound BER
// (Bit Error Rate) is at or below the IEC maximum of 3%.  More than this and it's likely bad
// packets will be created.
void C2Circ::errorCorrect()
{
    // The C2 error correction can correct, at most, 4 symbols

    // Convert the data and errors into the form expected by the ezpwd library
    std::vector<uint8_t> data;
    std::vector<int> erasures;
    data.resize(32);

    for (qint32 byteC = 0; byteC < 28; byteC++) {
        data[static_cast<size_t>(byteC)] = static_cast<uchar>(interleavedC2Data[byteC]);
        if (interleavedC2Errors[byteC] != static_cast<char>(0)) erasures.push_back(byteC);
    }

    // Perform error check and correction
    int fixed = -1;

    if (erasures.size() <= 4 ) {
        // Perform error check and correction

        // Initialise the error corrector
        C2RS<255,255-4> rs; // Up to 251 symbols data load with 4 symbols parity RS(32,28)

        // Perform decode
        std::vector<int> position;
        fixed = rs.decode(data, erasures, &position);

        // If there were more than 3 symbols in error, mark the C2 as an erasure
        if (fixed > 3) fixed = -1;

        if (fixed >= 0) {
            // Copy the result back to the output byte array (removing the parity symbols)
            for (qint32 byteC = 0; byteC < 28; byteC++) {
                outputC2Data[byteC] = static_cast<uchar>(data[static_cast<size_t>(byteC)]);
                if (fixed < 0) outputC2Errors[byteC] = 1; else outputC2Errors[byteC] = 0;
            }
        } else {
            // Erasure
            for (qint32 byteC = 0; byteC < 28; byteC++) {
                outputC2Data[byteC] = interleavedC2Data[byteC];
                outputC2Errors[byteC] = 1;
            }
        }
    } else {
        // If we have more than 4 input erasures we have to flag the output as erasures and
        // copy the original input data to the output (according to Sorin 2.4 p67)
        for (qint32 byteC = 0; byteC < 28; byteC++) {
            outputC2Data[byteC] = interleavedC2Data[byteC];
            outputC2Errors[byteC] = 1;
        }
        fixed = -1;
    }

    // Update the statistics
    if (fixed == 0) {
        statistics.c2Passed++;
    }
    if (fixed > 0) {
        statistics.c2Passed++;
        statistics.c2Corrected++;
    }
    if (fixed < 0) {
        statistics.c2Failed++;
    }
}
