/************************************************************************

    c1circ.cpp

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

#include "c1circ.h"

C1Circ::C1Circ()
{
    reset();
}

// Method to reset and flush all buffers
void C1Circ::reset()
{
    flush();
    resetStatistics();
}

// Methods to handle statistics
void C1Circ::resetStatistics()
{
    statistics.c1Passed = 0;
    statistics.c1Corrected = 0;
    statistics.c1Failed = 0;
    statistics.c1flushed = 0;
}

const C1Circ::Statistics &C1Circ::getStatistics() const
{
    return statistics;
}

// Method to write statistics information to qInfo
void C1Circ::reportStatistics() const
{
    qInfo() << "";
    qInfo() << "F3 to F2 frame C1 Error correction:";
    qInfo() << "  Total C1s processed:" << statistics.c1Passed + statistics.c1Corrected + statistics.c1Failed;
    qInfo() << "            Valid C1s:" << statistics.c1Passed + statistics.c1Corrected;
    qInfo() << "          Invalid C1s:" << statistics.c1Failed;
    qInfo() << "        C1s corrected:" << statistics.c1Corrected;
    qInfo() << " Delay buffer flushes:" << statistics.c1flushed;

    double c1ErrorRate = static_cast<double>(statistics.c1Passed) +
            static_cast<double>(statistics.c1Failed) +
            static_cast<double>(statistics.c1Corrected);

    c1ErrorRate = (100 / c1ErrorRate) * (statistics.c1Failed + statistics.c1Corrected);
    qInfo().nospace() << "        C1 Error rate: " << c1ErrorRate << "%";
}

void C1Circ::pushF3Frame(F3Frame f3Frame)
{
    for (qint32 i = 0; i < 32; i++) {
        previousF3Data[i] = currentF3Data[i];
        previousF3Errors[i] = currentF3Errors[i];

        currentF3Data[i] = f3Frame.getDataSymbols()[i];
        currentF3Errors[i] = f3Frame.getErrorSymbols()[i];
    }

    c1BufferLevel++;
    if (c1BufferLevel > 1) {
        c1BufferLevel = 2;

        // Interleave the F3 data and perform C1 error correction
        interleave();
        errorCorrect();
    }
}

// Return the C1 data symbols if available
const uchar *C1Circ::getDataSymbols() const
{
    if (c1BufferLevel > 1) return outputC1Data;
    return nullptr;
}

// Return the C1 error symbols if available
const uchar *C1Circ::getErrorSymbols() const
{
    if (c1BufferLevel > 1) return outputC1Errors;
    return nullptr;
}

// Method to flush the C1 buffers
void C1Circ::flush()
{
    for (qint32 i = 0; i < 32; i++) {
        currentF3Data[i] = 0;
        previousF3Data[i] = 0;
        currentF3Errors[i] = 0;
        previousF3Errors[i] = 0;
    }

    for (qint32 i = 0; i < 28; i++) {
        outputC1Data[i] = 0;
        outputC1Errors[i] = 0;
    }

    c1BufferLevel = 0;

    statistics.c1flushed++;
}

// Interleave current and previous F3 frame symbols and then invert parity symbols
void C1Circ::interleave()
{
    // Interleave the symbols
    for (qint32 byteC = 0; byteC < 32; byteC += 2) {
        interleavedC1Data[byteC] = currentF3Data[byteC];
        interleavedC1Data[byteC+1] = previousF3Data[byteC+1];

        interleavedC1Errors[byteC] = currentF3Errors[byteC];
        interleavedC1Errors[byteC+1] = previousF3Errors[byteC+1];
    }

    // Invert the Qm parity symbols
    interleavedC1Data[12] = static_cast<uchar>(interleavedC1Data[12]) ^ 0xFF;
    interleavedC1Data[13] = static_cast<uchar>(interleavedC1Data[13]) ^ 0xFF;
    interleavedC1Data[14] = static_cast<uchar>(interleavedC1Data[14]) ^ 0xFF;
    interleavedC1Data[15] = static_cast<uchar>(interleavedC1Data[15]) ^ 0xFF;

    // Invert the Pm parity symbols
    interleavedC1Data[28] = static_cast<uchar>(interleavedC1Data[28]) ^ 0xFF;
    interleavedC1Data[29] = static_cast<uchar>(interleavedC1Data[29]) ^ 0xFF;
    interleavedC1Data[30] = static_cast<uchar>(interleavedC1Data[30]) ^ 0xFF;
    interleavedC1Data[31] = static_cast<uchar>(interleavedC1Data[31]) ^ 0xFF;
}

// Perform a C1 level error check and correction
//
// Note: RS ERC isn't a checksum and, if there are too many error/erasure symbols passed to it,
// it is possible to receive false-positive corrections.  It is essential that the inbound BER
// (Bit Error Rate) is at or below the IEC maximum of 3%.  More than this and it's likely bad
// packets will be created.
void C1Circ::errorCorrect()
{
    // The C1 error correction can correct, at most, 2 symbols

    // Convert the data and errors into the form expected by the ezpwd library
    std::vector<uint8_t> data;
    std::vector<int> erasures;
    data.resize(32);

    for (qint32 byteC = 0; byteC < 32; byteC++) {
        data[static_cast<size_t>(byteC)] = static_cast<uchar>(interleavedC1Data[byteC]);
        if (interleavedC1Errors[byteC] == static_cast<char>(1)) erasures.push_back(byteC);
    }

    // Perform error check and correction
    int fixed = -1;

    if (erasures.size() <= 2) {
        // Perform error check and correction

        // Initialise the error corrector
        C1RS<255,255-4> rs; // Up to 251 symbols data load with 4 symbols parity RS(32,28)

        // Perform decode
        std::vector<int> position;
        fixed = rs.decode(data, erasures, &position);

        // If there were more than 2 symbols in error, mark the C1 as an erasure
        if (fixed > 2) fixed = -1;

        if (fixed >= 0) {
            // Copy the result back to the output byte array (removing the parity symbols)
            for (qint32 byteC = 0; byteC < 28; byteC++) {
                outputC1Data[byteC] = static_cast<uchar>(data[static_cast<size_t>(byteC)]);
                if (fixed < 0) outputC1Errors[byteC] = 1; else outputC1Errors[byteC] = 0;
            }
        } else {
            // Erasure
            for (qint32 byteC = 0; byteC < 28; byteC++) {
                outputC1Data[byteC] = interleavedC1Data[byteC];
                outputC1Errors[byteC] = 1;
            }
        }
    } else {
        // If we have more than 2 input erasures we have to flag the output as erasures and
        // copy the original input data to the output (according to Sorin 2.4 p66)
        for (qint32 byteC = 0; byteC < 28; byteC++) {
            outputC1Data[byteC] = interleavedC1Data[byteC];
            outputC1Errors[byteC] = 1;
        }
        fixed = -1;
    }

    // Update the statistics
    if (fixed == 0) {
        statistics.c1Passed++;
    }
    if (fixed > 0) {
        statistics.c1Passed++;
        statistics.c1Corrected++;
    }
    if (fixed < 0) {
        statistics.c1Failed++;
    }
}

