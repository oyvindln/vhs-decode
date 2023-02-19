/************************************************************************

    f3frame.cpp

    ld-process-efm - EFM data decoder
    Copyright (C) 2019-2022 Simon Inns
    Copyright (C) 2022 Adam Sampson

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

#include "f3frame.h"

// Note: Class for storing 'F3 frames' as defined by clause 18 of ECMA-130
//
// Each frame consists of 1 byte of subcode data and 32 bytes of payload
//
// Data is represented as data symbols (the actual payload) and error symbols
// that flag if a data symbol was detected as invalid during translation from EFM

F3Frame::F3Frame()
{
    validEfmSymbols = 0;
    invalidEfmSymbols = 0;
    correctedEfmSymbols = 0;

    isSync0 = false;
    isSync1 = false;
    subcodeSymbol = 0;

    for (qint32 i = 0; i < 32; i++) {
        dataSymbols[i] = 0;
        errorSymbols[i] = 0;
    }
}

F3Frame::F3Frame(const uchar *tValuesIn, qint32 tLength, bool audioIsDts)
{
    validEfmSymbols = 0;
    invalidEfmSymbols = 0;
    correctedEfmSymbols = 0;

    isSync0 = false;
    isSync1 = false;
    subcodeSymbol = 0;

    setTValues(tValuesIn, tLength, audioIsDts);
}

// This method sets the T-values for the F3 Frame
void F3Frame::setTValues(const uchar *tValuesIn, qint32 tLength, bool audioIsDts)
{
    // Does tValuesIn contain values?
    if (tLength == 0) {
        qDebug() << "F3Frame::setTValues(): T values array is empty!";
        return;
    }

    // Step 1:

    // Convert the T-values, which represent the spacing between 1 bits, into
    // EFM values.
    //
    // There are 33 EFM values per F3 frame (1 Subcode symbol and 32 data symbols)
    //
    // Composition of an EFM packet is as follows:
    //
    //  1 * (24 + 3) bits sync pattern         =  27
    //  1 * (14 + 3) bits control and display  =  17
    // 32 * (14 + 3) data+parity               = 544
    //                                   total = 588 bits
    //
    // We don't bother storing the sync pattern, or the 3 merging bits after
    // each EFM code.

    // Clear the EFM buffer
    qint16 efmValues[33];
    for (qint32 i = 0; i < 33; i++) efmValues[i] = 0;

    // Iterate through the T-values, keeping track of the bit position within the frame.
    // The loop executes an extra time at the end to write the final 1 bit.
    qint32 frameBits = 0;
    for (qint32 tPosition = 0; tPosition < tLength + 1; tPosition++) {
        const uchar tValue = (tPosition == tLength) ? 0 : tValuesIn[tPosition];

        // If we're inside an EFM value (not the sync pattern, or the merging
        // bits, or past the end of the buffer), write a 1 bit in the
        // appropriate place
        const qint32 efmBits = frameBits - (24 + 3);
        if (efmBits >= 0) {
            const qint32 efmIndex = efmBits / (14 + 3);
            const qint32 efmBit = efmBits % (14 + 3);
            if (efmIndex < 33 && efmBit < 14) {
                efmValues[efmIndex] |= (1 << (13 - efmBit));
            }
        }

        frameBits += tValue;
    }

    // Step 2:

    // Decode the subcode symbol.
    // Some (but not all) DTS LaserDiscs use a non-standard Sync 0 value.
    if (efmValues[0] == 0x801 || (audioIsDts && efmValues[0] == 0x812)) {
        // Sync 0
        subcodeSymbol = 0;
        isSync0 = true;
    } else if (efmValues[0] == 0x012) {
        // Sync 1
        subcodeSymbol = 0;
        isSync1 = true;
    } else {
        // Normal subcode symbol
        subcodeSymbol = static_cast<uchar>(translateEfm(efmValues[0]));
    }

    // Step 3:

    // Decode the data symbols
    for (qint32 i = 0; i < 32; i++) {
        qint32 value = translateEfm(efmValues[i + 1]);
        if (value != -1) {
            // Translated to a valid value
            dataSymbols[i] = static_cast<uchar>(value);
            errorSymbols[i] = 0;
        } else {
            // Translation was invalid, mark as error
            dataSymbols[i] = 0;
            errorSymbols[i] = 1;
        }
    }
}

// Return the number of valid EFM symbols in the frame
qint64 F3Frame::getNumberOfValidEfmSymbols() const
{
    return validEfmSymbols;
}

// Return the number of invalid EFM symbols in the frame
qint64 F3Frame::getNumberOfInvalidEfmSymbols() const
{
    return invalidEfmSymbols;
}

// Return the number of corrected EFM symbols in the frame
qint64 F3Frame::getNumberOfCorrectedEfmSymbols() const
{
    return correctedEfmSymbols;
}

// This method returns the 32 data symbols for the F3 Frame
const uchar *F3Frame::getDataSymbols() const
{
    return dataSymbols;
}

// This method returns the 32 error symbols for the F3 Frame
const uchar *F3Frame::getErrorSymbols() const
{
    return errorSymbols;
}

// This method returns the subcode symbol for the F3 frame
uchar F3Frame::getSubcodeSymbol() const
{
    return subcodeSymbol;
}

// This method returns true if the subcode symbol is a SYNC0 pattern
bool F3Frame::isSubcodeSync0() const
{
    return isSync0;
}

// This method returns true if the subcode symbol is a SYNC1 pattern
bool F3Frame::isSubcodeSync1() const
{
    return isSync1;
}

// Private methods ----------------------------------------------------------------------------------------------------

// Custom hash table mapping EFM symbols to their values.
//
// This uses 2 KiB of storage, so it will fit comfortably within L1 cache
// even on low-end machines. It does at most two lookups for each symbol,
// with each lookup needing a single 32-bit memory read.
class EfmHashTable
{
public:
    EfmHashTable() {
        // Zero out the table (since 0 is not a valid EFM symbol)
        for (qint32 i = 0; i < 512; i++) buckets[i] = 0;

        // Put the EFM codes into the table
        for (qint32 value = 0; value < 256; value++) {
            const qint16 symbol = efm2numberLUT[value];
            qint32 bucket = getHash(symbol) * 2;

            // If this bucket is already occupied, use the next one
            if (buckets[bucket] != 0) bucket++;

            // Store the value in the top half of the bucket
            buckets[bucket] = (value << 16) | symbol;
        }
    }

    // Look up the value of an EFM symbol, returning -1 if not found
    qint16 getValue(qint16 symbol) const {
        const qint32 bucket = getHash(symbol) * 2;

        // If present, the symbol must be in this bucket or the next one
        const quint32 thisBucket = buckets[bucket];
        if ((thisBucket & 0xFFFF) == static_cast<quint32>(symbol)) return thisBucket >> 16;
        const quint32 nextBucket = buckets[bucket + 1];
        if ((nextBucket & 0xFFFF) == static_cast<quint32>(symbol)) return nextBucket >> 16;

        // Not found
        return -1;
    }

private:
    // Hash an EFM value into an 8-bit result. This function was selected so
    // that at most two valid EFM codes give the same hash value.
    static qint32 getHash(qint16 symbol) {
        return (symbol ^ (symbol >> 1) ^ (symbol >> 3) ^ (symbol >> 7)) & 0xFF;
    }

    quint32 buckets[512];
};
static EfmHashTable efmHashTable;

// Method to translate 14-bit EFM value into 8-bit byte
// Returns -1 if the EFM value could not be converted (which never happens,
// since we always correct to the most likely value)
qint16 F3Frame::translateEfm(qint16 efmValue)
{
    // Look up the symbol in the hash table
    qint16 value = efmHashTable.getValue(efmValue);
    if (value != -1) {
        // Symbol was valid
        validEfmSymbols++;
        return value;
    }

    // Symbol was invalid. Correct it using cosine similarity lookup.
    invalidEfmSymbols++;
    correctedEfmSymbols++;
    return efmerr2valueLUT[efmValue & 0x3fff];
}
