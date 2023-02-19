/*******************************************************************************
 * Demodulator.hpp
 *
 * ld-process-ac3 - AC3-RF decoder
 * Copyright (C) 2022-2022 Leighton Smallshire & Ian Smallshire
 *
 * Derived from prior work by Staffan Ulfberg with feedback
 * to original author. (Copyright (C) 2021-2022)
 * https://bitbucket.org/staffanulfberg/ldaudio/src/master/
 *
 * This file is part of ld-decode-tools.
 *
 * ld-process-ac3 is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

#pragma once

#include "Resampler.hpp"

#include <cstdint>

// Return the number of 1 bits in a uint64_t
// XXX In C++20, we can use std::popcount
static inline int popcount64(uint64_t value) {
#if defined(__GNUC__)
    return __builtin_popcountll(value);
#else
    value -= (value >> 1) & 0x5555555555555555ULL;
    value = (value & 0x3333333333333333ULL) + ((value >> 2) & 0x3333333333333333ULL);
    value = (value + (value >> 4)) & 0x0F0F0F0F0F0F0F0FULL;
    return (value * 0x0101010101010101ULL) >> 56;
#endif
}

template<class DATA_SRC>
struct Demodulator {
    static constexpr int compareIntervalSize = 16;
    static constexpr int cyclesPerSymbol = 10;
    static constexpr int samplesBetweenSymbols = compareIntervalSize * cyclesPerSymbol;
    static constexpr int phaseShift = samplesPerCarrierCycle / 4;

    /* The overall idea here is that we're reading input sample bits from the
     * source, and we want to identify symbols by comparing the most recent
     * compareIntervalSize bits with the same number of bits from
     * samplesBetweenSymbols + (0..3) * phaseShift samples earlier. We do the
     * comparison by XORing the two bit-strings, and counting the number of 1
     * bits in the result.
     *
     * buffer holds the history of input bits, with buffer[0]'s LSB being the
     * most recent.
     *
     * buffer[0]: ................................................XXXXXXXXXXXXXXXX
     *
     * The samples we want to compare against are all in buffer[2] (with the
     * constant values above -- this might not be true if they were changed, in
     * which case we'd need to join together two words, so we check this below).
     *
     * buffer[2]: ................XXXXXXXXXXXXXXXX................................ (phase 0)
     * buffer[2]: ....XXXXXXXXXXXXXXXX............................................ (phase 3)
     *
     * So in next(), we just need to shift buffer[2] by the right amount, XOR
     * with buffer[0], mask off the bits we're interested in, and popcount the 1s. */

    // Find the start of the bit-string to compare against in the buffer
    static constexpr int offsetWords = samplesBetweenSymbols / 64;
    static constexpr int offsetShift = samplesBetweenSymbols % 64;

    // ... which also tells us how big the buffer needs to be
    static constexpr int bufferWords = offsetWords + 1;

    // Check how many bits are left at the MSB end of buffer[offsetWords]
    static constexpr int bitsLeft = 64 - offsetShift - (3 * phaseShift) - compareIntervalSize;
    static_assert(bitsLeft >= 0, "Would need to read two words");

    // How many samples to read into the buffer on startup.
    // (This is larger than it needs to be -- but keep it the same as the
    // original Scala code for now, so we can compare the output.)
    static constexpr int bufferPreload = samplesBetweenSymbols * 2;

    uint64_t buffer[bufferWords];
    DATA_SRC &source;

    explicit Demodulator(DATA_SRC &source) : source(source) {
        // Load bufferPreload samples into buffer
        for (int i = 0; i < bufferWords; i++)
            buffer[i] = 0;
        for (int i = 0; i < bufferPreload; i++)
            next();
    }

    // Votes on the value of a symbol from a window of samples
    char next() {
        // Read the next source sample into the LSB of buffer[0],
        // shifting the rest of buffer along by one bit
        for (int i = bufferWords - 1; i > 0; i--)
            buffer[i] = (buffer[i] << 1) | (buffer[i - 1] >> 63);
        buffer[0] = (buffer[0] << 1) | source.next();

        // Do the comparison (see above)
        int sums[4];
        for (int ph = 0; ph < 4; ph++) {
            sums[ph] = popcount64((buffer[0] ^ (buffer[offsetWords] >> (offsetShift + (ph * phaseShift))))
                                  & ((1 << compareIntervalSize) - 1));
        }

        // Work out which symbol this represents
        const int a = sums[2] - sums[0];
        const int b = sums[3] - sums[1];
        const char winner = (abs(a) > abs(b)) ? (a > 0 ? 0 : 3) : (b > 0 ? 1 : 2);

        return winner;
    }
};
