/*******************************************************************************
 * ADC.hpp
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

#include <stdexcept>


template<class DATA_SRC>
struct OneBitADC {
    // with the rolling average, this also might act as a primitive high-pass filter.
    // Compares each sample against the rolling average of the last N samples and returns high/low

    explicit OneBitADC(int buf_size, DATA_SRC &source) : source(source), buf_size(buf_size) {
        buffer = new uint8_t[buf_size];

        // initialize the buffer
        const auto default_val = 128;
        for (int i = 0; i < buf_size; ++i)
            buffer[i] = default_val;
        rolling_sum = default_val * buf_size;

        // todo; lookahead to fill buffer? requires seeking on source
        // for (int i = 0; i < buf_size; ++i) {
        //     auto byte = (uint8_t) source.get();
        //     buffer[i] = byte;
        //     total += byte;
        // }
        // source.seekg(0);
    }

    ~OneBitADC() {
        delete[] buffer;
    }

    DATA_SRC &source;
    int buf_size;
    uint8_t *buffer;
    int buffer_pos = 0;
    int rolling_sum = 0;

    // Returns 1 if the next sample is above the rolling average, 0 otherwise
    inline bool next() {
        auto byte = source.get();
        if (byte == EOF)
            throw std::range_error("EOF");

        // update the rolling sum / avg
        rolling_sum -= buffer[buffer_pos];
        buffer[buffer_pos] = byte;
        buffer_pos = (buffer_pos + 1) % buf_size;
        rolling_sum += byte;

        return byte > rolling_sum / buf_size;
    }
};
