/*******************************************************************************
 * AC3Framer.hpp
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

#include "Corrector.hpp"


template<class DATA_SRC>
struct StreamBuffer {
    explicit StreamBuffer(DATA_SRC &source) : source(source) {}

    DATA_SRC &source;
    static const unsigned long buffer_size = 66 * 32 * 2;
    uint8_t buffer[buffer_size]{};
    unsigned long buffer_pos = 0;

    // automatically gets the next block as needed, allows for treating the stream as a contiguous array
    auto operator[](unsigned long pos) {
        if (pos >= buffer_pos) {
            auto frame = source.next();
            auto remaining_before_loop = buffer_size - (buffer_pos % buffer_size);
            if (frame.size() <= remaining_before_loop) { // no need to wrap
                std::memcpy(buffer + (buffer_pos % buffer_size), frame.data(), frame.size());
            } else { // need to wrap; copy in two parts
                std::memcpy(buffer + (buffer_pos % buffer_size), frame.data(), remaining_before_loop);
                std::memcpy(buffer, frame.data() + remaining_before_loop, frame.size() - remaining_before_loop);
            }
            buffer_pos += frame.size();
        }
        return buffer[pos % buffer_size];
    }
};


template<class DATA_SRC>
struct AC3Framer {
    explicit AC3Framer(StreamBuffer<DATA_SRC> &buffer) : buffer(buffer) {}

    StreamBuffer<DATA_SRC> &buffer;

    unsigned long bytePosition = 0;
    int currentAc3Size = 0;
    std::vector<uint8_t> ac3Buffer;
    bool inSync = false;

    bool checkStartOfAc3Block(unsigned long offset) {
        return buffer[offset + 0] == 0x0B && buffer[offset + 1] == 0x77;
    }

    // extract and assemble the AC3 frames from the bit/bytestream
    auto next() {
        ac3Buffer.clear();
        while (true) { // until EOF, caught externally
            if (currentAc3Size == 0) {
                while (buffer[bytePosition] == 0x00)
                    bytePosition++; // skip padding

                if (checkStartOfAc3Block(bytePosition)) {
                    // http://www.atsc.org/wp-content/uploads/2015/03/A52-201212-17.pdf pg 51-52
                    currentAc3Size = 768 * 2; // todo; lookup from frame frmsizecod and fscod (buf[bytePosition+4])
                    ac3Buffer.reserve(currentAc3Size);
                    inSync = true;

                } else { // not the start of a block;
                    if (inSync)
                        // Logger(WARN, "WARN") << "Non-zero byte that does not seem to be the start of AC3 block "
                        //                << "-- searching for next block of zeros";
                        ;
                    inSync = false;
                    while (buffer[bytePosition] != 0x00)
                        bytePosition++; // skip until we find some padding
                }
            }

            if (currentAc3Size != 0) { // append to the ac3 block
                // ac3Buffer.emplace_back(buffer[i]); // todo
                ac3Buffer.insert(ac3Buffer.end(), buffer[bytePosition]);
                if (static_cast<int>(ac3Buffer.size()) == currentAc3Size) {
                    currentAc3Size = 0;
                    // XXX fscod and frmsizecod fixed until lookup above implemented;
                    // for now, SyncFrame::check_crc will check the equivalent of assert(ac3Buffer[4] == 0x1c)
                    return ac3Buffer;
                }
            }
            bytePosition++;
        }
    }
};
