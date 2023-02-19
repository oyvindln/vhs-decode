/*******************************************************************************
 * Corrector.hpp
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

#include "Blocker.hpp"
#include "ezpwd/rs"


// FCR = First consecutive root
template<size_t SYMBOLS, size_t PAYLOAD>
struct AC3RS;


template<size_t PAYLOAD> struct AC3RS<255, PAYLOAD> : public __RS(AC3RS, uint8_t, 255, PAYLOAD, 0x187, 120, 1, false);


template<class DATA_SRC>
struct Corrector {
    explicit Corrector(DATA_SRC &source) : source(source) {}

    DATA_SRC &source;
    AC3RS<255, 255 - (36 - 32)> RS; // RS(36,32)
    AC3RS<255, 255 - (37 - 33)> RS2; // RS(37,33)

    std::map<int, int> stats;
    std::map<int, int> total_stats;

    // Applies reed-solomon error correction and removes padding. return value is variable-length
    auto next() {
        QPSKBlock block = source.next();
        std::vector<uint8_t> data; // 'correctedDataBuffer' in scala
        data.reserve(36 * 66);

        bool erasures[72 * 37] = {false};

        // C1
        for (int rowI = 0; rowI < 36; ++rowI) { // 36 rows of 74
            for (int odd = 0; odd < 2; ++odd) { // odd vs even bytes
                uint8_t codeword[37];
                for (int i = 0; i < 37; ++i)
                    codeword[i] = block.bytes[rowI * 74 + i * 2 + odd];
                auto r = RS2.decode(codeword, 37);
                stats[r]++;

                if (r == -1)
                    for (int i = 0; i < 37; ++i)
                        erasures[rowI * 72 + i * 2 + odd] = true;

                // todo; if (r == -1) mark all bytes in frame as erasures for C2

                // put the word back; todo improved method
                for (int i = 0; i < 37; ++i)
                    block.bytes[rowI * 74 + i * 2 + odd] = codeword[i];
            }
        }
        // print & clear stats
        auto *logger = new Logger(INFO, "C1");
        for (int i = -1; i < 3; ++i)
            *logger << stats[i] << "\t";
        *logger << "-\t-\t";
        delete logger;

        for (auto p: stats)
            total_stats[p.first] += p.second;
        stats.clear();

        // bool c2_erasures[72 * 37] = {false};
        // C2
        for (int k = 0; k < 66; ++k) {
            uint8_t codeword[36];
            std::vector<unsigned> codeword_erasures{4};

            for (int i = 0; i < 36; ++i) {
                codeword[i] = block.bytes[k + i * 74];
                if (erasures[k + i * 74])
                    codeword_erasures.emplace_back(i);
            }

            // auto r1 = RS.decode(codeword, 36);
            int r;
            if (codeword_erasures.size() > RS.nroots())
                r = -1;
            else
                r = RS.decode(codeword, 32, codeword + 32, codeword_erasures.data(), codeword_erasures.size());
            stats[r]++;

            // todo what to do with dead blocks?
            // if (r == -1)
            //     for (int i = 0; i<36; i++)
            //         c2_erasures[k+i*74] = true;

            if (k == 0) {
                if (codeword[0] != 0x10 || codeword[1] != 0x00) {
                    fprintf(stderr, "Block does not start with 0x10\n");
                } else // copy skipping first two bytes
                    // data.insert(data.end(), codeword.begin() + 2, codeword.end());
                    data.insert(data.end(), &codeword[2], &codeword[32]);
            } else // just copy
                // data.insert(data.end(), codeword.begin(), codeword.end());
                data.insert(data.end(), &codeword[0], &codeword[32]);
        }

        // print & clear stats
        logger = new Logger(INFO, "C2");
        for (int i = -1; i < 5; ++i)
            *logger << stats[i] << "\t";
        delete logger;
        for (auto p: stats)
            total_stats[p.first] += p.second;
        stats.clear();
        // }

        data.shrink_to_fit();
        return data; // max size 66 * 36 = 2376
    }
};
