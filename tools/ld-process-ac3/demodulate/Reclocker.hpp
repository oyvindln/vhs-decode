/*******************************************************************************
 * Reclocker.hpp
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

#include <vector>

#include "Demodulator.hpp"


template<class DATA_SRC>
struct Reclocker {
    explicit Reclocker(DATA_SRC &source) : source(source) {}

    // https://diagramas.diagramasde.com/audio/SONY%20SDP-EP9ES.pdf
    // Page 18 block diagram IC901, page 36 pin descriptions
    static constexpr int counterBits = 16;
    static constexpr int nominalFrequency = 288000; // 2.88KHz QPSK eye pattern clock. PD4606A Pin 85 EPCK.
    static constexpr int sampleRate = 2.88e6 * samplesPerCarrierCycle; // PD4606A Pin 4, XIN 46.08MHz
    static constexpr int nominalAdd = int(((1LL << counterBits) * nominalFrequency) / sampleRate);

    DATA_SRC &source;
    int totalBitsIn = 0;
    int clkCounter = 0;

    uint8_t lastIn = 0;
    int error = 0;
    int errorSum = 0;
    static constexpr int maxErrorSum = 0x7ffff;
    static constexpr int minErrorSum = -0x80000;
    int filterOut = 0;

    std::vector<int> togglePositions{};

    // matches the reference scala
    // seems to detect the clock, and only output one symbol per clock
    uint8_t next() {
        while (true) {
            uint8_t dataIn = source.next();

            totalBitsIn++;

            if (dataIn != lastIn) {
                togglePositions.emplace_back(clkCounter);
                lastIn = dataIn;
            }

            int filterNow;
            if (filterOut < -nominalAdd) {
                filterOut += nominalAdd;
                filterNow = -nominalAdd;
            } else {
                filterNow = filterOut;
                filterOut = 0;
            }

            const int newCounter = (clkCounter + nominalAdd + filterNow) & ((1 << counterBits) - 1);
            if (newCounter < clkCounter) {
                if (!togglePositions.empty()) {
                    int togglePos = (togglePositions.front() + togglePositions.back()) / 2;
                    error = -(togglePos - (1 << (counterBits - 1)));
                    if (error > 0 && errorSum + error > maxErrorSum)
                        errorSum = maxErrorSum;
                    else if (error < 0 && errorSum + error < minErrorSum)
                        errorSum = minErrorSum;
                    else
                        errorSum += error;
                    filterOut = error / 128 + errorSum / (1 << 12);
                } else
                    filterOut = errorSum / (1 << 12);

                togglePositions.clear();
                clkCounter = newCounter;
                return lastIn;
            }
            clkCounter = newCounter;
        }
    }
};
