/*******************************************************************************
 * Resampler.hpp
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


constexpr int samplesPerCarrierCycle = 16;


// // loses 64 samples at end
// template<class DATA_SRC>
// struct Resampler {
//     double m_timeIn = 0.0, m_timeOu = 0.0;
//     DATA_SRC &source;
//     bool out;
//
//     double inputSampleFreq;
//     double inpDuration = 1 / inputSampleFreq;
//     double outDuration = 1 / (2.88e6 * samplesPerCarrierCycle);
//
//     explicit Resampler(double inputSampleFreq, DATA_SRC &source) : inputSampleFreq(inputSampleFreq), source(source) {
//         out = source.next();
//     };
//
//     bool next() {
//         while (m_timeOu >= m_timeIn) {
//             m_timeIn += inpDuration;
//             out = source.next();
//         }
//         // fileout << (int) out * 2 - 1 << "\t" << m_timeOu << "\t" << m_timeIn << "\n";
//         m_timeOu += outDuration;
//         return out;
//     }
// };
