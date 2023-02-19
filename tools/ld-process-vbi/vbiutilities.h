/************************************************************************

    vbiutilities.h

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2022 Adam Sampson

    This file is part of ld-decode-tools.

    ld-process-vbi is free software: you can redistribute it and/or
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

#ifndef VBIUTILITIES_H
#define VBIUTILITIES_H

// Common utility functions for VBI line decoders

#include <QtGlobal>
#include <QVector>

// Check data for even parity
template <typename Unsigned>
static inline bool isEvenParity(Unsigned data)
{
    qint32 count = 0;

    // Count non-zero bits
    // XXX In C++20, we can use std::popcount
    while (data != 0) {
        if ((data & 1) != 0) count++;
        data >>= 1;
    }

    return (count % 2) == 0;
}

// Convert input samples into a vector of binary values
template <typename InputData>
static inline QVector<bool> getTransitionMap(const InputData &lineData, qint32 zcPoint)
{
    // Read the data into a boolean array using debounce to remove transition noise
    bool previousState = false;
    bool currentState = false;
    qint32 debounce = 0;
    QVector<bool> transitionMap;

    // Each value is 2 bytes (16-bit greyscale data)
    for (qint32 xPoint = 0; xPoint < lineData.size(); xPoint++) {
        if (lineData[xPoint] > zcPoint) currentState = true; else currentState = false;

        if (currentState != previousState) debounce++;

        if (debounce > 3) {
            debounce = 0;
            previousState = currentState;
        }

        transitionMap.append(previousState);
    }

    return transitionMap;
}

// Find the next sample with a given value in the output of getTransitionMap.
// Return true if found before the limit, false if not found.
static inline bool findTransition(const QVector<bool> &transitionMap, bool wantValue,
                                  double &position, double positionLimit)
{
    while (position < positionLimit) {
        if (transitionMap[static_cast<qint32>(position)] == wantValue) return true;
        position += 1.0;
    }
    return false;
}

#endif
