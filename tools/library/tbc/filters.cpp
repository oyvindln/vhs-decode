/************************************************************************

    filters.cpp

    ld-decode-tools TBC library
    Copyright (C) 2018-2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-decode-tools is free software: you can redistribute it and/or
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

#include "filters.h"

#include "firfilter.h"

#include <array>

// PAL - Filter at Fsc/2 (Fsc = 4433618 (/2 = 2,216,809), sample rate = 17,734,472)
// 2.2 MHz LPF - 5 Taps
// import scipy.signal
// scipy.signal.firwin(5, [2.2e6/17734472], window='hamming')
static constexpr std::array<double, 5> palLumaFilterCoeffs {
    0.03283437,  0.23959832,  0.45513461,  0.23959832,  0.03283437
};

static constexpr auto palLumaFilter = makeFIRFilter(palLumaFilterCoeffs);

// NTSC - Filter at Fsc/2 (Fsc = 3579545 (/2 = 1,789,772.5), sample rate = 14,318,180)
// (PAL-M's fSC is very close to NTSC's, so we can use the same filter for both.)
// 1.8 MHz LPF - 5 Taps
// import scipy.signal
// scipy.signal.firwin(5, [1.8e6/14318180], window='hamming')
static constexpr std::array<double, 5> ntscLumaFilterCoeffs {
    0.03275786,  0.23955702,  0.45537024,  0.23955702,  0.03275786
};
static constexpr auto ntscLumaFilter = makeFIRFilter(ntscLumaFilterCoeffs);

// Public methods ----------------------------------------------------------------------------------------------------

// Apply a FIR filter to remove PAL chroma leaving just luma
// Accepts quint16 greyscale data and returns the filtered data into
// the same array
void Filters::palLumaFirFilter(quint16 *data, qint32 dataPoints)
{
    // Filter into a temporary buffer
    QVector<quint16> tmp(dataPoints);
    palLumaFilter.apply(data, tmp.data(), dataPoints);

    // Copy the result over the original array
    for (qint32 i = 0; i < dataPoints; i++) {
        data[i] = tmp[i];
    }
}

// Apply a FIR filter to remove PAL chroma leaving just luma
// Accepts qint32 greyscale data and returns the filtered data into
// the same array
void Filters::palLumaFirFilter(QVector<qint32> &data)
{
    palLumaFilter.apply(data);
}

// Apply a FIR filter to remove NTSC chroma leaving just luma
// Accepts quint16 greyscale data and returns the filtered data into
// the same array
void Filters::ntscLumaFirFilter(quint16 *data, qint32 dataPoints)
{
    // Filter into a temporary buffer
    QVector<quint16> tmp(dataPoints);
    ntscLumaFilter.apply(data, tmp.data(), dataPoints);

    // Copy the result over the original array
    for (qint32 i = 0; i < dataPoints; i++) {
        data[i] = tmp[i];
    }
}

// Apply a FIR filter to remove NTSC chroma leaving just luma
// Accepts qint32 greyscale data and returns the filtered data into
// the same array
void Filters::ntscLumaFirFilter(QVector<qint32> &data)
{
    ntscLumaFilter.apply(data);
}

// Apply a FIR filter to remove PAL-M chroma leaving just luma
// Accepts quint16 greyscale data and returns the filtered data into
// the same array
void Filters::palMLumaFirFilter(quint16 *data, qint32 dataPoints)
{
    ntscLumaFirFilter(data, dataPoints);
}

// Apply a FIR filter to remove PAL-M chroma leaving just luma
// Accepts qint32 greyscale data and returns the filtered data into
// the same array
void Filters::palMLumaFirFilter(QVector<qint32> &data)
{
    ntscLumaFirFilter(data);
}
