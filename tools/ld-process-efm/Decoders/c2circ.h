/************************************************************************

    c2circ.h

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

#ifndef C2CIRC_H
#define C2CIRC_H

#include <QCoreApplication>
#include <QDebug>
#include <vector>

#include <ezpwd/rs_base>
#include <ezpwd/rs>

// CD-ROM specific CIRC configuration for Reed-Solomon forward error correction
template < size_t SYMBOLS, size_t PAYLOAD > struct C2RS;
template < size_t PAYLOAD > struct C2RS<255, PAYLOAD>
    : public __RS(C2RS, uint8_t, 255, PAYLOAD, 0x11d, 0, 1, false);

class C2Circ
{
public:
    C2Circ();

    // Statistics data structure
    struct Statistics {
        qint32 c2Passed;
        qint32 c2Corrected;
        qint32 c2Failed;
        qint32 c2flushed;
    };

    void reset();
    void resetStatistics();
    const Statistics &getStatistics() const;
    void reportStatistics() const;
    void pushC1(const uchar *dataSymbols, const uchar *errorSymbols);
    const uchar *getDataSymbols() const;
    const uchar *getErrorSymbols() const;
    bool getDataValid() const;
    void flush();

private:
    struct C1Element {
        uchar c1Data[28];
        uchar c1Error[28];
    };
    std::vector<C1Element> c1DelayBuffer;

    uchar interleavedC2Data[28];
    uchar interleavedC2Errors[28];

    uchar outputC2Data[28];
    uchar outputC2Errors[28];

    Statistics statistics;

    void interleave();
    void errorCorrect();
};

#endif // C2CIRC_H
