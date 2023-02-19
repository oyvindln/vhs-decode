/************************************************************************

    f1todata.h

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

#ifndef F1TOSECTORS_H
#define F1TOSECTORS_H

#include <QCoreApplication>
#include <QDebug>
#include <vector>

#include "Datatypes/f1frame.h"
#include "Datatypes/sector.h"

class F1ToData
{
public:
    F1ToData();

    struct Statistics {
        qint32 validSectors;
        qint32 invalidSectors;
        qint32 missingSectors;
        qint32 totalSectors;
        qint32 missingSync;

        TrackTime startAddress;
        TrackTime currentAddress;
    };

    QByteArray process(const std::vector<F1Frame> &f1FramesIn, bool debugState);

    const Statistics &getStatistics() const;
    void reportStatistics() const;
    void reset();
    void clearStatistics();

private:
    bool debugOn;
    Statistics statistics;

    QByteArray f1DataBuffer;
    QByteArray f1IsCorruptBuffer;
    QByteArray f1IsMissingBuffer;

    QByteArray dataOutputBuffer;
    bool waitingForData;
    QByteArray syncPattern;
    qint32 missingSyncCount;

    TrackTime lastAddress;

    // State-machine variables
    enum StateMachine {
        state_initial,
        state_getInitialSync,
        state_getNextSync,
        state_processFrame,
        state_noSync
    };

    StateMachine currentState;
    StateMachine nextState;

    StateMachine sm_state_initial();
    StateMachine sm_state_getInitialSync();
    StateMachine sm_state_getNextSync();
    StateMachine sm_state_processFrame();
    StateMachine sm_state_noSync();
};

#endif // F1TOSECTORS_H
