/************************************************************************

    efmtof3frames.h

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

#ifndef EFMTOF3FRAMES_H
#define EFMTOF3FRAMES_H

#include <QCoreApplication>
#include <QDebug>
#include <vector>

#include "Datatypes/f3frame.h"

class EfmToF3Frames
{
public:
    EfmToF3Frames();

    // Statistics
    struct Statistics {
        qint32 undershootSyncs;
        qint32 validSyncs;
        qint32 overshootSyncs;
        qint32 syncLoss;

        qint32 undershootFrames;
        qint32 validFrames;
        qint32 overshootFrames;

        qint64 inRangeTValues;
        qint64 outOfRangeTValues;

        qint64 validEfmSymbols;
        qint64 invalidEfmSymbols;
        qint64 correctedEfmSymbols;
    };

    std::vector<F3Frame> process(QByteArray efmDataIn, bool debugState, bool _audioIsDts);
    const Statistics &getStatistics() const;
    void reportStatistics() const;
    void reset();

private:
    bool debugOn;
    bool audioIsDts;
    Statistics statistics;
    QByteArray efmDataBuffer;
    std::vector<F3Frame> f3FramesOut;

    // State machine state definitions
    enum StateMachine {
        state_initial,
        state_findInitialSyncStage1,
        state_findInitialSyncStage2,
        state_findSecondSync,
        state_syncLost,
        state_processFrame
    };

    StateMachine currentState;
    StateMachine nextState;
    bool waitingForData;

    qint32 sequentialGoodSyncCounter;
    qint32 sequentialBadSyncCounter;
    qint32 endSyncTransition;

    void clearStatistics();

    StateMachine sm_state_initial();
    StateMachine sm_state_findInitialSyncStage1();
    StateMachine sm_state_findInitialSyncStage2();
    StateMachine sm_state_findSecondSync();
    StateMachine sm_state_syncLost();
    StateMachine sm_state_processFrame();
};

#endif // EFMTOF3FRAMES_H
