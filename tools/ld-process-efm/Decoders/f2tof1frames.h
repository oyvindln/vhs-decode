/************************************************************************

    f2framestoaudio.h

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

#ifndef F2FRAMESTOAUDIO_H
#define F2FRAMESTOAUDIO_H

#include <QCoreApplication>
#include <QDebug>
#include <vector>

#include "Datatypes/f2frame.h"
#include "Datatypes/f1frame.h"

class F2ToF1Frames
{
public:
    F2ToF1Frames();

    // Statistics
    struct Statistics {
        qint32 totalFrames;
        qint32 validF2Frames;
        qint32 invalidF2Frames;
        qint32 initialPaddingFrames;
        qint32 missingSectionFrames;
        qint32 encoderOffFrames;
        TrackTime framesStart;
        TrackTime frameCurrent;
    };

    const std::vector<F1Frame> &process(const std::vector<F2Frame> &f2FramesIn, bool _debugState, bool _noTimeStamp);
    const Statistics &getStatistics() const;
    void reportStatistics() const;
    void reset();

private:
    bool debugOn;
    bool noTimeStamp;
    Statistics statistics;

    // State-machine variables
    enum StateMachine {
        state_initial,
        state_getInitialDiscTime,
        state_processSection
    };

    StateMachine currentState;
    StateMachine nextState;
    std::vector<F2Frame> f2FrameBuffer;
    std::vector<F1Frame> f1FramesOut;
    bool waitingForData;
    TrackTime lastDiscTime;

    StateMachine sm_state_initial();
    StateMachine sm_state_getInitialDiscTime();
    StateMachine sm_state_bufferF2Frames();
    StateMachine sm_state_processSection();

    void clearStatistics();
};

#endif // F2FRAMESTOAUDIO_H
