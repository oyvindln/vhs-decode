/************************************************************************

    f1toaudio.h

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

#ifndef F1TOAUDIO_H
#define F1TOAUDIO_H

#include <QCoreApplication>
#include <QDebug>
#include <vector>

#include "Datatypes/f1frame.h"
#include "Datatypes/audio.h"

class F1ToAudio
{
public:
    F1ToAudio();

    // Options for the treatment of audio errors
    enum ErrorTreatment {
        conceal,
        silence,
        passThrough
    };

    // Options for concealment of audio errors
    enum ConcealType {
        linear,
        prediction
    };

    struct Statistics {
        qint32 audioSamples;
        qint32 corruptSamples;
        qint32 missingSamples;
        qint32 concealedSamples;
        qint32 totalSamples;

        TrackTime startTime;
        TrackTime currentTime;
        TrackTime duration;
    };

    QByteArray process(const std::vector<F1Frame> &f1FramesIn, bool _padInitialDiscTime,
                       ErrorTreatment _errorTreatment, ConcealType _concealType, bool debugState);
    const Statistics &getStatistics() const;
    void reportStatistics() const;
    void reset();
    void clearStatistics();

private:
    bool debugOn;
    bool padInitialDiscTime;
    Statistics statistics;

    // State-machine variables
    enum StateMachine {
        state_initial,
        state_processFrame,
        state_findEndOfError
    };

    StateMachine currentState;
    StateMachine nextState;
    QByteArray pcmOutputBuffer;
    std::vector<F1Frame> f1FrameBuffer;
    bool waitingForData;
    ErrorTreatment errorTreatment;
    ConcealType concealType;
    bool gotFirstSample;
    bool initialDiscTimeSet;

    F1Frame lastGoodFrame;
    F1Frame nextGoodFrame;
    qint32 errorStartPosition;
    qint32 errorStopPosition;

    StateMachine sm_state_initial();
    StateMachine sm_state_processFrame();
    StateMachine sm_state_findEndOfError();

    // Concealment methods
    void linearInterpolationConceal();
    void predictiveInterpolationConceal();
};

#endif // F1TOAUDIO_H
