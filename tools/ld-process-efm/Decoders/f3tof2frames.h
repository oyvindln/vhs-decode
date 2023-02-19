/************************************************************************

    f3tof2frames.cpp

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

#ifndef F3TOF2FRAMES_H
#define F3TOF2FRAMES_H

#include <QCoreApplication>
#include <QDebug>
#include <vector>

#include "Datatypes/f3frame.h"
#include "Datatypes/f2frame.h"
#include "Datatypes/section.h"

#include "c1circ.h"
#include "c2circ.h"
#include "c2deinterleave.h"

class F3ToF2Frames
{
public:
    F3ToF2Frames();

    // Statistics
    struct Statistics {
        qint32 totalF3Frames;
        qint32 totalF2Frames;

        C1Circ::Statistics c1Circ_statistics;
        C2Circ::Statistics c2Circ_statistics;
        C2Deinterleave::Statistics c2Deinterleave_statistics;

        TrackTime initialDiscTime;
        TrackTime currentDiscTime;

        qint32 sequenceInterruptions;
        qint32 missingF3Frames;

        qint32 preempFrames;
    };

    const std::vector<F2Frame> &process(const std::vector<F3Frame> &f3FramesIn, bool debugState, bool noTimeStamp);
    const Statistics &getStatistics();
    void reportStatistics() const;
    void reset();

private:
    bool debugOn;
    Statistics statistics;

    void clearStatistics();

    C1Circ c1Circ;
    C2Circ c2Circ;
    C2Deinterleave c2Deinterleave;

    std::vector<F2Frame> f2FrameBuffer;
    std::vector<F2Frame> f2FramesOut;
    std::vector<Section> sectionBuffer;
    std::vector<TrackTime> sectionDiscTimes;

    bool initialDiscTimeSet;
    TrackTime lastDiscTime;
    bool lostSections;
};

#endif // F3TOF2FRAMES_H
