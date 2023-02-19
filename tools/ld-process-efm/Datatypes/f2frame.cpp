/************************************************************************

    f2frame.cpp

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

#include "f2frame.h"

// Note: Class for storing 'F2 frames' as defined by clause 17 of ECMA-130

F2Frame::F2Frame()
{
    for (qint32 i = 0; i < 24; i++) dataSymbols[i] = 0;

    discTime.setTime(0, 0, 0);
    trackTime.setTime(0, 0, 0);
    trackNumber = 0;
}

void F2Frame::setData(const uchar *dataParam, const uchar *erasuresParam)
{
    errorState = false;

    // Add the F2 frame data to the F2 data buffer and swap the byte    
    for (qint32 j = 0; j < 24; j++) {
        dataSymbols[j] = dataParam[j];
        if (erasuresParam[j] != static_cast<uchar>(0)) errorState = true;
    }
}

// This method returns the 24 data symbols for the F2 Frame
const uchar *F2Frame::getDataSymbols() const
{
    return dataSymbols;
}

// This method returns true if the data in the F2 Frame
// is marked with erasures (i.e. it's corrupt)
bool F2Frame::isFrameCorrupt() const
{
    return errorState;
}

// Time markers (not really part of an F2, but used to track the location of the F2 when processing audio)

void F2Frame::setDiscTime(TrackTime _discTime)
{
    discTime = _discTime;
}

void F2Frame::setTrackTime(TrackTime _trackTime)
{
    trackTime = _trackTime;
}

TrackTime F2Frame::getDiscTime() const
{
    return discTime;
}

TrackTime F2Frame::getTrackTime() const
{
    return trackTime;
}

void F2Frame::setTrackNumber(qint32 _trackNumber)
{
    trackNumber = _trackNumber;
}

qint32 F2Frame::getTrackNumber() const
{
    return trackNumber;
}

void F2Frame::setIsEncoderRunning(bool _isEncoderRunning)
{
    isEncoderRunning = _isEncoderRunning;
}

bool F2Frame::getIsEncoderRunning() const
{
    return isEncoderRunning;
}

