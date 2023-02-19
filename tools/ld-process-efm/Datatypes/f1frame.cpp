/************************************************************************

    f1frame.cpp

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

#include "f1frame.h"

F1Frame::F1Frame()
{
    for (qint32 i = 0; i < 24; i++) dataSymbols[i] = 0;

    isCorruptFlag = true;
    isEncoderOnFlag = false;
    isMissingFlag = true;
    discTime.setTime(0, 0, 0);
    trackTime.setTime(0, 0, 0);
    trackNumber = 0;
}

void F1Frame::setData(const uchar *dataParam, bool _isCorrupt, bool _isEncoderOn, bool _isMissing,
                      TrackTime _discTime, TrackTime _trackTime, qint32 _trackNumber)
{
    // Add the F1 frame data to the F1 data buffer and swap the byte
    // order (see ECMA-130 clause 16)
    for (qint32 j = 0; j < 24; j += 2) {
        dataSymbols[j] = dataParam[j+1];
        dataSymbols[j+1] = dataParam[j];
    }

    isCorruptFlag = _isCorrupt;
    isEncoderOnFlag = _isEncoderOn;
    isMissingFlag = _isMissing;

    discTime = _discTime;
    trackTime = _trackTime;
    trackNumber = _trackNumber;

    // Note: According the ECMA-130 audio data doesn't require byte swapping
    // however, since the required PCM sample format is little-endian (on a PC)
    // it is required.  Therefore we do it to the F1 frame data to save having
    // to perform the swapping twice (in the audio and data processing)
}

const uchar *F1Frame::getDataSymbols() const
{
    return dataSymbols;
}

bool F1Frame::isCorrupt() const
{
    return isCorruptFlag;
}

bool F1Frame::isEncoderOn() const
{
    return isEncoderOnFlag;
}

bool F1Frame::isMissing() const
{
    return isMissingFlag;
}

TrackTime F1Frame::getDiscTime() const
{
    return discTime;
}

TrackTime F1Frame::getTrackTime() const
{
    return trackTime;
}

qint32 F1Frame::getTrackNumber() const
{
    return trackNumber;
}
