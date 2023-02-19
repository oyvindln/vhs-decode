/************************************************************************

    tracktime.cpp

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

#include "tracktime.h"

// Note: Class for handling A-MIN, A-SEC, A-FRAC and P-MIN, P-SEC, P-FRAC
// time-codes as defined by clauses 22.3.3.5 and 22.3.4.2 of ECMA-130

TrackTime::TrackTime(qint32 minutesParam, qint32 secondsParam, qint32 framesParam)
{
    // Set the default track time
    setTime(minutesParam, secondsParam, framesParam);
}

TrackTime::TrackTime(TrackTime::Time timeParam)
{
    // Set the default track time
    setTime(timeParam);
}

// Set the track time using integer values
bool TrackTime::setTime(qint32 minutesParam, qint32 secondsParam, qint32 framesParam)
{
    // Range check
    if (framesParam > 74 || framesParam < 0) return false;
    if (secondsParam > 59 || secondsParam < 0) return false;
    if (minutesParam > 99 || minutesParam < 0) return false;

    // Set the time
    trackFrames = framesParam + (secondsParam * 75) + (minutesParam * 60 * 75);

    return true;
}

// Set the track time using a time structure
bool TrackTime::setTime(TrackTime::Time timeParam)
{
    // Range check
    if (timeParam.frames > 74 || timeParam.frames < 0) return false;
    if (timeParam.seconds > 59 || timeParam.seconds < 0) return false;
    if (timeParam.minutes > 99 || timeParam.minutes < 0) return false;

    // Set the time
    trackFrames = timeParam.frames + (timeParam.seconds * 75) + (timeParam.minutes * 60 * 75);

    return true;
}

// Method to add frames to the track time
void TrackTime::addFrames(qint32 frames)
{
    trackFrames += frames;
}

// Method to subtract frames from the track time
void TrackTime::subtractFrames(qint32 frames)
{
    trackFrames -= frames;
}

// Method to get the difference (in frames) between two track times
qint32 TrackTime::getDifference(TrackTime::Time timeToCompare) const
{
    qint32 framesToCompare = timeToCompare.frames + (timeToCompare.seconds * 75) + (timeToCompare.minutes * 60 * 75);
    return trackFrames - framesToCompare;
}

// Method to get the track time
TrackTime::Time TrackTime::getTime() const
{
    Time time;

    qint32 remainingFrames = trackFrames;

    time.minutes = remainingFrames / (60 * 75);
    remainingFrames = remainingFrames - (time.minutes * (60 * 75));
    time.seconds = remainingFrames / 75;
    time.frames = remainingFrames - (time.seconds * 75);

    return time;
}

// Method to return the track time as a string
QString TrackTime::getTimeAsQString() const
{
    QString timeString;

    timeString  = QString("%1").arg(getTime().minutes, 2, 10, QChar('0')) + ":";
    timeString += QString("%1").arg(getTime().seconds, 2, 10, QChar('0')) + ".";
    timeString += QString("%1").arg(getTime().frames, 2, 10, QChar('0'));

    return timeString;
}

// Method to return track time in number of frames
qint32 TrackTime::getFrames() const
{
    return trackFrames;
}

// Overloaded operator for writing class data to a data-stream
QDataStream &operator<<(QDataStream &out, const TrackTime &trackTime)
{
    out << trackTime.trackFrames;
    return out;
}

// Overloaded operator for reading class data from a data-stream
QDataStream &operator>>(QDataStream &in, TrackTime &trackTime)
{
    in >> trackTime.trackFrames;
    return in;
}
