/************************************************************************

    tracktime.h

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

#ifndef TRACKTIME_H
#define TRACKTIME_H

#include <QCoreApplication>
#include <QDebug>
#include <QDataStream>

class TrackTime
{
    friend QDataStream &operator<<(QDataStream &, const TrackTime &);
    friend QDataStream &operator>>(QDataStream &, TrackTime &);

public:
    struct Time {
        qint32 minutes;
        qint32 seconds;
        qint32 frames;
    };

    TrackTime(qint32 minutesParam = 0, qint32 secondsParam = 0, qint32 framesParam = 0);
    TrackTime(TrackTime::Time timeParam);

    bool setTime(qint32 minutesParam, qint32 secondsParam, qint32 framesParam);
    bool setTime(TrackTime::Time timeParam);
    void addFrames(qint32 frames);
    void subtractFrames(qint32 frames);
    qint32 getDifference(TrackTime::Time timeToCompare) const;
    Time getTime() const;
    QString getTimeAsQString() const;
    qint32 getFrames() const;

private:
    qint32 trackFrames;
};

#endif // TRACKTIME_H
