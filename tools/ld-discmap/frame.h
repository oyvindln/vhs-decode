/************************************************************************

    frame.h

    ld-discmap - TBC and VBI alignment and correction
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-discmap is free software: you can redistribute it and/or
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

#ifndef FRAME_H
#define FRAME_H

#include <QCoreApplication>
#include <QDebug>

class Frame
{
public:
    Frame(const qint32 seqFrameNumber = -1, const qint32 vbiFrameNumber = -1, const bool isPictureStop = false,
          const bool isPullDown = false, const bool isLeadInOrOut = false, const bool isMarkedForDeletion = false,
          const double frameQuality = 0, const bool isPadded = false, const bool isClvOffset = false,
          const qint32 firstField = -1, const qint32 secondField = -1,
          const qint32 firstFieldPhase = -1, const qint32 secondFieldPhase = -1);
    ~Frame() = default;
    Frame(const Frame &) = default;
    Frame &operator=(const Frame &) = default;

    // Get
    qint32 seqFrameNumber() const;
    qint32 vbiFrameNumber() const;
    bool isPictureStop() const;
    bool isPullDown() const;
    bool isLeadInOrOut() const;
    bool isMarkedForDeletion() const;
    double frameQuality() const;
    bool isPadded() const;
    bool isClvOffset() const;
    qint32 firstField() const;
    qint32 secondField() const;
    qint32 firstFieldPhase() const;
    qint32 secondFieldPhase() const;

    // Set
    void seqFrameNumber(qint32 value);
    void vbiFrameNumber(qint32 value);
    void isPictureStop(bool value);
    void isPullDown(bool value);
    void isLeadInOrOut(bool value);
    void isMarkedForDeletion(bool value);
    void frameQuality(double value);
    void isPadded(bool value);
    void isClvOffset(bool value);
    void firstField(qint32 value);
    void secondField(qint32 value);
    void firstFieldPhase(qint32 value);
    void secondFieldPhase(qint32 value);

    // Operators
    bool operator<(const Frame &) const;

private:
    qint32 m_seqFrameNumber;
    qint32 m_vbiFrameNumber;
    bool m_isPictureStop;
    bool m_isPullDown;
    bool m_isLeadInOrOut;
    bool m_isMarkedForDeletion;
    double m_frameQuality;
    bool m_isPadded;
    bool m_isClvOffset;
    qint32 m_firstField;
    qint32 m_secondField;
    qint32 m_firstFieldPhase;
    qint32 m_secondFieldPhase;
};

// Custom streaming operator for debug
QDebug operator<<(QDebug dbg, const Frame &frame);

// Custom meta-type declaration
Q_DECLARE_METATYPE(Frame)

#endif // FRAME_H
