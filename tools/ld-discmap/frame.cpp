/************************************************************************

    frame.cpp

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

#include "frame.h"

Frame::Frame(const qint32 seqFrameNumber, const qint32 vbiFrameNumber, const bool isPictureStop,
             const bool isPullDown, const bool isLeadInOrOut, const bool isMarkedForDeletion,
             const double frameQuality, const bool isPadded, const bool isClvOffset,
             const qint32 firstField, const qint32 secondField,
             const qint32 firstFieldPhase, const qint32 secondFieldPhase)
           : m_seqFrameNumber(seqFrameNumber),  m_vbiFrameNumber(vbiFrameNumber), m_isPictureStop(isPictureStop),
             m_isPullDown(isPullDown), m_isLeadInOrOut(isLeadInOrOut), m_isMarkedForDeletion(isMarkedForDeletion),
             m_frameQuality(frameQuality), m_isPadded(isPadded), m_isClvOffset(isClvOffset),
             m_firstField(firstField), m_secondField(secondField),
             m_firstFieldPhase(firstFieldPhase), m_secondFieldPhase(secondFieldPhase)
{
}

// Custom streaming operator (for debug)
QDebug operator<<(QDebug dbg, const Frame &frame)
{
    // Output the object's contents to debug:
    dbg.nospace().noquote() << "Frame(" <<
                               "seqFrameNumber " << frame.seqFrameNumber() <<
                               ", vbiFrameNumber " << frame.vbiFrameNumber() <<
                               ", isPictureStop " << frame.isPictureStop() <<
                               ", isLeadInOrOut " << frame.isLeadInOrOut() <<
                               ", isMarkedForDeletion " << frame.isMarkedForDeletion() <<
                               ", frameQuality " << frame.frameQuality() <<
                               ", isPadded " << frame.isPadded() <<
                               ", isClvOffset " << frame.isClvOffset() <<
                               ", firstField " << frame.firstField() <<
                               ", secondField" << frame.secondField() <<
                               ", firstFieldPhase " << frame.firstFieldPhase() <<
                               ", secondFieldPhase " << frame.secondFieldPhase() <<
                               ")";

    return dbg.maybeSpace();
}

// Get methods
qint32 Frame::seqFrameNumber() const
{
    return m_seqFrameNumber;
}

qint32 Frame::vbiFrameNumber() const
{
    return m_vbiFrameNumber;
}

bool Frame::isPictureStop() const
{
    return m_isPictureStop;
}

bool Frame::isPullDown() const
{
    return m_isPullDown;
}

bool Frame::isLeadInOrOut() const
{
    return m_isLeadInOrOut;
}

bool Frame::isMarkedForDeletion() const
{
    return m_isMarkedForDeletion;
}

double Frame::frameQuality() const
{
    return m_frameQuality;
}

bool Frame::isPadded() const
{
    return m_isPadded;
}

bool Frame::isClvOffset() const
{
    return m_isClvOffset;
}

qint32 Frame::firstField() const
{
    return m_firstField;
}

qint32 Frame::secondField() const
{
    return m_secondField;
}

qint32 Frame::firstFieldPhase() const
{
    return m_firstFieldPhase;
}

qint32 Frame::secondFieldPhase() const
{
    return m_secondFieldPhase;
}


// Set methods
void Frame::seqFrameNumber(qint32 value)
{
    m_seqFrameNumber = value;
}

void Frame::vbiFrameNumber(qint32 value)
{
    m_vbiFrameNumber = value;
}

void Frame::isPictureStop(bool value)
{
    m_isPictureStop = value;
}

void Frame::isPullDown(bool value)
{
    m_isPullDown = value;
}

void Frame::isLeadInOrOut(bool value)
{
    m_isLeadInOrOut = value;
}

void Frame::isMarkedForDeletion(bool value)
{
    m_isMarkedForDeletion = value;
}

void Frame::frameQuality(double value)
{
    m_frameQuality = value;
}

void Frame::isPadded(bool value)
{
    m_isPadded = value;
}

void Frame::isClvOffset(bool value)
{
    m_isClvOffset = value;
}

void Frame::firstField(qint32 value)
{
    m_firstField = value;
}

void Frame::secondField(qint32 value)
{
    m_secondField = value;
}

void Frame::firstFieldPhase(qint32 value)
{
    m_firstFieldPhase = value;
}

void Frame::secondFieldPhase(qint32 value)
{
    m_secondFieldPhase = value;
}

// Overide less than operator for sorting
bool Frame::operator<(const Frame& other) const
{
    return (m_vbiFrameNumber < other.m_vbiFrameNumber) ||
            ((m_vbiFrameNumber == other.m_vbiFrameNumber) && (other.m_isPullDown));
}
