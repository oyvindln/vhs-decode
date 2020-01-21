/************************************************************************

    discmap.h

    ld-discmap - TBC and VBI alignment and correction
    Copyright (C) 2019-2020 Simon Inns

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

#ifndef DISCMAP_H
#define DISCMAP_H

#include <QCoreApplication>
#include <QDebug>

// TBC library includes
#include "lddecodemetadata.h"
#include "vbidecoder.h"

#include "frame.h"

class DiscMap
{
public:
    DiscMap() = default;
    ~DiscMap() = default;
    DiscMap(const DiscMap &) = default;
    DiscMap &operator=(const DiscMap &) = default;

    DiscMap(const QFileInfo &metadataFileInfo, const bool &reverseFieldOrder);

    QString filename() const;
    bool valid() const;
    qint32 numberOfFrames() const;
    bool isDiscCav() const;
    bool isDiscPal() const;
    QString discType() const;
    QString discFormat() const;

    qint32 vbiFrameNumber(qint32 frameNumber) const;
    void setVbiFrameNumber(qint32 frameNumber, qint32 vbiFrameNumber);
    qint32 seqFrameNumber(qint32 frameNumber) const;
    bool isPulldown(qint32 frameNumber) const;
    qint32 numberOfPulldowns() const;
    bool isPictureStop(qint32 frameNumber) const;
    bool isLeadInOut(qint32 frameNumber) const;
    qreal frameQuality(qint32 frameNumber) const;
    bool isPadded(qint32 frameNumber) const;

    void setMarkedForDeletion(qint32 frameNumber);
    qint32 flush();
    void sort();

private:
    // Miscellaneous
    QFileInfo m_metadataFileInfo;
    bool m_reverseFieldOrder;
    bool m_tbcValid;
    qint32 m_numberOfFrames;
    bool m_isDiscPal;
    bool m_isDiscCav;
    qint32 m_numberOfPulldowns;

    QVector<Frame> m_frames;
};

// Custom streaming operator for debug
QDebug operator<<(QDebug dbg, const DiscMap &tbcInformation);

// Custom meta-type declaration
Q_DECLARE_METATYPE(DiscMap);

#endif // DISCMAP_H
