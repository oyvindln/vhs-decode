/************************************************************************

    discmapper.h

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

#ifndef DISCMAPPER_H
#define DISCMAPPER_H

#include <QCoreApplication>
#include <QDebug>
#include <QFileInfo>
#include <QFile>

// TBC library includes
#include "sourcevideo.h"
#include "sourceaudio.h"
#include "lddecodemetadata.h"

#include "discmap.h"

class DiscMapper
{
public:
    DiscMapper();

    bool process(QFileInfo _inputFileInfo, QFileInfo _inputMetadataFileInfo,
                 QFileInfo _outputFileInfo, bool _reverse, bool _mapOnly, bool _noStrict,
                 bool _deleteUnmappable, bool _noAudio);

private:
    QFileInfo inputFileInfo;
    QFileInfo inputMetadataFileInfo;
    QFileInfo outputFileInfo;
    bool reverse;
    bool mapOnly;
    bool noStrict;
    bool deleteUnmappable;
    bool noAudio;

    void removeLeadInOut(DiscMap &discMap);
    void removeInvalidFramesByPhase(DiscMap &discMap);
    void correctVbiFrameNumbersUsingSequenceAnalysis(DiscMap &discMap);
    void removeDuplicateNumberedFrames(DiscMap &discMap);
    void numberPulldownFrames(DiscMap &discMap);
    bool verifyFrameNumberPresence(DiscMap &discMap);
    void reorderFrames(DiscMap &discMap);
    void padDiscMap(DiscMap &discMap);
    void rewriteFrameNumbers(DiscMap &discMap);
    void deleteUnmappableFrames(DiscMap &discMap);

    bool saveDiscMap(DiscMap &discMap);
};

#endif // DISCMAPPER_H
