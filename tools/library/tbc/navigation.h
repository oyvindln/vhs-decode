/************************************************************************

    navigation.h

    ld-decode-tools TBC library
    Copyright (C) 2019-2023 Adam Sampson

    This file is part of ld-decode-tools.

    ld-decode-tools is free software: you can redistribute it and/or
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

#ifndef NAVIGATION_H
#define NAVIGATION_H

#include "lddecodemetadata.h"

#include <QtGlobal>
#include <set>
#include <vector>

// Navigation information extracted from LaserDisc metadata.
// Positions are given in 0-based fields, relative to the start of the TBC file
// (in case we're dealing with a clip from the middle of a disc).
struct NavigationInfo {
    NavigationInfo(LdDecodeMetaData &metaData);

    struct Chapter {
        // First field number
        qint32 startField;
        // Last field number (exclusive, i.e. first field of next chapter)
        qint32 endField;
        // Chapter number
        qint32 number;
    };

    // Field numbers containing stop codes
    std::set<qint32> stopCodes;
    // Chapters
    std::vector<Chapter> chapters;
};

#endif
