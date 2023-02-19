/************************************************************************

    navigation.cpp

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

#include "navigation.h"

#include "vbidecoder.h"

// Construct a NavigationInfo from a disc's metadata
NavigationInfo::NavigationInfo(LdDecodeMetaData &metaData)
{
    const qint32 numFields = metaData.getVideoParameters().numberOfSequentialFields;

    // Scan through the fields in the input, collecting VBI information
    std::vector<Chapter> rawChapters;
    qint32 chapter = -1;
    qint32 firstFieldIndex = 0;
    VbiDecoder vbiDecoder;

    for (qint32 fieldIndex = 0; fieldIndex < numFields; fieldIndex++) {
        // Get the (1-based) field
        const auto& field = metaData.getField(fieldIndex + 1);

        // Codes may be in either field; we want the index of the first
        if (field.isFirstField) {
            firstFieldIndex = fieldIndex;
        }

        // Decode this field's VBI
        const auto vbi = vbiDecoder.decode(field.vbi.vbiData[0], field.vbi.vbiData[1], field.vbi.vbiData[2]);

        if (vbi.chNo != -1 && vbi.chNo != chapter) {
            // Chapter change
            chapter = vbi.chNo;
            rawChapters.emplace_back(Chapter { firstFieldIndex, -1, chapter });
        }

        if (vbi.picStop) {
            // Stop code
            stopCodes.insert(firstFieldIndex);
        }
    }

    // Add a dummy chapter at the end of the input, so we can get the length of
    // the last chapter
    rawChapters.emplace_back(Chapter { numFields, -1, -1 });

    // Because chapter markers have no error detection, a corrupt marker will
    // result in a spurious chapter change. Remove suspiciously short chapters.
    // XXX This could be smarter for sequences like 1 1 1 1 *2 2 3* 2 2 2 2
    for (qint32 i = 0; i < static_cast<qint32>(rawChapters.size() - 1); i++) {
        const auto &chapter = rawChapters[i];
        const auto &nextChapter = rawChapters[i + 1];

        if ((nextChapter.startField - chapter.startField) < 10) {
            // Chapters should be at least 30 tracks (= 60 or more fields) long. So
            // this is too short -- drop it.
            qDebug() << "NavigationInfo::NavigationInfo: Dropped too-short chapter" << chapter.number << "at field" << chapter.startField;
        } else if ((!chapters.empty()) && (chapter.number == chapters.back().number)) {
            // Change to the same chapter - drop
        } else {
            // Keep it
            chapters.emplace_back(chapter);
        }
    }

    // Keep the dummy chapter for now
    chapters.emplace_back(rawChapters.back());

    // Fill in endField for all the chapters we've kept
    for (qint32 i = 0; i < static_cast<qint32>(chapters.size() - 1); i++) {
        chapters[i].endField = chapters[i + 1].startField;
    }

    // And drop the dummy chapter
    chapters.pop_back();
}
