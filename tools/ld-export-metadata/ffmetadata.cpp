/************************************************************************

    ffmetadata.cpp

    ld-export-metadata - Export JSON metadata into other formats
    Copyright (C) 2019-2020 Adam Sampson

    This file is part of ld-decode-tools.

    ld-export-metadata is free software: you can redistribute it and/or
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

#include "ffmetadata.h"

#include "vbidecoder.h"

#include <QtGlobal>
#include <QFile>
#include <QTextStream>
#include <set>
#include <vector>

using std::set;
using std::vector;

struct ChapterChange {
    qint32 field;
    qint32 chapter;
};

bool writeFfmetadata(LdDecodeMetaData &metaData, const QString &fileName)
{
    const auto videoParameters = metaData.getVideoParameters();

    // We'll give positions using 0-based field indexes directly, rather than
    // using the times encoded in the VBI, because we might be working with a
    // capture of only part of a disc. Select the appropriate timebase to make
    // this work.
    const QString timeBase = videoParameters.isSourcePal ? "1/50" : "1001/60000";

    // Scan through the fields in the input, collecting VBI information
    vector<ChapterChange> chapterChanges;
    set<qint32> stopCodes;
    qint32 chapter = -1;
    qint32 firstFieldIndex = 0;
    VbiDecoder vbiDecoder;

    for (qint32 fieldIndex = 0; fieldIndex < videoParameters.numberOfSequentialFields; fieldIndex++) {
        // Get the (1-based) field
        const auto field = metaData.getField(fieldIndex + 1);

        // Codes may be in either field; we want the index of the first
        if (field.isFirstField) {
            firstFieldIndex = fieldIndex;
        }

        // Decode this field's VBI
        const auto vbi = vbiDecoder.decode(field.vbi.vbiData[0], field.vbi.vbiData[1], field.vbi.vbiData[2]);

        if (vbi.chNo != -1 && vbi.chNo != chapter) {
            // Chapter change
            chapter = vbi.chNo;
            chapterChanges.emplace_back(ChapterChange {firstFieldIndex, chapter});
        }

        if (vbi.picStop) {
            // Stop code
            stopCodes.insert(firstFieldIndex);
        }
    }

    // Add a dummy change at the end of the input, so we can get the length of
    // the last chapter
    chapterChanges.emplace_back(ChapterChange {videoParameters.numberOfSequentialFields, -1});

    // Because chapter markers have no error detection, a corrupt marker will
    // result in a spurious chapter change. Remove suspiciously short chapters.
    // XXX This could be smarter for sequences like 1 1 1 1 *2 2 3* 2 2 2 2
    vector<ChapterChange> cleanChanges;
    for (qint32 i = 0; i < static_cast<qint32>(chapterChanges.size() - 1); i++) {
        const auto &change = chapterChanges[i];
        const auto &nextChange = chapterChanges[i + 1];

        if ((nextChange.field - change.field) < 10) {
            // Chapters should be at least 30 tracks (= 60 or more fields) long. So
            // this is too short -- drop it.
        } else if ((!cleanChanges.empty()) && (change.chapter == cleanChanges.back().chapter)) {
            // Change to the same chapter - drop
        } else {
            // Keep
            cleanChanges.emplace_back(change);
        }
    }

    // Keep the dummy change
    cleanChanges.emplace_back(chapterChanges.back());

    // Open the output file
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug("writeFfmetadata: Could not open file for output");
        return false;
    }
    QTextStream stream(&file);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    stream.setCodec("UTF-8");
#endif

    // Write the header
    stream << ";FFMETADATA1\n";

    // Write the chapter changes, skipping the dummy one at the end
    for (qint32 i = 0; i < static_cast<qint32>(cleanChanges.size() - 1); i++) {
        const auto &change = cleanChanges[i];
        const auto &nextChange = cleanChanges[i + 1];

        stream << "\n";
        stream << "[CHAPTER]\n";
        stream << "TIMEBASE=" << timeBase << "\n";
        stream << "START=" << change.field << "\n";
        stream << "END=" << (nextChange.field - 1) << "\n";
        stream << "title=" << QString("Chapter %1").arg(change.chapter) << "\n";
    }

    if (!stopCodes.empty()) {
        // Write the stop codes, as comments
        // XXX Is there a way to represent these properly?
        stream << "\n";
        for (qint32 field : stopCodes) {
            stream << "; Stop code at " << field << "\n";
        }
    }

    // Done!
    file.close();
    return true;
}
