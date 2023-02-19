/************************************************************************

    ffmetadata.cpp

    ld-export-metadata - Export JSON metadata into other formats
    Copyright (C) 2019-2023 Adam Sampson

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

#include "navigation.h"

#include <QtGlobal>
#include <QFile>
#include <QTextStream>

bool writeFfmetadata(LdDecodeMetaData &metaData, const QString &fileName)
{
    const auto videoParameters = metaData.getVideoParameters();

    // Select the appropriate timebase to make 0-based field numbers work
    const QString timeBase = videoParameters.system == PAL ? "1/50" : "1001/60000";

    // Extract navigation information
    const NavigationInfo navInfo(metaData);

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

    // Write the chapter changes
    for (const auto &chapter: navInfo.chapters) {
        stream << "\n";
        stream << "[CHAPTER]\n";
        stream << "TIMEBASE=" << timeBase << "\n";
        stream << "START=" << chapter.startField << "\n";
        stream << "END=" << chapter.endField - 1 << "\n";
        stream << "title=" << QString("Chapter %1").arg(chapter.number) << "\n";
    }

    if (!navInfo.stopCodes.empty()) {
        // Write the stop codes, as comments
        // XXX Is there a way to represent these properly?
        stream << "\n";
        for (qint32 field: navInfo.stopCodes) {
            stream << "; Stop code at " << field << "\n";
        }
    }

    // Done!
    file.close();
    return true;
}
