/************************************************************************

    audacity.cpp

    ld-export-metadata - Export JSON metadata into other formats
    Copyright (C) 2023 Adam Sampson

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

#include "audacity.h"

#include "navigation.h"

#include <QtGlobal>
#include <QFile>
#include <QTextStream>

bool writeAudacityLabels(LdDecodeMetaData &metaData, const QString &fileName)
{
    const auto videoParameters = metaData.getVideoParameters();

    // Positions are given in seconds, with exclusive ranges.
    // Select a scale factor to convert from 0-based field numbers to seconds.
    const double timeFactor = videoParameters.system == PAL ? (1.0 / 50.0) : (1001.0 / 60000.0);

    // Extract navigation information
    const NavigationInfo navInfo(metaData);

    // Open the output file
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug("writeAudacityLabels: Could not open file for output");
        return false;
    }
    QTextStream stream(&file);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    stream.setCodec("UTF-8");
#endif

    // Write the chapter changes
    for (const auto &chapter: navInfo.chapters) {
        stream << QString("%1\t%2\tChapter %3\n")
                      .arg(static_cast<double>(chapter.startField) * timeFactor, 0, 'f')
                      .arg(static_cast<double>(chapter.endField) * timeFactor, 0, 'f')
                      .arg(chapter.number);
    }

    // Write the stop codes
    for (qint32 field: navInfo.stopCodes) {
        stream << QString("%1\t%2\tStop code\n")
                      .arg(static_cast<double>(field) * timeFactor, 0, 'f')
                      .arg(static_cast<double>(field) * timeFactor, 0, 'f');
    }

    // Done!
    file.close();
    return true;
}
