/************************************************************************

    closedcaptions.cpp

    ld-export-metadata - Export JSON metadata into other formats
    Copyright (C) 2019-2020 Adam Sampson
    Copyright (C) 2021 Simon Inns

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

#include "closedcaptions.h"

#include "vbidecoder.h"

#include <QtGlobal>
#include <QFile>
#include <QTextStream>
#include <set>
#include <vector>

using std::set;
using std::vector;

// Generate an SCC format timestamp based on the field index
QString generateTimeStamp(qint32 fieldIndex, VideoSystem system)
{
    // Convert to a 0-based count of frames
    double frameIndex = static_cast<double>((fieldIndex - 1) / 2);

    // Set some constants for the timecode calculations.
    // We are generating non-drop timecode (:ff not ;ff), so
    // the clock actually counts at 29.97 FPS.
    const double framesPerSecond = (system == PAL) ? 25.0 : 29.97;
    const double framesPerMinute = framesPerSecond * 60.0;
    const double framesPerHour = framesPerMinute * 60.0;

    // Since the subtitle is relative to the video we
    // can simply calculate the timecode from the sequential
    // field number (which should work even in the input
    // is a snippet from a LaserDisc sample
    //
    // Note: There should probably be the option to choose if the
    // subtitle timecodes are relative to the video or the VBI
    // frame-number/CLV timecode; as both are useful depending on
    // the use-case?
    //
    const qint32 hh = static_cast<qint32>(frameIndex / framesPerHour);
    frameIndex -= static_cast<double>(hh) * framesPerHour;
    const qint32 mm = static_cast<qint32>(frameIndex / framesPerMinute);
    frameIndex -= static_cast<double>(mm) * framesPerMinute;
    const qint32 ss = static_cast<qint32>(frameIndex / framesPerSecond);
    frameIndex -= static_cast<double>(ss) * framesPerSecond;
    const qint32 ff = static_cast<qint32>(frameIndex);

    // Create the timestamp
    return QString("%1:%2:%3:%4").arg(hh, 2, 10, QLatin1Char('0'))
                                 .arg(mm, 2, 10, QLatin1Char('0'))
                                 .arg(ss, 2, 10, QLatin1Char('0'))
                                 .arg(ff, 2, 10, QLatin1Char('0'));
}

// Sanity check the CC data byte and set to -1 if it probably is invalid
qint32 sanityCheckData(qint32 dataByte)
{
    // Already marked as invalid?
    if (dataByte == -1) return -1;

    // Is it in the valid command byte range?
    if (dataByte >= 0x10 && dataByte <= 0x1F) {
        // Valid command byte
        return dataByte;
    }

    // Valid 7-bit ASCII range?
    if (dataByte >= 0x20 && dataByte <= 0x7E) {
        // Valid character byte
        return dataByte;
    }

    // Invalid byte
    return 0;
}

// Extract any available CC data and output it in Scenarist Closed Caption format (SCC) V1.0
// Protocol description:  http://www.theneitherworld.com/mcpoodle/SCC_TOOLS/DOCS/SCC_FORMAT.HTML
bool writeClosedCaptions(LdDecodeMetaData &metaData, const QString &fileName)
{
    const auto videoParameters = metaData.getVideoParameters();

    // Open the output file
    QFile file(fileName);
    if (!file.open(QFile::WriteOnly | QFile::Text)) {
        qDebug("writeClosedCaptions: Could not open file for output");
        return false;
    }
    QTextStream stream(&file);
#if QT_VERSION < QT_VERSION_CHECK(6, 0, 0)
    stream.setCodec("UTF-8");
#endif

    // Output the SCC V1.0 header
    stream << "Scenarist_SCC V1.0";

    // Extract the closed captions data and stream to the text file
    bool captionInProgress = false;
    QString debugCaption;
    for (qint32 fieldIndex = 1; fieldIndex <= videoParameters.numberOfSequentialFields; fieldIndex++) {
        // Get the CC data bytes from the field
        qint32 data0 = sanityCheckData(metaData.getFieldClosedCaption(fieldIndex).data0);
        qint32 data1 = sanityCheckData(metaData.getFieldClosedCaption(fieldIndex).data1);

        // Sometimes random data is passed through; so this sanity check makes sure
        // each new caption starts with data0 = 0x14 which (according to wikipedia)
        // should always be the case
        if (!captionInProgress && data0 > 0) {
            if (data0 != 0x14) {
                data0 = 0;
                data1 = 0;
            }
        }

        // Check incoming data is valid
        if (data0 == -1 || data1 == -1) {
            // Invalid
        } else {
            // Valid
            if (data0 > 0 || data1 > 0) {
                if (captionInProgress == false) {
                    // Start of new caption

                    // Output a timecode followed by a tab character (in SCC format)
                    QString timeStamp = generateTimeStamp(fieldIndex, videoParameters.system);
                    stream << "\n\n" << timeStamp << "\t";
                    debugCaption = "writeClosedCaptions(): Caption data at " + timeStamp + " : [";

                    // Set the caption in progress flag
                    captionInProgress = true;
                }

                // Output the 2 bytes of data as 2 hexadecimal values
                // i.e. 0x12 and 0x41 would be 1241 followed by a space
                // Hex output is padded with leading zeros
                stream << QString("%1").arg(data0, 2, 16, QLatin1Char('0'));
                stream << QString("%1").arg(data1, 2, 16, QLatin1Char('0'));
                stream << " ";

                // Add the 2 bytes of the data output to the debug caption too
                if (data0 >= 0x10 && data0 <= 0x1F) {
                    // This is a command byte, so output a space
                    debugCaption = debugCaption + " ";
                } else {
                    // Normal text - display

                    // Create a string from the two characters
                    char string[3];
                    string[0] = static_cast<char>(data0);
                    string[1] = static_cast<char>(data1);
                    string[2] = static_cast<char>(0);

                    // Add it to the debug output
                    debugCaption = debugCaption + QString::fromLocal8Bit(string);
                }
            } else {
                // No CC data for this frame
                if (captionInProgress) {
                    // End of current caption
                    debugCaption = debugCaption + "]";
                    qDebug() << debugCaption;
                }
                captionInProgress = false;
            }
        }
    }

    // Add some trailing white space
    stream << "\n\n";

    // Done!
    file.close();
    return true;
}
