/************************************************************************

    sourcevideo.cpp

    ld-decode-tools TBC library
    Copyright (C) 2018-2019 Simon Inns

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

#include "sourcevideo.h"

#include <cstdio>

// Class constructor
SourceVideo::SourceVideo()
{
    // Default object settings
    isSourceVideoOpen = false;
    inputFilePos = -1;
    availableFields = -1;
    fieldLength = -1;
    fieldByteLength = -1;
    fieldLineLength = -1;

    // Set up the cache
    fieldCache.setMaxCost(100);
}

SourceVideo::~SourceVideo()
{
    if (isSourceVideoOpen) inputFile.close();
}

// Source Video file manipulation methods -----------------------------------------------------------------------------

// Open an input video data file. If filename is "-", read from stdin.
// Returns true on success.
bool SourceVideo::open(QString filename, qint32 _fieldLength, qint32 _fieldLineLength)
{
    fieldLength = _fieldLength;
    fieldByteLength = _fieldLength * 2;
    if (_fieldLineLength != -1) {
        fieldLineLength = _fieldLineLength * 2;
    } else fieldLineLength = -1;
    qDebug() << "SourceVideo::open(): Called with field byte length =" << fieldByteLength;

    if (isSourceVideoOpen) {
        // Video file is already open, close it
        qInfo() << "A source video input file is already open, cannot open a new one";
        return false;
    }

    // Open the source video file
    inputFile.setFileName(filename);
    if (filename == "-") {
        if (!inputFile.open(stdin, QIODevice::ReadOnly)) {
            // Failed to open stdin
            qWarning() << "Could not open stdin as source video input file";
            return false;
        }

        // When reading from stdin, we don't know how long the input will be
        availableFields = -1;
    } else {
        if (!inputFile.open(QIODevice::ReadOnly)) {
            // Failed to open named input file
            qWarning() << "Could not open" << filename << "as source video input file";
            return false;
        }

        // File open successful - configure source video parameters
        qint64 tAvailableFields = (inputFile.size() / fieldByteLength);
        availableFields = static_cast<qint32>(tAvailableFields);
        qDebug() << "SourceVideo::open(): Successful -" << availableFields << "fields available";
    }

    // Initialise cache
    fieldCache.clear();

    isSourceVideoOpen = true;
    inputFilePos = 0;

    return true;
}

// Close an input video data file
void SourceVideo::close()
{
    if (!isSourceVideoOpen) {
        qDebug() << "SourceVideo::close(): Called but no source video input file is open";
        return;
    }

    qDebug() << "SourceVideo::close(): Called, closing the source video file and emptying the frame cache";
    inputFile.close();
    isSourceVideoOpen = false;
    inputFilePos = -1;

    qDebug() << "SourceVideo::close(): Source video input file closed";
}

// Get the validity of the source video file
bool SourceVideo::isSourceValid()
{
    return isSourceVideoOpen;
}

// Get the number of fields available from the source video file.
// Returns -1 if the length is unknown (e.g. we're reading from stdin).
qint32 SourceVideo::getNumberOfAvailableFields()
{
    return availableFields;
}

// Get the number of samples in a field
qint32 SourceVideo::getFieldLength()
{
    return fieldLength;
}

// Frame data retrieval methods ---------------------------------------------------------------------------------------

// Method to retrieve a range of field lines from a single video field.
// If startFieldLine and endFieldLine are both -1, read the whole field.
SourceVideo::Data SourceVideo::getVideoField(qint32 fieldNumber, qint32 startFieldLine, qint32 endFieldLine)
{
    // Adjust the field number to index from zero
    fieldNumber--;

    // Ensure source video is open
    if (!isSourceVideoOpen) qFatal("Application requested TBC field before opening TBC file - Fatal error");

    // Calculate the position of the require field line data
    qint64 requiredStartPosition = static_cast<qint64>(fieldByteLength) * static_cast<qint64>(fieldNumber);
    qint64 requiredReadLength;

    if (startFieldLine == -1 && endFieldLine == -1) {
        // Read the whole field

        // Check the cache (we only cache whole fields)
        if (fieldCache.contains(fieldNumber)) {
            return *fieldCache.object(fieldNumber);
        }

        requiredReadLength = static_cast<qint64>(fieldByteLength);
    } else {
        // Read a range of lines

        // Adjust the field line range to index from zero
        startFieldLine--;
        endFieldLine--;

        // Verify the required range
        if (fieldLineLength == -1) qFatal("Application did not set field line length when opening TBC file");
        if (startFieldLine < 0) qFatal("Application requested out-of-bounds field line");

        requiredStartPosition += static_cast<qint64>(fieldLineLength) * static_cast<qint64>(startFieldLine);
        requiredReadLength = static_cast<qint64>(endFieldLine - startFieldLine + 1) * static_cast<qint64>(fieldLineLength);
    }

    // Check the requested field and lines are valid
    if (availableFields != -1
        && (requiredStartPosition < 0
            || requiredStartPosition + requiredReadLength > (static_cast<qint64>(fieldByteLength) * availableFields))) {
        qFatal("Application requested field line range that exceeds the boundaries of the input TBC file");
    }

    // Resize the output buffer
    outputFieldData.resize(static_cast<qint32>(requiredReadLength) / 2);

    // Seek to the correct file position (if not already there)
    if (inputFilePos != requiredStartPosition) {
        if (!inputFile.seek(requiredStartPosition)) {
            // Seek failed

            if (inputFilePos > requiredStartPosition) {
                qFatal("Could not seek backwards to required field position in input TBC file");
            } else {
                // Seeking forwards -- try reading and discarding data instead
                qint64 discardBytes = requiredStartPosition - inputFilePos;
                while (discardBytes > 0) {
                    qint64 readBytes = inputFile.read(reinterpret_cast<char *>(outputFieldData.data()),
                                                      qMin(discardBytes, static_cast<qint64>(outputFieldData.size() * 2)));
                    if (readBytes <= 0) {
                        qFatal("Could not seek or read forwards to required field position in input TBC file");
                    }
                    discardBytes -= readBytes;
                }
            }
        }
        inputFilePos = requiredStartPosition;
    }

    // Read the field lines from the input
    qint64 totalReceivedBytes = 0;
    qint64 receivedBytes = 0;
    do {
        receivedBytes = inputFile.read(reinterpret_cast<char *>(outputFieldData.data()) + totalReceivedBytes,
                                       requiredReadLength - totalReceivedBytes);
        if (receivedBytes > 0) {
            totalReceivedBytes += receivedBytes;
            inputFilePos += receivedBytes;
        }
    } while (receivedBytes > 0 && totalReceivedBytes < requiredReadLength);

    // Verify read was ok
    if (totalReceivedBytes != requiredReadLength) qFatal("Could not read field data from input TBC file");

    if (startFieldLine == -1 && endFieldLine == -1) {
        // Insert the field data into the cache
        fieldCache.insert(fieldNumber, new Data(outputFieldData), 1);
    }

    // Return the data
    return outputFieldData;
}













