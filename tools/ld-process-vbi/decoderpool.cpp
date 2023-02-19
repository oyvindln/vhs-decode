/************************************************************************

    decoderpool.cpp

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-vbi is free software: you can redistribute it and/or
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

#include "decoderpool.h"

DecoderPool::DecoderPool(QString _inputFilename, QString _outputJsonFilename,
                         qint32 _maxThreads, LdDecodeMetaData &_ldDecodeMetaData)
    : inputFilename(_inputFilename), outputJsonFilename(_outputJsonFilename),
      maxThreads(_maxThreads), ldDecodeMetaData(_ldDecodeMetaData)
{
}

bool DecoderPool::process()
{
    // Get the metadata for the video parameters
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();
    qInfo().noquote() << "Input TBC source dimensions are" << videoParameters.fieldWidth << "x" <<
                videoParameters.fieldHeight;

    // Open the source video
    if (!sourceVideo.open(inputFilename, videoParameters.fieldWidth * videoParameters.fieldHeight, videoParameters.fieldWidth)) {
        // Could not open source video file
        qCritical() << "Source TBC file could not be opened";
        return false;
    }

    // Check TBC and JSON field numbers match
    if (sourceVideo.getNumberOfAvailableFields() != ldDecodeMetaData.getNumberOfFields()) {
        qWarning() << "Warning: TBC file contains" << sourceVideo.getNumberOfAvailableFields() <<
                   "fields but the JSON indicates" << ldDecodeMetaData.getNumberOfFields() <<
                   "fields - some fields will be ignored";
    }

    // Show some information for the user
    qInfo() << "Using" << maxThreads << "threads to process" << ldDecodeMetaData.getNumberOfFields() << "fields";

    // Initialise processing state
    inputFieldNumber = 1;
    lastFieldNumber = ldDecodeMetaData.getNumberOfFields();
    totalTimer.start();

    // Start a vector of decoding threads to process the video
    QVector<QThread *> threads;
    threads.resize(maxThreads);
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i] = new VbiLineDecoder(abort, *this);
        threads[i]->start(QThread::LowPriority);
    }

    // Wait for the workers to finish
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i]->wait();
        delete threads[i];
    }

    // Did any of the threads abort?
    if (abort) {
        sourceVideo.close();
        return false;
    }

    // Show the processing speed to the user
    double totalSecs = (static_cast<double>(totalTimer.elapsed()) / 1000.0);
    qInfo() << "VBI Processing complete -" << lastFieldNumber << "fields in" << totalSecs << "seconds (" <<
               lastFieldNumber / totalSecs << "FPS )";

    // Write the JSON metadata file
    qInfo() << "Writing JSON metadata file...";
    ldDecodeMetaData.write(outputJsonFilename);
    qInfo() << "VBI processing complete";

    // Close the source video
    sourceVideo.close();

    return true;
}

// Get the next field that needs processing from the input.
//
// Returns true if a field was returned, false if the end of the input has been
// reached.
bool DecoderPool::getInputField(qint32 &fieldNumber, SourceVideo::Data &fieldVideoData,
                                LdDecodeMetaData::Field &fieldMetadata, LdDecodeMetaData::VideoParameters &videoParameters)
{
    QMutexLocker locker(&inputMutex);

    if (inputFieldNumber > lastFieldNumber) {
        // No more input fields
        return false;
    }

    fieldNumber = inputFieldNumber;
    inputFieldNumber++;

    // Show what we are about to process
    qDebug() << "DecoderPool::process(): Processing field number" << fieldNumber;

    // Fetch the input data
    fieldVideoData = sourceVideo.getVideoField(fieldNumber, VbiLineDecoder::startFieldLine, VbiLineDecoder::endFieldLine);
    fieldMetadata = ldDecodeMetaData.getField(fieldNumber);
    videoParameters = ldDecodeMetaData.getVideoParameters();

    return true;
}

// Put a decoded frame into the output stream.
//
// Returns true on success, false on failure.
bool DecoderPool::setOutputField(qint32 fieldNumber, const LdDecodeMetaData::Field& fieldMetadata)
{
    QMutexLocker locker(&outputMutex);

    // Save the field data to the metadata (only some metadata is affected)
    ldDecodeMetaData.updateFieldVbi(fieldMetadata.vbi, fieldNumber);
    ldDecodeMetaData.updateFieldNtsc(fieldMetadata.ntsc, fieldNumber);
    ldDecodeMetaData.updateFieldVitc(fieldMetadata.vitc, fieldNumber);
    ldDecodeMetaData.updateFieldClosedCaption(fieldMetadata.closedCaption, fieldNumber);

    return true;
}


