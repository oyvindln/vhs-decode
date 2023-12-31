/************************************************************************

    stackingpool.cpp

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
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

#include "stackingpool.h"

StackingPool::StackingPool(QString _outputFilename, QString _outputJsonFilename,
                             qint32 _maxThreads, QVector<LdDecodeMetaData *> &_ldDecodeMetaData, QVector<SourceVideo *> &_sourceVideos,
                             bool _reverse, bool _noDiffDod, bool _passThrough, QObject *parent)
    : QObject(parent), outputFilename(_outputFilename), outputJsonFilename(_outputJsonFilename),
      maxThreads(_maxThreads), reverse(_reverse), noDiffDod(_noDiffDod), passThrough(_passThrough),
      abort(false), ldDecodeMetaData(_ldDecodeMetaData), sourceVideos(_sourceVideos)
{
}

bool StackingPool::process()
{
    qInfo() << "Performing final sanity checks...";
    // Open the target video
    targetVideo.setFileName(outputFilename);
    if (outputFilename == "-") {
        if (!targetVideo.open(stdout, QIODevice::WriteOnly)) {
                // Could not open stdout
                qInfo() << "Unable to open stdout";
                return false;
        }
    } else {
        if (!targetVideo.open(QIODevice::WriteOnly)) {
                // Could not open target video file
                qInfo() << "Unable to open output video file";
                return false;
        }
    }

    // If there is a leading field in the TBC which is out of field order, we need to copy it
    // to ensure the JSON metadata files match up
    qInfo() << "Verifying leading fields match...";
    qint32 firstFieldNumber = ldDecodeMetaData[0]->getFirstFieldNumber(1);
    qint32 secondFieldNumber = ldDecodeMetaData[0]->getSecondFieldNumber(1);

    if (firstFieldNumber != 1 && secondFieldNumber != 1) {
        SourceVideo::Data sourceField = sourceVideos[0]->getVideoField(1);
        if (!writeOutputField(sourceField)) {
            // Could not write to target TBC file
            qInfo() << "Writing first field to the output TBC file failed";
            targetVideo.close();
            return false;
        }
    }

    qInfo() << "Scanning source videos for VBI frame number ranges...";
    // Get the VBI frame range for all sources
    if (!setMinAndMaxVbiFrames()) {
        qInfo() << "It was not possible to determine the VBI frame number range for the source video - cannot continue!";
        return false;
    }

    // Show some information for the user
    qInfo() << "Using" << maxThreads << "threads to process" << ldDecodeMetaData[0]->getNumberOfFrames() << "frames";

    // Initialise processing state
    inputFrameNumber = 1;
    outputFrameNumber = 1;
    lastFrameNumber = ldDecodeMetaData[0]->getNumberOfFrames();
    totalTimer.start();

    // Start a vector of decoding threads to process the video
    qInfo() << "Beginning multi-threaded disc stacking process...";
    QVector<QThread *> threads;
    threads.resize(maxThreads);
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i] = new Stacker(abort, *this);
        threads[i]->start(QThread::LowPriority);
    }

    // Wait for the workers to finish
    for (qint32 i = 0; i < maxThreads; i++) {
        threads[i]->wait();
        delete threads[i];
    }

    // Did any of the threads abort?
    if (abort) {
        targetVideo.close();
        return false;
    }

    // Show the processing speed to the user
    double totalSecs = (static_cast<double>(totalTimer.elapsed()) / 1000.0);
    qInfo() << "Disc stacking complete -" << lastFrameNumber << "frames in" << totalSecs << "seconds (" <<
               lastFrameNumber / totalSecs << "FPS )";

    qInfo() << "Creating JSON metadata file for stacked TBC...";
    correctMetaData().write(outputJsonFilename);

    // Close the target video
    targetVideo.close();

    return true;
}

// Get the next frame that needs processing from the input.
//
// Returns true if a frame was returned, false if the end of the input has been
// reached.
bool StackingPool::getInputFrame(qint32& frameNumber,
                                  QVector<qint32>& firstFieldNumber, QVector<SourceVideo::Data>& firstFieldVideoData, QVector<LdDecodeMetaData::Field>& firstFieldMetadata,
                                  QVector<qint32>& secondFieldNumber, QVector<SourceVideo::Data>& secondFieldVideoData, QVector<LdDecodeMetaData::Field>& secondFieldMetadata,
                                  QVector<LdDecodeMetaData::VideoParameters>& videoParameters,
                                  bool& _reverse, bool& _noDiffDod, bool& _passThrough,
                                  QVector<qint32>& availableSourcesForFrame)
{
    QMutexLocker locker(&inputMutex);

    if (inputFrameNumber > lastFrameNumber) {
        // No more input frames
        return false;
    }

    frameNumber = inputFrameNumber;
    inputFrameNumber++;

    // Determine the number of sources available (included padded sources)
    qint32 numberOfSources = sourceVideos.size();

    qDebug().nospace() << "Processing sequential frame number #" <<
                          frameNumber << " from " << numberOfSources << " possible source(s)";

    // Prepare the vectors
    firstFieldNumber.resize(numberOfSources);
    firstFieldVideoData.resize(numberOfSources);
    firstFieldMetadata.resize(numberOfSources);
    secondFieldNumber.resize(numberOfSources);
    secondFieldVideoData.resize(numberOfSources);
    secondFieldMetadata.resize(numberOfSources);
    videoParameters.resize(numberOfSources);

    // Get the current VBI frame number based on the first source
    qint32 currentVbiFrame = -1;
    if (numberOfSources > 1) currentVbiFrame = convertSequentialFrameNumberToVbi(frameNumber, 0);

    for (qint32 sourceNo = 0; sourceNo < numberOfSources; sourceNo++) {
        // Determine the fields for the input frame
        firstFieldNumber[sourceNo] = -1;
        secondFieldNumber[sourceNo] = -1;

        if (sourceNo == 0) {
            // No need to perform VBI frame number mapping on the first source
            firstFieldNumber[sourceNo] = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(frameNumber);
            secondFieldNumber[sourceNo] = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(frameNumber);
            qDebug().nospace() << "Source #0 fields are " <<
                                  firstFieldNumber[sourceNo] << "/" << secondFieldNumber[sourceNo];
        } else if (currentVbiFrame >= sourceMinimumVbiFrame[sourceNo] && currentVbiFrame <= sourceMaximumVbiFrame[sourceNo]) {
            // Use VBI frame number mapping to get the same frame from the
            // current additional source
            qint32 currentSourceFrameNumber = convertVbiFrameNumberToSequential(currentVbiFrame, sourceNo);

            // Check the current source contains the frame
            if (ldDecodeMetaData[sourceNo]->getNumberOfFrames() < currentSourceFrameNumber) {
                firstFieldNumber[sourceNo] = -1;
                secondFieldNumber[sourceNo] = -1;

                qDebug().nospace() << "Source #" << sourceNo << " does not contain VBI frame number " << currentVbiFrame;
            } else {
                firstFieldNumber[sourceNo] = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(currentSourceFrameNumber);
                secondFieldNumber[sourceNo] = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(currentSourceFrameNumber);

                qDebug().nospace() << "Source #" << sourceNo << " has VBI frame number " << currentVbiFrame <<
                            " and fields " << firstFieldNumber[sourceNo] << "/" << secondFieldNumber[sourceNo];
            }
        } else {
            qDebug().nospace() << "Source #" << sourceNo << " does not contain a usable frame";
        }

        // If the field numbers are valid - get the rest of the required data
        if (firstFieldNumber[sourceNo] != -1 && secondFieldNumber[sourceNo] != -1) {
            // Fetch the input data (get the fields in TBC sequence order to save seeking)
            if (firstFieldNumber[sourceNo] < secondFieldNumber[sourceNo]) {
                firstFieldVideoData[sourceNo] = sourceVideos[sourceNo]->getVideoField(firstFieldNumber[sourceNo]);
                secondFieldVideoData[sourceNo] = sourceVideos[sourceNo]->getVideoField(secondFieldNumber[sourceNo]);
            } else {
                secondFieldVideoData[sourceNo] = sourceVideos[sourceNo]->getVideoField(secondFieldNumber[sourceNo]);
                firstFieldVideoData[sourceNo] = sourceVideos[sourceNo]->getVideoField(firstFieldNumber[sourceNo]);
            }

            firstFieldMetadata[sourceNo] = ldDecodeMetaData[sourceNo]->getField(firstFieldNumber[sourceNo]);
            secondFieldMetadata[sourceNo] = ldDecodeMetaData[sourceNo]->getField(secondFieldNumber[sourceNo]);
            videoParameters[sourceNo] = ldDecodeMetaData[sourceNo]->getVideoParameters();
        }
    }

    // Figure out which of the available sources can be used to correct the current frame
    availableSourcesForFrame.clear();
    if (numberOfSources > 1) {
        availableSourcesForFrame = getAvailableSourcesForFrame(currentVbiFrame);
    } else {
        availableSourcesForFrame.append(0);
    }

    // Set the other miscellaneous parameters
    _reverse = reverse;
    _noDiffDod = noDiffDod;
    _passThrough = passThrough;

    return true;
}

// Put a corrected frame into the output stream.
//
// The worker threads will complete frames in an arbitrary order, so we can't
// just write the frames to the output file directly. Instead, we keep a map of
// frames that haven't yet been written; when a new frame comes in, we check
// whether we can now write some of them out.
//
// Returns true on success, false on failure.
bool StackingPool::setOutputFrame(qint32 frameNumber,
                                   SourceVideo::Data firstTargetFieldData, SourceVideo::Data secondTargetFieldData,
                                   qint32 firstFieldSeqNo, qint32 secondFieldSeqNo,
                                   DropOuts firstTargetFieldDropOuts, DropOuts secondTargetFieldDropouts)
{
    QMutexLocker locker(&outputMutex);

    // Put the output frame into the map
    OutputFrame pendingFrame;
    pendingFrame.firstTargetFieldData = firstTargetFieldData;
    pendingFrame.secondTargetFieldData = secondTargetFieldData;
    pendingFrame.firstFieldSeqNo = firstFieldSeqNo;
    pendingFrame.secondFieldSeqNo = secondFieldSeqNo;
    pendingFrame.firstTargetFieldDropOuts = firstTargetFieldDropOuts;
    pendingFrame.secondTargetFieldDropOuts = secondTargetFieldDropouts;

    pendingOutputFrames[frameNumber] = pendingFrame;

    // Write out as many frames as possible
    while (pendingOutputFrames.contains(outputFrameNumber)) {
        const OutputFrame &outputFrame = pendingOutputFrames.value(outputFrameNumber);

        // Save the frame data to the output file (with the fields in the correct order)
        bool writeFail = false;
        if (outputFrame.firstFieldSeqNo < outputFrame.secondFieldSeqNo) {
            // Save the first field and then second field to the output file
            if (!writeOutputField(outputFrame.firstTargetFieldData)) writeFail = true;
            if (!writeOutputField(outputFrame.secondTargetFieldData)) writeFail = true;
        } else {
            // Save the second field and then first field to the output file
            if (!writeOutputField(outputFrame.secondTargetFieldData)) writeFail = true;
            if (!writeOutputField(outputFrame.firstTargetFieldData)) writeFail = true;
        }

        // Was the write successful?
        if (writeFail) {
            // Could not write to target TBC file
            qCritical() << "Writing fields to the output TBC file failed";
            targetVideo.close();
            return false;
        }

        // Clear any existing dropout data
        ldDecodeMetaData[0]->clearFieldDropOuts(outputFrame.firstFieldSeqNo);
        ldDecodeMetaData[0]->clearFieldDropOuts(outputFrame.secondFieldSeqNo);

        // Write the new dropout data into the LdDecodeMetaData output
        ldDecodeMetaData[0]->updateFieldDropOuts(outputFrame.firstTargetFieldDropOuts, outputFrame.firstFieldSeqNo);
        ldDecodeMetaData[0]->updateFieldDropOuts(outputFrame.secondTargetFieldDropOuts, outputFrame.secondFieldSeqNo);

        // Show debug
        qDebug().nospace() << "Processed frame " << outputFrameNumber;

        if (outputFrameNumber % 100 == 0) {
            qInfo() << "Processed and written frame" << outputFrameNumber;
        }

        pendingOutputFrames.remove(outputFrameNumber);
        outputFrameNumber++;
    }

    return true;
}

// Determine the minimum and maximum VBI frame numbers for all sources
// Expects sourceVideos[] and ldDecodeMetaData[] to be populated
// Note: This function returns frame number even if the disc is CLV - conversion
// from timecodes is performed automatically.
bool StackingPool::setMinAndMaxVbiFrames()
{
    // Determine the number of sources available
    qint32 numberOfSources = sourceVideos.size();

    // Resize vectors
    sourceDiscTypeCav.resize(numberOfSources);
    sourceMaximumVbiFrame.resize(numberOfSources);
    sourceMinimumVbiFrame.resize(numberOfSources);

    for (qint32 sourceNumber = 0; sourceNumber < numberOfSources; sourceNumber++) {
        // Determine the disc type and max/min VBI frame numbers
        VbiDecoder vbiDecoder;
        qint32 cavCount = 0;
        qint32 clvCount = 0;
        qint32 cavMin = 1000000;
        qint32 cavMax = 0;
        qint32 clvMin = 1000000;
        qint32 clvMax = 0;

        sourceMinimumVbiFrame[sourceNumber] = 0;
        sourceMaximumVbiFrame[sourceNumber] = 0;
        sourceDiscTypeCav[sourceNumber] = false;

        // Using sequential frame numbering starting from 1
        for (qint32 seqFrame = 1; seqFrame <= ldDecodeMetaData[sourceNumber]->getNumberOfFrames(); seqFrame++) {
            // Get the VBI data and then decode
            auto vbi1 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getFirstFieldNumber(seqFrame)).vbiData;
            auto vbi2 = ldDecodeMetaData[sourceNumber]->getFieldVbi(ldDecodeMetaData[sourceNumber]->getSecondFieldNumber(seqFrame)).vbiData;
            VbiDecoder::Vbi vbi = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2], vbi2[0], vbi2[1], vbi2[2]);

            // Look for a complete, valid CAV picture number or CLV time-code
            if (vbi.picNo > 0) {
                cavCount++;

                if (vbi.picNo < cavMin) cavMin = vbi.picNo;
                if (vbi.picNo > cavMax) cavMax = vbi.picNo;
            }

            if (vbi.clvHr != -1 && vbi.clvMin != -1 &&
                    vbi.clvSec != -1 && vbi.clvPicNo != -1) {
                clvCount++;

                LdDecodeMetaData::ClvTimecode timecode;
                timecode.hours = vbi.clvHr;
                timecode.minutes = vbi.clvMin;
                timecode.seconds = vbi.clvSec;
                timecode.pictureNumber = vbi.clvPicNo;
                qint32 cvFrameNumber = ldDecodeMetaData[sourceNumber]->convertClvTimecodeToFrameNumber(timecode);

                if (cvFrameNumber < clvMin) clvMin = cvFrameNumber;
                if (cvFrameNumber > clvMax) clvMax = cvFrameNumber;
            }
        }
        qDebug() << "StackingPool::setMinAndMaxVbiFrames(): Got" << cavCount << "CAV picture codes and" << clvCount << "CLV timecodes";

        // If the metadata has no picture numbers or time-codes, we cannot use the source
        if (cavCount == 0 && clvCount == 0) {
            qDebug() << "StackingPool::setMinAndMaxVbiFrames(): Source does not seem to contain valid CAV picture numbers or CLV time-codes - cannot process";
            return false;
        }

        // Determine disc type
        if (cavCount > clvCount) {
            sourceDiscTypeCav[sourceNumber] = true;
            qDebug() << "StackingPool::setMinAndMaxVbiFrames(): Got" << cavCount << "valid CAV picture numbers - source disc type is CAV";
            qInfo().nospace() << "Source #" << sourceNumber << " has a disc type of CAV (uses VBI frame numbers)";

            sourceMaximumVbiFrame[sourceNumber] = cavMax;
            sourceMinimumVbiFrame[sourceNumber] = cavMin;
        } else {
            sourceDiscTypeCav[sourceNumber] = false;
            qDebug() << "StackingPool::setMinAndMaxVbiFrames(): Got" << clvCount << "valid CLV picture numbers - source disc type is CLV";
            qInfo().nospace() << "Source #" << sourceNumber << " has a disc type of CLV (uses VBI time codes)";

            sourceMaximumVbiFrame[sourceNumber] = clvMax;
            sourceMinimumVbiFrame[sourceNumber] = clvMin;
        }

        qInfo().nospace() << "Source #" << sourceNumber << " has a VBI frame number range of " << sourceMinimumVbiFrame[sourceNumber] << " to " <<
            sourceMaximumVbiFrame[sourceNumber];
    }

    return true;
}

// Method to convert the first source sequential frame number to a VBI frame number
qint32 StackingPool::convertSequentialFrameNumberToVbi(qint32 sequentialFrameNumber, qint32 sourceNumber)
{
    return (sourceMinimumVbiFrame[sourceNumber] - 1) + sequentialFrameNumber;
}

// Method to convert a VBI frame number to a sequential frame number
qint32 StackingPool::convertVbiFrameNumberToSequential(qint32 vbiFrameNumber, qint32 sourceNumber)
{
    // Offset the VBI frame number to get the sequential source frame number
    return vbiFrameNumber - sourceMinimumVbiFrame[sourceNumber] + 1;
}

// Method that returns a vector of the sources that contain data for the required VBI frame number
QVector<qint32> StackingPool::getAvailableSourcesForFrame(qint32 vbiFrameNumber)
{
    QVector<qint32> availableSourcesForFrame;
    for (qint32 sourceNo = 0; sourceNo < sourceVideos.size(); sourceNo++) {
        if (vbiFrameNumber >= sourceMinimumVbiFrame[sourceNo] && vbiFrameNumber <= sourceMaximumVbiFrame[sourceNo]) {
            // Get the field numbers for the frame - THIS CRASHES
            qint32 sequentialFrameNumber = convertVbiFrameNumberToSequential(vbiFrameNumber, sourceNo);

            // Check the source contains enough frames to have the required sequential frame
            if (ldDecodeMetaData[sourceNo]->getNumberOfFrames() < sequentialFrameNumber)
            {
                // Sequential frame is out of bounds
                qDebug() << "VBI Frame number" << vbiFrameNumber << "is out of bounds for source " << sourceNo;
            } else {
                // Sequential frame is in bounds
                qint32 firstFieldNumber = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(sequentialFrameNumber);
                qint32 secondFieldNumber = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(sequentialFrameNumber);

                // Ensure the frame is not a padded field (i.e. missing)
                if (ldDecodeMetaData[sourceNo]->getField(firstFieldNumber).pad == false && ldDecodeMetaData[sourceNo]->getField(secondFieldNumber).pad == false) {
                    availableSourcesForFrame.append(sourceNo);
                } else {
                    if (ldDecodeMetaData[sourceNo]->getField(firstFieldNumber).pad == true) qDebug() << "First field number" << firstFieldNumber << "of source" << sourceNo << "is padded";
                    if (ldDecodeMetaData[sourceNo]->getField(secondFieldNumber).pad == true) qDebug() << "Second field number" << firstFieldNumber << "of source" << sourceNo << "is padded";
                }
            }
        }
    }

    if (availableSourcesForFrame.size() != sourceVideos.size()) {
        if (availableSourcesForFrame.size() > 0) {
            qDebug() << "VBI Frame number" << vbiFrameNumber << "has only" << availableSourcesForFrame.size() << "available sources";
        } else {
            qInfo() << "Warning: VBI Frame number" << vbiFrameNumber << "has ZERO available sources (all sources padded?)";
        }
    }

    return availableSourcesForFrame;
}

// Write a field to the output file.
// Returns true on success, false on failure.
bool StackingPool::writeOutputField(const SourceVideo::Data &fieldData)
{
    return targetVideo.write(reinterpret_cast<const char *>(fieldData.data()), 2 * fieldData.size());
}

void StackingPool::correctPhaseIDs()
{
    constexpr qint32 PHASE_COUNT = 4;

    const qint32 fieldCount = ldDecodeMetaData[0]->getNumberOfFields();

    // Find the first non-padded field
    qint32 pivotField = 1;
    while (pivotField <= fieldCount && ldDecodeMetaData[0]->getField(pivotField).pad) {
        ++pivotField;
    }
    if (pivotField >= fieldCount)
    {
        return;
    }

    // Get the starting phase ID - 1
    qint32 currentPhaseID = ldDecodeMetaData[0]->getField(pivotField).fieldPhaseID - 1;
    currentPhaseID -= (pivotField - 1) % PHASE_COUNT;
    currentPhaseID += PHASE_COUNT;
    currentPhaseID %= PHASE_COUNT;

    // Overwrite phase IDs
    for (qint32 fieldNumber = 1; fieldNumber <= fieldCount; ++fieldNumber)
    {
        LdDecodeMetaData::Field field = ldDecodeMetaData[0]->getField(fieldNumber);
        field.fieldPhaseID = currentPhaseID + 1;
        ldDecodeMetaData[0]->updateField(field, fieldNumber);
        ++currentPhaseID;
        currentPhaseID %= PHASE_COUNT;
    }
}

template<int field>
void StackingPool::replaceFieldMetaData(qint32 frameNumber)
{
    const qint32 currentVbiFrame = convertSequentialFrameNumberToVbi(frameNumber, 0);

    qint32 fieldNumber = 0;
    if constexpr (field == 1) {
        fieldNumber = ldDecodeMetaData[0]->getFirstFieldNumber(frameNumber);
    } else {
        fieldNumber = ldDecodeMetaData[0]->getSecondFieldNumber(frameNumber);
    }

    const LdDecodeMetaData::Field &currentField = ldDecodeMetaData[0]->getField(fieldNumber);
    if (currentField.pad) {
        for (int sourceNo = 1; sourceNo < ldDecodeMetaData.size(); ++sourceNo) {
            if (currentVbiFrame < sourceMinimumVbiFrame[sourceNo] || currentVbiFrame > sourceMaximumVbiFrame[sourceNo]) {
                continue;
            }
            const qint32 currentSourceFrameNumber = convertVbiFrameNumberToSequential(currentVbiFrame, sourceNo);
            if (ldDecodeMetaData[sourceNo]->getNumberOfFrames() < currentSourceFrameNumber) {
                continue;
            }
            qint32 otherFieldNumber = 0;
            if constexpr (field == 1) {
                otherFieldNumber = ldDecodeMetaData[sourceNo]->getFirstFieldNumber(frameNumber);
            } else {
                otherFieldNumber = ldDecodeMetaData[sourceNo]->getSecondFieldNumber(frameNumber);
            }
            if (ldDecodeMetaData[sourceNo]->getField(otherFieldNumber).pad) {
                continue;
            }
            LdDecodeMetaData::Field potentialField = ldDecodeMetaData[sourceNo]->getField(otherFieldNumber);
            potentialField.seqNo = currentField.seqNo;
            potentialField.fieldPhaseID = currentField.fieldPhaseID;
            potentialField.dropOuts = currentField.dropOuts;
            ldDecodeMetaData[0]->updateField(potentialField, fieldNumber);
            break;
        }
    }
}

LdDecodeMetaData &StackingPool::correctMetaData()
{
    correctPhaseIDs();
    const qint32 frameCount = ldDecodeMetaData[0]->getNumberOfFrames();
    for (qint32 frameNumber = 1; frameNumber <= frameCount; ++frameNumber) {
        replaceFieldMetaData<1>(frameNumber);
        replaceFieldMetaData<2>(frameNumber);
    }
    return *ldDecodeMetaData[0];
}
