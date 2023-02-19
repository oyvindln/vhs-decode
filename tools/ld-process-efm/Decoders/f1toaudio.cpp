/************************************************************************

    f1toaudio.cpp

    ld-process-efm - EFM data decoder
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-efm is free software: you can redistribute it and/or
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

#include "f1toaudio.h"

F1ToAudio::F1ToAudio()
{
    debugOn = false;
    reset();
}

// Public methods -----------------------------------------------------------------------------------------------------

// Method to feed the audio processing state-machine with F1 frames
QByteArray F1ToAudio::process(const std::vector<F1Frame> &f1FramesIn, bool _padInitialDiscTime,
                              ErrorTreatment _errorTreatment, ConcealType _concealType,
                              bool debugState)
{
    debugOn = debugState;
    padInitialDiscTime = _padInitialDiscTime;
    errorTreatment = _errorTreatment;
    concealType = _concealType;

    // Clear the output buffer
    pcmOutputBuffer.clear();

    if (f1FramesIn.empty()) return pcmOutputBuffer;

    // Append input data to the processing buffer
    f1FrameBuffer.insert(f1FrameBuffer.end(), f1FramesIn.begin(), f1FramesIn.end());

    waitingForData = false;
    while (!waitingForData) {
        currentState = nextState;

        switch (currentState) {
        case state_initial:
            nextState = sm_state_initial();
            break;
        case state_processFrame:
            nextState = sm_state_processFrame();
            break;
        case state_findEndOfError:
            nextState = sm_state_findEndOfError();
            break;
        }
    }

    return pcmOutputBuffer;
}

// Get method - retrieve statistics
const F1ToAudio::Statistics &F1ToAudio::getStatistics() const
{
    return statistics;
}

// Method to report decoding statistics to qInfo
void F1ToAudio::reportStatistics() const
{
    qInfo()           << "";
    qInfo()           << "F1 Frames to Audio:";
    qInfo()           << "       Audio samples:" << statistics.audioSamples;
    qInfo()           << "     Corrupt samples:" << statistics.corruptSamples;
    qInfo()           << "     Missing samples:" << statistics.missingSamples;
    qInfo()           << "   Concealed samples:" << statistics.concealedSamples;
    qInfo()           << "       Total samples:" << statistics.totalSamples;
    qInfo()           << "";
    qInfo().noquote() << "    Audio start time:" << statistics.startTime.getTimeAsQString();
    qInfo().noquote() << "  Audio current time:" << statistics.currentTime.getTimeAsQString();
    qInfo().noquote() << "      Audio duration:" << statistics.duration.getTimeAsQString();
}

// Reset the object
void F1ToAudio::reset()
{
    f1FrameBuffer.clear();
    pcmOutputBuffer.clear();
    waitingForData = false;
    currentState = state_initial;
    nextState = currentState;
    padInitialDiscTime = false;
    gotFirstSample = false;
    initialDiscTimeSet = false;

    clearStatistics();
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to clear the statistics counters
void F1ToAudio::clearStatistics()
{
    statistics.audioSamples = 0;
    statistics.corruptSamples = 0;
    statistics.missingSamples = 0;
    statistics.concealedSamples = 0;
    statistics.totalSamples = 0;

    statistics.currentTime.setTime(0, 0, 0);
    statistics.startTime.setTime(0, 0, 0);
    statistics.duration.setTime(0, 0, 0);
}

// State-machine methods ----------------------------------------------------------------------------------------------

F1ToAudio::StateMachine F1ToAudio::sm_state_initial()
{
    if (debugOn) qDebug() << "AudioSampleFramesToPcm::sm_state_initial(): Called";

    // If we are padding, set initial disc time to 00:00.00
    if (padInitialDiscTime) {
        statistics.startTime.setTime(0, 0, 0);
        initialDiscTimeSet = true;
    }

    return state_processFrame;
}

// Process the waiting audio frames into PCM data
F1ToAudio::StateMachine F1ToAudio::sm_state_processFrame()
{
    // If error treatment is silence or pass-though, use a fast, simple method
    if (errorTreatment == ErrorTreatment::silence || errorTreatment == ErrorTreatment::passThrough) {
        for (const F1Frame &f1Frame: f1FrameBuffer) {
            uchar f1FrameData[24];

            // If error correction is silence, set corrupt samples to silence
            if (f1Frame.isCorrupt() || f1Frame.isMissing()) {
                // Frame is corrupt, use zeroed data
                for (qint32 j = 0; j < 24; j++) f1FrameData[j] = 0;
                if (f1Frame.isCorrupt()) statistics.corruptSamples += 6;
                if (f1Frame.isMissing()) {
                    if (!padInitialDiscTime) {
                        // Only count as a missing sample after the first good sample is seen
                        if (gotFirstSample) statistics.missingSamples += 6;
                    }
                }
            } else {
                // Frame is good - Get the frame data
                for (qint32 j = 0; j < 24; j++) f1FrameData[j] = f1Frame.getDataSymbols()[j];
                statistics.audioSamples += 6;
                gotFirstSample = true;
                if (!initialDiscTimeSet) {
                    statistics.startTime = f1Frame.getDiscTime();
                    initialDiscTimeSet = true;
                }
            }

            // Append the F1 frame data to the PCM output buffer
            if (padInitialDiscTime) {
                // Padding to initial disc time
                pcmOutputBuffer.append(QByteArray(reinterpret_cast<char*>(f1FrameData), 24));
                statistics.totalSamples += 6;
            } else {
                // Only pad after first good sample
                if (gotFirstSample) {
                    pcmOutputBuffer.append(QByteArray(reinterpret_cast<char*>(f1FrameData), 24));
                    statistics.totalSamples += 6;
                }
            }

            statistics.currentTime = f1Frame.getDiscTime();
            statistics.duration.setTime(0, 0, 0);
            statistics.duration.addFrames(statistics.currentTime.getDifference(statistics.startTime.getTime()));
        }

        // Remove the consumed audio frames from the buffer
        f1FrameBuffer.clear();

        waitingForData = true;
        return state_processFrame;
    }

    // Error treatment is conceal
    qint32 bufferPosition = 0;
    uchar f1FrameData[24];
    while (bufferPosition < static_cast<qint32>(f1FrameBuffer.size())) {
        if (!f1FrameBuffer[bufferPosition].isCorrupt()) {
            if (!f1FrameBuffer[bufferPosition].isMissing()) {
                // Frame is not corrupt and not missing... good Frame
                // Append the audio sample's frame data to the output buffer
                for (qint32 j = 0; j < 24; j++) f1FrameData[j] = f1FrameBuffer[bufferPosition].getDataSymbols()[j];
                pcmOutputBuffer.append(QByteArray(reinterpret_cast<char*>(f1FrameData), 24));
                statistics.audioSamples += 6;
                statistics.totalSamples += 6;
                gotFirstSample = true;
                if (!initialDiscTimeSet) {
                    statistics.startTime = f1FrameBuffer[bufferPosition].getDiscTime();
                    initialDiscTimeSet = true;
                }
            } else {
                // Frame is not corrupt, but is missing...
                if (padInitialDiscTime) {
                    // Append silent frame data to the output buffer
                    for (qint32 j = 0; j < 24; j++) f1FrameData[j] = 0;
                    pcmOutputBuffer.append(QByteArray(reinterpret_cast<char*>(f1FrameData), 24));
                    statistics.missingSamples += 6;
                    statistics.totalSamples += 6;
                } else {
                    // Only pad after first good sample
                    if (gotFirstSample) {
                        // Append silent frame data to the output buffer
                        for (qint32 j = 0; j < 24; j++) f1FrameData[j] = 0;
                        pcmOutputBuffer.append(QByteArray(reinterpret_cast<char*>(f1FrameData), 24));
                        statistics.missingSamples += 6;
                        statistics.totalSamples += 6;
                    }
                }
            }

            // Store the frame as the last good frame seen (as the last known good
            // might be in the previously consumed buffer with the start error at the
            // start of the new buffer)
            lastGoodFrame = f1FrameBuffer[bufferPosition];

            statistics.currentTime = f1FrameBuffer[bufferPosition].getDiscTime();
            statistics.duration.setTime(0, 0, 0);
            statistics.duration.addFrames(statistics.currentTime.getDifference(statistics.startTime.getTime()));
        } else {
            // Frame is corrupt...
            errorStartPosition = bufferPosition;
            return state_findEndOfError;
        }
        bufferPosition++;
    }

    // We only get here if there is no more data in the input buffer.
    // Remove the consumed audio frames from the buffer and request more
    f1FrameBuffer.clear();
    waitingForData = true;
    return state_processFrame;
}

// Find the end of an error run
F1ToAudio::StateMachine F1ToAudio::sm_state_findEndOfError()
{
    qint32 bufferPosition = errorStartPosition;
    errorStopPosition = -1;
    while (bufferPosition < static_cast<qint32>(f1FrameBuffer.size()) && errorStopPosition == -1) {
        if (!f1FrameBuffer[bufferPosition].isCorrupt()) {
            errorStopPosition = bufferPosition - 1; // Last corrupt frame
        }
        bufferPosition++;
    }

    // If we didn't find the end of the error in the buffer, we need to
    // request more data and then try again
    if (errorStopPosition == -1) {
        if (debugOn) qDebug() << "AudioSampleFramesToPcm::sm_state_findEndOfError(): End of the error run not in buffer - requesting more data";
        waitingForData = true;
        return state_findEndOfError;
    }

    // Report the location of the error to debug
    if (debugOn) qDebug() << "F1ToAudio::sm_state_findEndOfError():" <<
                             "Found error run from section" << f1FrameBuffer[errorStartPosition].getDiscTime().getTimeAsQString() <<
                             "to section" << f1FrameBuffer[errorStopPosition].getDiscTime().getTimeAsQString() <<
                             "which is" << errorStopPosition - errorStartPosition + 1 << "frame(s) long." <<
                             "Buffer start" << errorStartPosition << "to end" << errorStopPosition;

    // Mark the next good frame
    nextGoodFrame = f1FrameBuffer[errorStopPosition + 1];

    // Perform concealment
    switch (concealType) {
    case ConcealType::linear:
        linearInterpolationConceal();
        break;
    case ConcealType::prediction:
        predictiveInterpolationConceal();
        break;
    }

    // Remove the contents of the input buffer (up to the error start)
    f1FrameBuffer.erase(f1FrameBuffer.begin(), f1FrameBuffer.begin() + errorStopPosition + 1);

    // Make sure the buffer isn't completely empty
    if (f1FrameBuffer.size() == 0) waitingForData = true;
    return state_processFrame;
}

// Conceal audio error using simple linear interpolation (draws a straight 'line' between
// the start and end sample values)
void F1ToAudio::linearInterpolationConceal()
{
    Audio lastGoodAudio(lastGoodFrame.getDataSymbols());
    Audio nextGoodAudio(nextGoodFrame.getDataSymbols());

    // Get the start and end values from the last and next known-good frames
    qint16 leftChannelStartValue = lastGoodAudio.getSampleValues().leftSamples[5];
    qint16 leftChannelEndValue = nextGoodAudio.getSampleValues().leftSamples[0];
    qint16 rightChannelStartValue = lastGoodAudio.getSampleValues().rightSamples[5];
    qint16 rightChannelEndValue = nextGoodAudio.getSampleValues().rightSamples[0];

    qint32 framesToGenerate = errorStopPosition - errorStartPosition + 1;
    qint32 samplesToGenerate = framesToGenerate * 6; // Per stereo channel

    // Create some temporary buffers
    std::vector<qint16> leftSamples(samplesToGenerate);
    std::vector<qint16> rightSamples(samplesToGenerate);

    // Calculate sample step values and initial values
    double leftStep = (static_cast<double>(leftChannelEndValue) -
                       static_cast<double>(leftChannelStartValue)) / static_cast<double>(samplesToGenerate);
    double leftValue = static_cast<double>(leftChannelStartValue);

    double rightStep = (static_cast<double>(rightChannelEndValue) -
                        static_cast<double>(rightChannelStartValue)) / static_cast<double>(samplesToGenerate);
    double rightValue = static_cast<double>(rightChannelStartValue);

    // Generate the interpolated samples
    for (qint32 i = 0; i < samplesToGenerate; i++) {
        leftValue += leftStep;
        leftSamples[i] = static_cast<qint16>(leftValue);
        rightValue += rightStep;
        rightSamples[i] = static_cast<qint16>(rightValue);
    }

    // Copy the interpolated sample values into the output buffer
    qint32 samplePointer = 0;
    for (qint32 i = 0; i < framesToGenerate; i++) {
        Audio outputSample;
        Audio::SampleValues sampleValues;

        for (qint32 x = 0; x < 6; x++) {
            sampleValues.leftSamples[x] = leftSamples[samplePointer];
            sampleValues.rightSamples[x] = rightSamples[samplePointer];
            samplePointer++;
        }
        outputSample.setSampleValues(sampleValues);
        pcmOutputBuffer.append(QByteArray(reinterpret_cast<const char *>(outputSample.getSampleFrame()), 24));
        statistics.concealedSamples += 6;
        statistics.totalSamples += 6;
    }
}

// Conceal audio error using Interpolated error prediction - this is a custom
// form of (experimental) concealment
void F1ToAudio::predictiveInterpolationConceal()
{
    Audio lastGoodAudio(lastGoodFrame.getDataSymbols());
    Audio nextGoodAudio(nextGoodFrame.getDataSymbols());

    // Error threshold (in 16-bit signed sample amplitude units)
    qint32 errorThreshold = 1024;

    // Get the start and end values from the last and next known-good frames
    qint16 leftChannelStartValue = lastGoodAudio.getSampleValues().leftSamples[5];
    qint16 leftChannelEndValue = nextGoodAudio.getSampleValues().leftSamples[0];
    qint16 rightChannelStartValue = lastGoodAudio.getSampleValues().rightSamples[5];
    qint16 rightChannelEndValue = nextGoodAudio.getSampleValues().rightSamples[0];

    qint32 framesToGenerate = errorStopPosition - errorStartPosition + 1;
    qint32 samplesToGenerate = framesToGenerate * 6; // Per stereo channel

    // Create some temporary buffers
    std::vector<qint16> leftSamples(samplesToGenerate);
    std::vector<qint16> rightSamples(samplesToGenerate);

    // Calculate sample step values and initial values
    double leftStep = (static_cast<double>(leftChannelEndValue) -
                       static_cast<double>(leftChannelStartValue)) / static_cast<double>(samplesToGenerate);
    double leftValue = static_cast<double>(leftChannelStartValue);

    double rightStep = (static_cast<double>(rightChannelEndValue) -
                        static_cast<double>(rightChannelStartValue)) / static_cast<double>(samplesToGenerate);
    double rightValue = static_cast<double>(rightChannelStartValue);

    // Generate the interpolated samples
    for (qint32 i = 0; i < samplesToGenerate; i++) {
        leftValue += leftStep;
        leftSamples[i] = static_cast<qint16>(leftValue);
        rightValue += rightStep;
        rightSamples[i] = static_cast<qint16>(rightValue);
    }

    // Copy the interpolated sample values into the frame(s) if the threshold is exceeded
    // otherwise use the original sample value
    qint32 samplePointer = 0;
    for (qint32 i = 0; i < framesToGenerate; i++) {
        Audio nextSample(f1FrameBuffer[errorStartPosition + i].getDataSymbols());
        Audio outputSample;
        Audio::SampleValues sampleValues;

        for (qint32 x = 0; x < 6; x++) {
            qint32 leftDifference = abs(leftSamples[samplePointer] - nextSample.getSampleValues().leftSamples[x]);
            qint32 rightDifference = abs(rightSamples[samplePointer] - nextSample.getSampleValues().rightSamples[x]);

            if (leftDifference <= errorThreshold) sampleValues.leftSamples[x] = nextSample.getSampleValues().leftSamples[x];
            else sampleValues.leftSamples[x] = leftSamples[samplePointer];

            if (rightDifference <= errorThreshold) sampleValues.rightSamples[x] = nextSample.getSampleValues().rightSamples[x];
            else sampleValues.rightSamples[x] = rightSamples[samplePointer];

            samplePointer++;
        }
        outputSample.setSampleValues(sampleValues);
        pcmOutputBuffer.append(QByteArray(reinterpret_cast<const char *>(outputSample.getSampleFrame()), 24));
        statistics.concealedSamples += 6;
        statistics.totalSamples += 6;
    }
}
