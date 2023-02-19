/************************************************************************

    f2framestoaudio.cpp

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

#include "f2tof1frames.h"

F2ToF1Frames::F2ToF1Frames()
{
    debugOn = false;
    reset();
}

// Public methods -----------------------------------------------------------------------------------------------------

// Method to feed the audio processing state-machine with F2Frames
const std::vector<F1Frame> &F2ToF1Frames::process(const std::vector<F2Frame> &f2FramesIn, bool _debugState, bool _noTimeStamp)
{
    debugOn = _debugState;
    noTimeStamp = _noTimeStamp;

    // Clear the output buffer
    f1FramesOut.clear();

    if (f2FramesIn.empty()) return f1FramesOut;

    // Append input data to the processing buffer
    f2FrameBuffer.insert(f2FrameBuffer.end(), f2FramesIn.begin(), f2FramesIn.end());

    waitingForData = false;
    while (!waitingForData) {
        currentState = nextState;

        switch (currentState) {
        case state_initial:
            nextState = sm_state_initial();
            break;
        case state_getInitialDiscTime:
            nextState = sm_state_getInitialDiscTime();
            break;
        case state_processSection:
            nextState = sm_state_processSection();
            break;
        }
    }

    return f1FramesOut;
}

// Get method - retrieve statistics
const F2ToF1Frames::Statistics &F2ToF1Frames::getStatistics() const
{
    return statistics;
}

// Method to report decoding statistics to qInfo
void F2ToF1Frames::reportStatistics() const
{
    qInfo()           << "";
    qInfo()           << "F2 Frames to F1 Frames:";
    qInfo()           << "            Valid F2 frames:" << statistics.validF2Frames;
    qInfo()           << "          Invalid F2 frames:" << statistics.invalidF2Frames;
    qInfo()           << "     Initial padding frames:" << statistics.initialPaddingFrames;
    qInfo()           << "     Missing section frames:" << statistics.missingSectionFrames;
    qInfo()           << "         Encoder off frames:" << statistics.encoderOffFrames;
    qInfo()           << "               TOTAL frames:" << statistics.totalFrames;
    qInfo()           << "";
    qInfo().noquote() << "       Frames start time:" << statistics.framesStart.getTimeAsQString();
    qInfo().noquote() << "         Frames end time:" << statistics.frameCurrent.getTimeAsQString();
}

// Method to reset the class
void F2ToF1Frames::reset()
{
    // Initialise variables to track the disc time
    lastDiscTime.setTime(0, 0, 0);

    f2FrameBuffer.clear();
    f1FramesOut.clear();
    waitingForData = false;
    currentState = state_initial;
    nextState = currentState;

    clearStatistics();
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to clear the statistics counters
void F2ToF1Frames::clearStatistics()
{
    statistics.validF2Frames = 0;
    statistics.initialPaddingFrames = 0;
    statistics.missingSectionFrames = 0;
    statistics.encoderOffFrames = 0;
    statistics.totalFrames = 0;
    statistics.invalidF2Frames = 0;

    statistics.framesStart.setTime(0, 0, 0);
    statistics.frameCurrent.setTime(0, 0, 0);
}

// State-machine methods ----------------------------------------------------------------------------------------------

F2ToF1Frames::StateMachine F2ToF1Frames::sm_state_initial()
{
    if (debugOn) qDebug() << "F2ToF1Frames::sm_state_initial(): Called";
    return state_getInitialDiscTime;
}

// Get the initial disc time
F2ToF1Frames::StateMachine F2ToF1Frames::sm_state_getInitialDiscTime()
{
    lastDiscTime = f2FrameBuffer[0].getDiscTime();
    statistics.framesStart = lastDiscTime;
    statistics.frameCurrent = lastDiscTime;
    if (debugOn) qDebug() << "F2ToF1Frames::sm_state_getInitialDiscTime(): Initial disc time is" << lastDiscTime.getTimeAsQString();

    // If the F1 frame isn't at an initial disc time of 00:00.00, then we pad the F1 frames up to the
    // first F1 frame received

    // Set the startDiscTime to 00:00.00
    TrackTime startDiscTime;
    startDiscTime.setTime(0, 0, 0);

    // Check that this section is one frame difference from the startDiscTime
    // and pad the output sample data if sections are missing
    qint32 sectionFrameGap = lastDiscTime.getDifference(startDiscTime.getTime());
    if (sectionFrameGap > 1) {
        if (debugOn) qDebug().noquote() << "F2ToF1Frames::sm_state_getInitialDiscTime(): Initial disc time gap - Adding" <<
                                           sectionFrameGap - 1 << "section(s) of padding (" << (sectionFrameGap - 1) * 98 << "F1 frames )";

        // Pad the output F1 frames according to the gap
        F1Frame f1Frame;
        uchar outputData[24];
        for (qint32 i = 0; i < 24; i++) outputData[i] = 0;

        // Loop per section
        for (qint32 p = 0; p < sectionFrameGap - 1; p++) {
            // 98 F1 frames per section
            lastDiscTime.addFrames(1);

            f1Frame.setData(outputData, false, true, true, lastDiscTime, TrackTime(0, 0, 0), 0);

            for (qint32 s = 0; s < 98; s++) {
                f1FramesOut.push_back(f1Frame);
            }

            // Add filled section to statistics
            statistics.initialPaddingFrames += 98;
            statistics.totalFrames += 98;
        }
    }

    // To make sure the current time is processed correctly we have to
    // set the last disc time to -1 frame
    lastDiscTime.subtractFrames(1);

    return state_processSection;
}

F2ToF1Frames::StateMachine F2ToF1Frames::sm_state_processSection()
{
    // Get the current disc time for the section
    TrackTime currentDiscTime = f2FrameBuffer[0].getDiscTime();
    //if (debugOn) qDebug() << "F2ToF1Frames::sm_state_processSection(): Current disc time is" << currentDiscTime.getTimeAsQString();

    // Check that this section is one frame difference from the previous
    // and pad the output sample data if sections are missing
    qint32 sectionFrameGap = currentDiscTime.getDifference(lastDiscTime.getTime());
    if (sectionFrameGap > 1) {
        if (debugOn) qDebug().noquote() << "F2ToF1Frames::sm_state_processSection(): Section gap - Last seen time was" << lastDiscTime.getTimeAsQString() <<
                                 "current disc time is" << currentDiscTime.getTimeAsQString() <<
                                 "Adding" << sectionFrameGap - 1 << "section(s) of padding (" << (sectionFrameGap - 1) * 98 << "frames )";

        // Pad the output F1 frames according to the gap
        F1Frame f1Frame;
        uchar outputData[24];
        for (qint32 i = 0; i < 24; i++) outputData[i] = 0;

        // Loop per section
        for (qint32 p = 0; p < sectionFrameGap - 1; p++) {
            // 98 Audio frames per section
            lastDiscTime.addFrames(1);

            f1Frame.setData(outputData, false, true, true, lastDiscTime, TrackTime(0, 0, 0), 0);

            for (qint32 s = 0; s < 98; s++) {
                f1FramesOut.push_back(f1Frame);
            }

            // Add filled section to statistics
            statistics.missingSectionFrames += 98;
            statistics.totalFrames += 98;
        }
    }

    // Store the current disc time as the last disc time for the next cycle of processing
    lastDiscTime = currentDiscTime;
    statistics.frameCurrent = currentDiscTime;

    // Determine if the section is flagged as encoder on or off (using a threshold to prevent false-negatives)
    bool sectionEncoderState = false;
    qint32 encoderStateCount = 0;
    for (qint32 i = 0; i < 98; i++) {
        if (f2FrameBuffer[i].getIsEncoderRunning()) encoderStateCount++;
    }
    if (encoderStateCount > 10) sectionEncoderState = true; else sectionEncoderState = false;

    // Override the encoder state for non-standard EFM with no time-stamps
    if (noTimeStamp) sectionEncoderState = true;

    // Output the F2 Frames as F1 Frames
    F1Frame f1Frame;
    for (qint32 i = 0; i < 98; i++) {
        f1Frame.setData(f2FrameBuffer[i].getDataSymbols(), f2FrameBuffer[i].isFrameCorrupt(), sectionEncoderState, false,
                        f2FrameBuffer[i].getDiscTime(), f2FrameBuffer[i].getTrackTime(), f2FrameBuffer[i].getTrackNumber());
        f1FramesOut.push_back(f1Frame);

        // Update the statistics
        if (f2FrameBuffer[i].isFrameCorrupt()) statistics.invalidF2Frames++; else statistics.validF2Frames++;
        if (!sectionEncoderState) statistics.encoderOffFrames++;
        statistics.totalFrames++;
    }

    // Remove the processed section from the F2 frame buffer
    f2FrameBuffer.erase(f2FrameBuffer.begin(), f2FrameBuffer.begin() + 98);

    // Request more F2 frame data if required
    if (f2FrameBuffer.size() < 98) waitingForData = true;

    return state_processSection;
}



