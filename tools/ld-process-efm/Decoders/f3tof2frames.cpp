/************************************************************************

    f3tof2frames.cpp

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

#include "f3tof2frames.h"

F3ToF2Frames::F3ToF2Frames()
{
    debugOn = false;
    reset();
}

// Public methods -----------------------------------------------------------------------------------------------------

const std::vector<F2Frame> &F3ToF2Frames::process(const std::vector<F3Frame> &f3FramesIn, bool debugState, bool noTimeStamp)
{
    debugOn = debugState;

    // Clear the output buffer
    f2FramesOut.clear();

    // Make sure there is something to process
    if (f3FramesIn.empty()) return f2FramesOut;

    // Ensure that the upstream is providing only complete sections of
    // 98 frames... otherwise we have an upstream bug.
    if (f3FramesIn.size() % 98 != 0) {
        qFatal("F3ToF2Frames::process(): Upstream has provided incomplete sections of 98 F3 frames - This is a bug!");
        // Exection stops...
        // return f2FramesOut;
    }

    // Process the incoming F3 Frames.
    // Input data must be available in sections of 98 F3 frames, synchronised with a section.
    const qint32 numInputFrames = static_cast<qint32>(f3FramesIn.size());
    for (qint32 inputIndex = 0; inputIndex < numInputFrames; inputIndex += 98) {
        statistics.totalF3Frames += 98;

        // Collect the 98 subcode data symbols
        uchar sectionData[98];
        for (qint32 i = 0; i < 98; i++) {
            sectionData[i] = f3FramesIn[inputIndex + i].getSubcodeSymbol();
        }

        // Process the subcode data into a section
        Section section;
        section.setData(sectionData);

        // Check the timestamp is plausible. There's a 1/65536 chance of
        // corrupt data making it through the CRC, so if the timestamp is
        // clearly wrong then the rest of the Q data should be ignored too.
        if (section.getQMode() == 1 || section.getQMode() == 4) {
            const TrackTime currentDiscTime = section.getQMetadata().qMode1And4.discTime;
            const qint32 framesSinceLast = currentDiscTime.getDifference(statistics.initialDiscTime.getTime());
            // A CD/LaserDisc side shouldn't be more than 100 minutes long
            if (framesSinceLast > (100 * 60 * 75)) {
                if (debugOn) qDebug().noquote() << "F3ToF2Frames::process(): Implausible section time stamp" << currentDiscTime.getTimeAsQString() << "given initial time" << statistics.initialDiscTime.getTimeAsQString() << "- ignoring section Q data";
                section = Section();
            }
        }

        // Check the audio preemp flag (false = pre-emp audio)
        if (section.getQMode() == 1 || section.getQMode() == 4) {
            if (!section.getQMetadata().qControl.isNoPreempNotPreemp) statistics.preempFrames++;
        }

        // Do we have an initial disc time?
        if (!initialDiscTimeSet) {
            // Initial disc time is not set...
            if (noTimeStamp) {
                // This is a special condition for when the EFM doesn't follow the standards and no
                // time-stamp information is available.  We can only assume that it starts from
                // zero and that there are no skips or jumps in the original disc data...
                TrackTime currentDiscTime;
                currentDiscTime.setTime(0, 0, 0);
                statistics.initialDiscTime = currentDiscTime;
                lastDiscTime = currentDiscTime;
                lastDiscTime.subtractFrames(1);
                if (debugOn) qDebug().noquote() << "F3ToF2Frames::process(): No time stamps... Initial disc time is set to" << currentDiscTime.getTimeAsQString();
                initialDiscTimeSet = true;
            } else {
                // Ensure the QMode is valid
                if ((section.getQMode() == 1 || section.getQMode() == 4) &&
                        (!section.getQMetadata().qMode1And4.isLeadIn && !section.getQMetadata().qMode1And4.isLeadOut)) {
                    TrackTime currentDiscTime;
                    statistics.initialDiscTime = section.getQMetadata().qMode1And4.discTime;
                    currentDiscTime = section.getQMetadata().qMode1And4.discTime;

                    lastDiscTime = currentDiscTime;
                    lastDiscTime.subtractFrames(1);

                    if (debugOn) qDebug().noquote() << "F3ToF2Frames::process(): Initial disc time is" << currentDiscTime.getTimeAsQString();
                    initialDiscTimeSet = true;
                } else {
                    // We can't use the current section, report why and then disregard
                    if (section.getQMode() != 1 && section.getQMode() != 4) if (debugOn) qDebug() << "F3ToF2Frames::process(): Current section is not QMode 1 or 4";
                    if (section.getQMetadata().qMode1And4.isLeadIn || section.getQMetadata().qMode1And4.isLeadOut) if (debugOn) qDebug() << "F3ToF2Frames::process(): Current section is lead in/out";

                    // Drop the section
                    if (debugOn) qDebug() << "F3ToF2Frames::process(): Ignoring section (disregards 98 F3 frames)";
                }
            }
        }

        if (initialDiscTimeSet) {
            // We have an initial disc time
            TrackTime currentDiscTime;

            // Compare the last known disc time to the current disc time
            if (section.getQMode() == 1 || section.getQMode() == 4) {
                if (!noTimeStamp) {
                    // Just checkin'
                    if (section.getQMetadata().qMode1And4.isLeadIn || section.getQMetadata().qMode1And4.isLeadOut) {
                        if (debugOn) qDebug() << "F3ToF2Frames::process(): Weird!  Seeing lead/out frames after a valid initial disc time";
                    }

                    // Current section has a valid disc time - read it
                    currentDiscTime = section.getQMetadata().qMode1And4.discTime;
                } else {
                    // We have to fake the time-stamp here
                    currentDiscTime = lastDiscTime;
                    currentDiscTime.addFrames(1); // We assume this section is contiguous
                }

                if (lostSections) {
                    if (debugOn) qDebug().noquote() << "F3ToF2Frames::process(): First valid time after section loss is" << currentDiscTime.getTimeAsQString();
                    lostSections = false;
                }
            } else {
                // Current section does not have a valid disc time - estimate it
                currentDiscTime = lastDiscTime;
                currentDiscTime.addFrames(1); // We assume this section is contiguous
                if (debugOn) qDebug().noquote() << "F3ToF2Frames::process(): Section disc time not valid, setting current disc time to" << currentDiscTime.getTimeAsQString() <<
                                                   "based on last disc time of" << lastDiscTime.getTimeAsQString();

                if (lostSections) {
                    if (debugOn) qDebug().noquote() << "F3ToF2Frames::process(): First invalid guessed time after section loss is" << currentDiscTime.getTimeAsQString();
                    lostSections = false;
                }
            }

            // Check that this section is one frame difference from the previous
            qint32 sectionFrameGap = currentDiscTime.getDifference(lastDiscTime.getTime());

            if (sectionFrameGap > 1) {
                // The incoming F3 section isn't contiguous with the previous F3 section
                // this means the C1, C2 and deinterleave buffers are full of the wrong
                // data... so here we flush them to speed up the recovery time
                if (debugOn) qDebug() << "F3ToF2Frames::process(): Non-contiguous F3 section with" << sectionFrameGap - 1 << "sections missing - Last disc time was" <<
                            lastDiscTime.getTimeAsQString() << "current disc time is" << currentDiscTime.getTimeAsQString();
                if (debugOn) qDebug() << "F3ToF2Frames::process(): Lost" << (sectionFrameGap - 1) * 98 << "F3 frames (" << (sectionFrameGap - 1) <<
                                         "sections ) - Flushing C1, C2 buffers and section metadata";
                statistics.sequenceInterruptions++;
                statistics.missingF3Frames += (sectionFrameGap - 1) * 98;
                c1Circ.flush();
                c2Circ.flush();
                c2Deinterleave.flush();

                // Also flush the section metadata as it's now out of sync
                sectionBuffer.clear();
                sectionDiscTimes.clear();

                // Mark section loss
                lostSections = true;
            }

            // Store the current disc time as last
            lastDiscTime = currentDiscTime;
            statistics.currentDiscTime = currentDiscTime;

            // Add the new section to our section buffer
            sectionBuffer.push_back(section);
            sectionDiscTimes.push_back(currentDiscTime);

            // Process the F3 frames into F2 frames (payload data)
            for (qint32 i = 0; i < 98; i++) {
                // Process C1 CIRC
                c1Circ.pushF3Frame(f3FramesIn[inputIndex + i]);

                // If we have C1 results, process C2
                if (c1Circ.getDataSymbols() != nullptr) {
                    // Get C1 results
                    uchar c1DataSymbols[28];
                    uchar c1ErrorSymbols[28];
                    for (qint32 i = 0; i < 28; i++) {
                        c1DataSymbols[i] = c1Circ.getDataSymbols()[i];
                        c1ErrorSymbols[i] = c1Circ.getErrorSymbols()[i];
                    }

                    // Process C2 CIRC
                    c2Circ.pushC1(c1DataSymbols, c1ErrorSymbols);

                    // Only process the F2 frames if we received data
                    if (c2Circ.getDataSymbols() != nullptr) {
                        // Get C2 results
                        uchar c2DataSymbols[28];
                        uchar c2ErrorSymbols[28];
                        for (qint32 i = 0; i < 28; i++) {
                            c2DataSymbols[i] = c2Circ.getDataSymbols()[i];
                            c2ErrorSymbols[i] = c2Circ.getErrorSymbols()[i];
                        }

                        // Deinterleave the C2
                        c2Deinterleave.pushC2(c2DataSymbols, c2ErrorSymbols);

                        // If we have deinterleaved C2s, create an F2 frame
                        if (c2Deinterleave.getDataSymbols() != nullptr) {
                            // Get C2 deinterleave results
                            uchar c2DeinterleavedData[24];
                            uchar c2DeinterleavedErrors[24];
                            for (qint32 i = 0; i < 24; i++) {
                                c2DeinterleavedData[i] = c2Deinterleave.getDataSymbols()[i];
                                c2DeinterleavedErrors[i] = c2Deinterleave.getErrorSymbols()[i];
                            }

                            F2Frame newF2Frame;
                            newF2Frame.setData(c2DeinterleavedData, c2DeinterleavedErrors);

                            // Add the section metadata to the F2 Frame (each section is applied to
                            // 98 F2 frames)

                            // Always output the disc time from the corrected local version
                            newF2Frame.setDiscTime(sectionDiscTimes[0]);

                            // Only use the real metadata if it is valid and available
                            if (sectionBuffer[0].getQMode() == 1 || sectionBuffer[0].getQMode() == 4) {
                                newF2Frame.setTrackTime(sectionBuffer[0].getQMetadata().qMode1And4.trackTime);
                                newF2Frame.setTrackNumber(sectionBuffer[0].getQMetadata().qMode1And4.trackNumber);
                                newF2Frame.setIsEncoderRunning(sectionBuffer[0].getQMetadata().qMode1And4.isEncoderRunning);
                            } else {
                                newF2Frame.setTrackTime(TrackTime(0, 0, 0));
                                newF2Frame.setTrackNumber(1);
                                newF2Frame.setIsEncoderRunning(true);
                            }

                            // Add the F2 frame to our output buffer
                            f2FrameBuffer.push_back(newF2Frame);

                        }
                    }
                }

                // If we have 98 F2 frames, move them to the output buffer
                if (f2FrameBuffer.size() == 98) {
                    f2FramesOut.insert(f2FramesOut.end(), f2FrameBuffer.begin(), f2FrameBuffer.end());
                    statistics.totalF2Frames += 98;
                    f2FrameBuffer.clear();

                    sectionBuffer.erase(sectionBuffer.begin());
                    sectionDiscTimes.erase(sectionDiscTimes.begin());
                }
            }
        }
    }

    return f2FramesOut;
}

// Get method - retrieve statistics
const F3ToF2Frames::Statistics &F3ToF2Frames::getStatistics()
{
    // Ensure sub-class statistics are updated
    statistics.c1Circ_statistics = c1Circ.getStatistics();
    statistics.c2Circ_statistics = c2Circ.getStatistics();
    statistics.c2Deinterleave_statistics = c2Deinterleave.getStatistics();

    return statistics;
}

// Method to report decoding statistics to qInfo
void F3ToF2Frames::reportStatistics() const
{
    qInfo() << "";
    qInfo() << "F3 Frame to F2 Frame decode:";
    qInfo() << "      Total input F3 Frames:" << statistics.totalF3Frames;
    qInfo() << "     Total output F2 Frames:" << statistics.totalF2Frames;
    qInfo() << "        Total Preemp Frames:" << statistics.preempFrames;
    qInfo() << "  F3 Sequence Interruptions:" << statistics.sequenceInterruptions;
    qInfo() << "          Missing F3 Frames:" << statistics.missingF3Frames;
    qInfo().noquote() << "          Initial disc time:" << statistics.initialDiscTime.getTimeAsQString();
    qInfo().noquote() << "            Final disc time:" << statistics.currentDiscTime.getTimeAsQString();

    // Show C1 CIRC statistics
    c1Circ.reportStatistics();

    // Show C2 CIRC statistics
    c2Circ.reportStatistics();

    // Show C2 Deinterleave statistics
    c2Deinterleave.reportStatistics();
}

// Method to reset the class
void F3ToF2Frames::reset()
{
    // Initialise variables to track the disc time
    initialDiscTimeSet = false;
    lastDiscTime.setTime(0, 0, 0);

    f2FrameBuffer.clear();
    sectionBuffer.clear();
    sectionDiscTimes.clear();

    c1Circ.reset();
    c2Circ.reset();
    c2Deinterleave.reset();
    clearStatistics();

    lostSections = false;
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to clear the statistics counters
void F3ToF2Frames::clearStatistics()
{
    statistics.totalF3Frames = 0;
    statistics.totalF2Frames = 0;

    c1Circ.resetStatistics();
    c2Circ.resetStatistics();
    c2Deinterleave.resetStatistics();

    statistics.initialDiscTime.setTime(0, 0, 0);
    statistics.currentDiscTime.setTime(0, 0, 0);

    statistics.sequenceInterruptions = 0;
    statistics.missingF3Frames = 0;

    statistics.preempFrames = 0;
}
