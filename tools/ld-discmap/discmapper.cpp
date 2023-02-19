/************************************************************************

    discmapper.cpp

    ld-discmap - TBC and VBI alignment and correction
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-discmap is free software: you can redistribute it and/or
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

#include "discmapper.h"

DiscMapper::DiscMapper()
{
    // This space for sale; please enquire within
}

// Method to perform disc mapping process
bool DiscMapper::process(QFileInfo _inputFileInfo, QFileInfo _inputMetadataFileInfo,
                         QFileInfo _outputFileInfo, bool _reverse, bool _mapOnly, bool _noStrict,
                         bool _deleteUnmappable, bool _noAudio)
{
    inputFileInfo = _inputFileInfo;
    inputMetadataFileInfo = _inputMetadataFileInfo;
    outputFileInfo = _outputFileInfo;
    reverse = _reverse;
    mapOnly = _mapOnly;
    noStrict = _noStrict;
    deleteUnmappable = _deleteUnmappable;
    noAudio = _noAudio;

    // Some info for the user...
    qInfo() << "LaserDisc mapping tool";
    qInfo() << "";
    qInfo() << "Please note that disc mapping is not fool-proof - if you";
    qInfo() << "have a disc that does not map correctly run ld-discmap";
    qInfo() << "with the --debug option for more details about the process";
    qInfo() << "(and the additional information required for issue reporting";
    qInfo() << "to the ld-decode project).";
    qInfo() << "";
    qInfo() << "Some early LaserDiscs do not provide frame numbering or";
    qInfo() << "time-code information and cannot be automatically mapped -";
    qInfo() << "if in doubt verify your source TBC file using the ld-analyse";
    qInfo() << "application.";
    qInfo() << "";
    qInfo() << "Note that NTSC CAV pulldown support currently only handles";
    qInfo() << "discs that follow the standard 1-in-5 pulldown pattern.";
    qInfo() << "";

    // Create the source disc map object
    qInfo().noquote() << "Processing input metadata for" << inputFileInfo.filePath();
    if (noStrict) qInfo() << "Not enforcing strict pulldown checking - this can cause false-positive detection";
    DiscMap discMap(inputMetadataFileInfo, reverse, noStrict);
    if (!discMap.valid()) {
        qInfo() << "Could not process TBC metadata successfully - cannot map this disc";
        return false;
    }
    qDebug() << discMap;

    // Show the disc type and video format:
    qInfo().noquote() << "Input TBC is a" << discMap.discType() << "disc using" << discMap.discFormat();

    // Remove lead-in and lead-out frames from the map
    removeLeadInOut(discMap);

    // Remove any frames that have incorrected intra-frame phase sequences
    removeInvalidFramesByPhase(discMap);

    // Detect and correct bad VBI frame numbers using sequence analysis
    correctVbiFrameNumbersUsingSequenceAnalysis(discMap);

    // Detect and remove duplicated frames (does not process pull-down frames)
    removeDuplicateNumberedFrames(discMap);

    // Add numbering to any pulldown frames in the disc map
    numberPulldownFrames(discMap);

    // Verify that there are no frames without frame numbers in the map
    // otherwise reordering will fail
    if (!verifyFrameNumberPresence(discMap)) {
        if (!deleteUnmappable) {
            qInfo() << "";
            qInfo() << "Disc mapping has failed as there are unmappable frames in the disc map!";
            qInfo() << "It is possible that running ld-discmap again with the --delete-unmappable-frames";
            qInfo() << "option set could recitfy this issue.";
            return false;
        }

        qInfo() << "Verification has failed, there are unmappable frames...";
        qInfo() << "--delete-unmappable-frames is set, so the unmappable frames will be deleted";
        deleteUnmappableFrames(discMap);
    }

    // Reorder the frames according to VBI frame number order in the disc map
    reorderFrames(discMap);

    // Pad any gaps in the sequential disc map
    padDiscMap(discMap);

    // If the disc map contains pulldown frames (which are not numbered)
    // rewrite all VBI frame numbers in the disc map in order to number
    // them.
    rewriteFrameNumbers(discMap);

    // All done
    qInfo() << "Disc mapping process completed";

    if (mapOnly) {
        qInfo() << "--maponly selected.  No output file will be written.";
        return true;
    }

    if (noAudio) {
        qInfo() << "-no-audio selected.  No analogue audio output wil be written.";
    }

    qInfo() << "Writing output video and metadata information...";
    saveDiscMap(discMap);

    return true;
}

// Method to remove lead in and lead out frames from the map
void DiscMapper::removeLeadInOut(DiscMap &discMap)
{
    qInfo() << "Checking for lead in and out frames...";
    qint32 leadInOutCounter = 0;
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        if (discMap.isLeadInOut(frameNumber)) {
            discMap.setMarkedForDeletion(frameNumber);
            leadInOutCounter++;
        }

        if (discMap.vbiFrameNumber(frameNumber) == 0) discMap.debugFrameDetails(frameNumber);

        if (!discMap.isLeadInOut(frameNumber) && discMap.isDiscCav() && discMap.vbiFrameNumber(frameNumber) == 0) {
            qInfo() << "Warning: Frame with illegal CAV frame number of 0 found... Assuming illegal leadin and deleting.";
            discMap.setMarkedForDeletion(frameNumber);
            leadInOutCounter++;
        }
    }

    qInfo() << "Removing" << leadInOutCounter << "frames marked as lead in/out";
    discMap.flush();
}

// Method to check that each frame is a valid set of fields according to phase
void DiscMapper::removeInvalidFramesByPhase(DiscMap &discMap)
{
    qInfo() << "Removing invalid frames by phase analysis...";

    qint32 removals = 0;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        // Check that the phase of the first field and second field of the current frame are in sequence
        qint32 expectedNextPhase = discMap.getFirstFieldPhase(frameNumber) + 1; //   m_frames[frameNumber].firstFieldPhase() + 1;
        if (discMap.isDiscPal() && expectedNextPhase == 9) expectedNextPhase = 1;
        if (!discMap.isDiscPal() && expectedNextPhase == 5) expectedNextPhase = 1;
        if (discMap.getSecondFieldPhase(frameNumber) != expectedNextPhase) {
            if (discMap.vbiFrameNumber(frameNumber) != -1) {
                qDebug() << "Marking frame" << frameNumber << "for deletion (VBI Frame#" << discMap.vbiFrameNumber(frameNumber) << ") as first and second field phases are not in sequence! -"
                << expectedNextPhase << "expected but got" << discMap.getSecondFieldPhase(frameNumber);
            } else {
                qDebug() << "Marking frame" << frameNumber << "for deletion (VBI Frame# invalid) as first and second field phases are not in sequence! -"
                << expectedNextPhase << "expected but got" << discMap.getSecondFieldPhase(frameNumber);
            }

            discMap.setMarkedForDeletion(frameNumber);
            removals++;
        }
    }

    qInfo() << "Removing" << removals << "frames marked as invalid due to incorrect phase sequence";
    discMap.flush();
}

// Method to find and correct bad VBI numbers using sequence analysis
void DiscMapper::correctVbiFrameNumbersUsingSequenceAnalysis(DiscMap &discMap)
{
    qInfo() << "Correcting frame numbers using sequence analysis...";

    qint32 scanDistance = 10;
    qint32 corrections = 0;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames() - scanDistance; frameNumber++) {
        // Don't start on a pulldown or a frame with no VBI frame number
        if (!discMap.isPulldown(frameNumber) && discMap.vbiFrameNumber(frameNumber) != -1) {
            qint32 startOfSequence = discMap.vbiFrameNumber(frameNumber);
            qint32 expectedIncrement = 1;

            QVector<bool> vbiGood;
            vbiGood.resize(scanDistance);
            bool sequenceIsGood = true;

            for (qint32 i = 0; i < scanDistance; i++) {
                if (!discMap.isPulldown(frameNumber + i + 1)) {
                    if ((discMap.vbiFrameNumber(frameNumber + i + 1) == startOfSequence + expectedIncrement) ||
                            (discMap.isPulldown(frameNumber + i + 1))) {
                        // Sequence is good
                        sequenceIsGood = true;
                    } else {
                        // Sequence is bad
                        sequenceIsGood = false;
                    }

                    // Set the good/bad flag and increase the frame number increment
                    if (sequenceIsGood) vbiGood[i] = true; else vbiGood[i] = false;
                    expectedIncrement++;
                } else {
                    // Set the good/bad flag but don't increase the frame number
                    // increment due to the pulldown
                    if (sequenceIsGood) vbiGood[i] = true; else vbiGood[i] = false;
                }
            }

            // Did the check pass?
            qint32 count = 0;
            for (qint32 i = 0; i < scanDistance; i++) {
                if (vbiGood[i]) count++;
            }

            // If any frame numbers were bad, check does not pass
            if (count != scanDistance) {
                // Do we have at least 2 good frame numbers (which are not pulldowns) before the error
                qint32 check1 = 0;
                for (qint32 i = 0; i < scanDistance; i++) {
                    if (vbiGood[i] && !discMap.isPulldown(frameNumber + i + 1)) check1++;
                    else if (!discMap.isPulldown(frameNumber + i + 1)) break;
                }

                // and another 2 good frame numbers (which are not pulldowns) after the error?
                qint32 check2 = 0;
                for (qint32 i = scanDistance - 1; i >= 0; i--) {
                    if (vbiGood[i] && !discMap.isPulldown(frameNumber + i + 1)) check2++;
                    else if (!discMap.isPulldown(frameNumber + i + 1)) break;
                }

                if (check1 >= 2 && check2 >= 2) {
                    // We have enough leading and trailing good frame numbers to be sure we are looking
                    // at a real error.  Now correct the error
                    qDebug() << "Broken VBI frame number sequence detected:";

                    bool inError = false;
                    expectedIncrement = 1;
                    for (qint32 i = 0; i < scanDistance; i++) {
                        if (!vbiGood[i]) {
                            inError = true;
                            // Out of sequence frame number
                            if (!discMap.isPulldown(frameNumber + i + 1)) {
                                // Not a pulldown

                                // Ensure this is an error, not a repeating frame
                                if ((discMap.vbiFrameNumber(frameNumber + i + 1) != discMap.vbiFrameNumber(frameNumber + i)) && discMap.isPhaseCorrect(frameNumber + i + 1)) {
                                    qDebug() << "  Position BAD   " << i << "Seq." <<
                                                discMap.seqFrameNumber(frameNumber + i + 1) <<
                                                "VBI was" << discMap.vbiFrameNumber(frameNumber + i + 1) << "now" << (startOfSequence + expectedIncrement)  <<
                                                "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/" <<
                                                discMap.getSecondFieldPhase(frameNumber + i + 1);
                                    discMap.setVbiFrameNumber(frameNumber + i + 1, startOfSequence + expectedIncrement);
                                    if (!discMap.isPulldown(frameNumber + i + 1)) expectedIncrement++;
                                    corrections++;
                                } else {
                                    // Look at the phases to ensure this really is a repeating frame
                                    if (discMap.isPhaseRepeating(frameNumber + i + 1)) {
                                        // Repeating frame
                                        qDebug() << "  Position REPEAT" << i << "Seq." <<
                                                    discMap.seqFrameNumber(frameNumber + i + 1) <<
                                                    "VBI" << discMap.vbiFrameNumber(frameNumber + i + 1)  <<
                                                    "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/" <<
                                                    discMap.getSecondFieldPhase(frameNumber + i + 1);
                                        qDebug() << "  Ignoring sequence break as frame is repeating (VBI and phase) rather than out of sequence";

                                        // If we have a repeat, this probably isn't a sequence issue, so we give up
                                        if (inError) break;
                                    }
                                }
                            } else {
                                // A pulldown (no frame number)
                                qDebug() << "  Position BAD   " << i << "Seq." <<
                                            discMap.seqFrameNumber(frameNumber + i + 1) <<
                                            "VBI pulldown" <<
                                            "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/" <<
                                            discMap.getSecondFieldPhase(frameNumber + i + 1);
                            }
                        } else {
                            // In sequence frame number
                            if (!discMap.isPulldown(frameNumber + i + 1))
                                qDebug() << "  Position GOOD  " << i << "Seq." << discMap.seqFrameNumber(frameNumber + i + 1) <<
                                            "VBI" << discMap.vbiFrameNumber(frameNumber + i + 1) <<
                                            "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/" <<
                                            discMap.getSecondFieldPhase(frameNumber + i + 1);
                            else qDebug() << "  Position GOOD  " << i << "Seq." << discMap.seqFrameNumber(frameNumber + i + 1) <<
                                             "VBI pulldown"  <<
                                             "- Phase" << discMap.getFirstFieldPhase(frameNumber + i + 1) << "/" <<
                                             discMap.getSecondFieldPhase(frameNumber + i + 1);

                            if (!discMap.isPulldown(frameNumber + i + 1)) expectedIncrement++;

                            // Stop once we get a good frame after the bad ones
                            if (inError) break;
                        }
                    }
                }
            }
        }
    }

    qInfo() << "Sequence analysis corrected" << corrections << "frame numbers";
}

// Method to find and remove repeating frames
void DiscMapper::removeDuplicateNumberedFrames(DiscMap &discMap)
{
    qInfo() << "Searching for duplicate frames";
    qDebug() << "Building list of VBIs that have more than one entry in the discmap...";
    QVector<qint32> duplicatedFrameList;
    duplicatedFrameList.reserve(discMap.numberOfFrames()); // just to speed things up a little
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        if (!discMap.isPulldown(frameNumber)) {
            for (qint32 i = frameNumber + 1; i < discMap.numberOfFrames(); i++) {
                // Does the current frameNumber have a duplicate?
                if (discMap.vbiFrameNumber(frameNumber) == discMap.vbiFrameNumber(i) && !discMap.isPulldown(i)) {
                    duplicatedFrameList.append(discMap.vbiFrameNumber(frameNumber));
                }
            }
        }
    }

    qDebug() << "Sorting the duplicated frame list into numerical order...";
    std::sort(duplicatedFrameList.begin(), duplicatedFrameList.end());
    qDebug() << "Removing any repeated frame numbers from the duplicated frame list...";
    auto last = std::unique(duplicatedFrameList.begin(), duplicatedFrameList.end());
    duplicatedFrameList.erase(last, duplicatedFrameList.end());

    qDebug() << "Found" << duplicatedFrameList.size() << "VBI frame numbers with more than 1 entry in the discmap";

    // The duplicated frame list is a list of VBI frame numbers that have duplicates

    // Process the list of duplications one by one
    for (qint32 i = 0; i < duplicatedFrameList.size(); i++) {
        if (duplicatedFrameList[i] != -1) {
            qDebug() << "VBI Frame number" << duplicatedFrameList[i] << "has duplicates; searching for them...";
            QVector<qint32> discMapDuplicateAddress;
            for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
                // Does the current frameNumber's VBI match the VBI in the duplicated frame list?
                if (discMap.vbiFrameNumber(frameNumber) == duplicatedFrameList[i]) {
                    // Add the frame number ot the duplicate disc map address list
                    discMapDuplicateAddress.append(frameNumber);
//                    qDebug() << "  Seq frame" << discMap.seqFrameNumber(frameNumber) << "is a duplicate of" <<
//                                duplicatedFrameList[i] <<
//                                "with a quality of" << discMap.frameQuality(frameNumber);
                }
            }

            // Show the number of duplicates in the discMap that were found
            qDebug() << "  Found" << discMapDuplicateAddress.size() << "duplicates of VBI frame" << duplicatedFrameList[i];

            // Pick the sequential frame duplicate with the best quality
            qint32 bestDiscMapFrame = discMapDuplicateAddress.first();
            for (qint32 i = 0; i < discMapDuplicateAddress.size(); i++) {
                if (discMap.frameQuality(bestDiscMapFrame) < discMap.frameQuality(discMapDuplicateAddress[i])) {
                    bestDiscMapFrame = discMapDuplicateAddress[i];
                }
            }

            qDebug() << "  Highest quality duplicate of VBI" << duplicatedFrameList[i] << "is sequential frame" <<
                        discMap.seqFrameNumber(bestDiscMapFrame) << "with a quality of" << discMap.frameQuality(bestDiscMapFrame);

            // Delete all duplicates except the best sequential frame
            for (qint32 i = 0; i < discMapDuplicateAddress.size(); i++) {
                if (discMapDuplicateAddress[i] != bestDiscMapFrame) {
                    discMap.setMarkedForDeletion(discMapDuplicateAddress[i]);
                    //qDebug() << " Seq. frame" << discMap.seqFrameNumber(discMapDuplicateAddress[i]) << "marked for deletion";
                }
            }
        } else {
            if (!discMap.isDiscPal()) {
                // Having NTSC frames without numbering (that are not pulldown) is a bad thing...
                qInfo() << "";
                qInfo() << "Warning:";
                qInfo() << "There are frames without a frame number (that are not flagged as pulldown) in the duplicate frame list";
                qInfo() << "This probably means that the disc map contains pulldown frames that do not follow the normal 1 in 5";
                qInfo() << "pulldown pattern - and disc mapping will likely fail!";
                qInfo() << "";
            } else {
                qInfo() << "";
                qInfo() << "Warning:";
                qInfo() << "There are frames without a frame number in the duplicate frame list.  Since numberless frames are";
                qInfo() << "usually unmappable, disc mapping will likely fail unless the --delete-unmappable-frames option is";
                qInfo() << "used.";
                qInfo() << "";
            }
        }
    }

    // Delete duplicates
    qint32 originalSize = discMap.numberOfFrames();
    discMap.flush();
    qInfo() << "Removed" << originalSize - discMap.numberOfFrames() <<
               "duplicate frames - disc map size now" << discMap.numberOfFrames() << "frames";
}

// Method to give pulldown frames a real frame number so they can be
// sorted correctly with the other frames
void DiscMapper::numberPulldownFrames(DiscMap &discMap)
{
    if (discMap.isDiscCav() && !discMap.isDiscPal()) {
        qInfo() << "Numbering pulldown frames in the disc map...";

        // This gives each pulldown frame the same frame number as the previous frame
        // (later sorting uses both the frame number and the pulldown flag to ensure
        // that the frame order remains correct)
        for (qint32 i = 1; i < discMap.numberOfFrames(); i++) {
            if (discMap.isPulldown(i)) discMap.setVbiFrameNumber(i, discMap.vbiFrameNumber(i - 1));
        }

        // Check that the very first frame isn't a pull-down
        if (discMap.isPulldown(0)) {
            discMap.setVbiFrameNumber(0, discMap.vbiFrameNumber(1) - 1);
            qInfo() << "Attempted to number pulldown frames, but first frame is a pulldown.";
            qInfo() << "This probably isn't a good thing, but continuing anyway...";
        }

        qInfo() << "Numbering complete";
    }
}

// Method to verify that all frames in the map have VBI frame numbers
bool DiscMapper::verifyFrameNumberPresence(DiscMap &discMap)
{
    qInfo() << "Verifying frame numbers are present for all frames in the disc map...";
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        // For CLV discs a timecode of 00:00:00.00 is valid, so technically a frame number of 0 is legal
        // (only for CLV discs, but it probably doesn't matter if we apply that to CAV too here)
        if (discMap.vbiFrameNumber(frameNumber) < 0) {
            qInfo() << "Verification failed - First failed frame was" << discMap.seqFrameNumber(frameNumber);
            discMap.debugFrameDetails(frameNumber);
            return false;
        }
    }
    qInfo() << "Verification successful";
    return true;
}

// Method to reorder frames according to VBI frame number order
void DiscMapper::reorderFrames(DiscMap &discMap)
{
    qInfo() << "Sorting the disc map into numerical frame order...";

    // Perform the sort
    discMap.sort();
    qInfo() << "Sorting complete";

    if (discMap.numberOfFrames() > 2) {
        // There is an edge case where the first VBI frame number can be corrupt and cause
        // a large gap between the first and second frames; we catch that edge case here
        qint32 frameNumber = 0;
        qint32 initialGap = discMap.vbiFrameNumber(frameNumber + 1) - discMap.vbiFrameNumber(frameNumber);

        if (initialGap > 1000) {
            qInfo() << "Warning: The gap between the first and second VBI number is" << initialGap;
            qInfo() << "this is over the 1000 frame threshold, so the first frame will be deleted to";
            qInfo() << "avoid generating a big gap.";

            discMap.setMarkedForDeletion(frameNumber);
            discMap.flush();
            qInfo() << "Removed first frame - disc map size now" << discMap.numberOfFrames() << "frames";
        }
    }
}

// Pad the disc map if there are missing frames in the disc map sequence
void DiscMapper::padDiscMap(DiscMap &discMap)
{
    qInfo() << "Looking for sequence gaps to pad in the disc map...";

    qint32 numberOfGaps = 0;
    qint32 totalMissingFrames = 0;
    qint32 clvOffsetFrames = 0;
    QVector<qint32> startFrame;
    QVector<qint32> paddingLength;
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames() - 1; frameNumber++) {
        if (discMap.vbiFrameNumber(frameNumber) + 1 != discMap.vbiFrameNumber(frameNumber + 1)) {
            // Is the current frame a pulldown?
            if (discMap.isPulldown(frameNumber)) {
                // Can't check anything for this condition, just skip it
            } else {
                // Is the next frame a pulldown?
                if (discMap.isPulldown(frameNumber + 1)) {
                    if (discMap.vbiFrameNumber(frameNumber) + 1 != discMap.vbiFrameNumber(frameNumber + 2)) {
                        if ((discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber)) != 0) {
                            qDebug() << "Sequence break over pulldown: Current VBI frame is" << discMap.vbiFrameNumber(frameNumber) <<
                                        "next frame (+1) is" << discMap.vbiFrameNumber(frameNumber + 2) << "gap of" <<
                                        discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber) << "frames";

                            numberOfGaps++;
                            qint32 missingFrames = discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber);
                            totalMissingFrames += missingFrames;

                            // Add to the gap list
                            startFrame.append(frameNumber);
                            paddingLength.append(missingFrames);
                        } else {
                            // Got a gap of 0????

                            // There seems to be an edge case here (issue 539) that caused a crash.  The case seems to be that
                            // there is a pulldown but the two frames on either side of the pull down have the same VBI frame
                            // number, resulting in a gap of 0.  Need to find a better test case for this so, for now, let's just
                            // quit with grace
                            if ((discMap.vbiFrameNumber(frameNumber + 2) - discMap.vbiFrameNumber(frameNumber)) == 0) {
                               qInfo() << "Analysis got a gap of 0 - this is a edge case reported in issue 539.  If you are";
                               qInfo() << "seeing this then you have a useful TBC that can be diagnosed... Please take the";
                               qInfo() << "to make your TBC file available to the developers so we can cure this bug.";
                            }
                        }
                    }
                } else {
                    // Check if this is a CLV IEC ammendment 2 timecode gap
                    if (!discMap.isClvOffset(frameNumber)) {
                        // Not a CLV offset frame
                        qint32 gapLength = discMap.vbiFrameNumber(frameNumber + 1) - discMap.vbiFrameNumber(frameNumber) - 1;
                        qDebug() << "Sequence break: Current VBI frame is" << discMap.vbiFrameNumber(frameNumber) <<
                                    "next frame is" << discMap.vbiFrameNumber(frameNumber + 1) << "gap of" <<
                                    gapLength << "frames";

                        // If the gap is long, use Info to warn the user than discmapping might have failed...
                        if (gapLength > 1000) {
                            qInfo() << "Warning: Detected a sequence break between VBI frame"  << discMap.vbiFrameNumber(frameNumber) << "and" <<
                                       "VBI frame" << discMap.vbiFrameNumber(frameNumber + 1) << "representing";
                            qInfo() << "a gap of" << gapLength << "frames.  This is over the threshold of 1000 frames and could indicate that mapping";
                            qInfo() << "has failed due to badly corrupted VBI frame number data in the source TBC file.";
                        }

                        numberOfGaps++;
                        qint32 missingFrames = discMap.vbiFrameNumber(frameNumber + 1) - discMap.vbiFrameNumber(frameNumber) - 1;
                        totalMissingFrames += missingFrames;

                        // Add to the gap list
                        startFrame.append(frameNumber);
                        paddingLength.append(missingFrames);
                    } else {
                        // CLV offset frame
                        clvOffsetFrames++;
                    }
                }
            }
        }
    }

    // Apply the padding
    for (qint32 i = 0; i < startFrame.size(); i++) {
        discMap.addPadding(startFrame[i], paddingLength[i]);
    }

    if (totalMissingFrames > 0) {
        // Sort the disc map to put the padding frames in the correct place
        discMap.sort();
    }

    // Report the result to the user
    if (numberOfGaps > 0) {
        if (numberOfGaps > 0) qInfo() << "Found" << numberOfGaps << "gaps representing" <<
                                         totalMissingFrames << "missing frames in the disc map";

        qInfo() << "Note: The disc map has been padded.  This means that there were missing frames";
        qInfo() << "that will be represented by black frames in the output video.";
    } else qInfo() << "No gaps found in the disc map";


    if (clvOffsetFrames > 0) qInfo() << "There were" << clvOffsetFrames << "CLV timecode offsets in the disc map";
    qInfo() << "After padding the disc map contains" << discMap.numberOfFrames() << "frames";
}

// Method to rewrite all frame numbers in the disc map when pulldown frames
// are present.  This ensures that all frames in the map (including the pulldown
// frames) have valid frame numbers
void DiscMapper::rewriteFrameNumbers(DiscMap &discMap)
{
    if (discMap.isDiscCav() && !discMap.isDiscPal()) {
        // Check if pulldown frames are present in the disc map
        qInfo() << "Searching disc map for pulldown frames...";
        bool present = false;
        for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
            if (discMap.isPulldown(frameNumber)) {
                present = true;
                break;
            }
        }

        if (present) qInfo() << "Search complete; pulldown frames are present.  Renumbering disc map...";
        else {
            qInfo() << "Search complete; no pulldown frames present";
            return;
        }

        qint32 newVbi = discMap.vbiFrameNumber(0);
        for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
            discMap.setVbiFrameNumber(frameNumber, newVbi);
            newVbi++;
        }

        qInfo() << "Renumbering complete";
    }
}

// Method to delete any unmappable frames from the disc map
void DiscMapper::deleteUnmappableFrames(DiscMap &discMap)
{
    qInfo() << "Deleting unmappable frames from the disc map...";
    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        // For CLV discs a timecode of 00:00:00.00 is valid, so technically a frame number of 0 is legal
        // (only for CLV discs, but it probably doesn't matter if we apply that to CAV too here)
        if (discMap.vbiFrameNumber(frameNumber) < 0 && !discMap.isPulldown(frameNumber)) {
            qDebug() << "Marking frame" << frameNumber << "for deletion (unmappable)";
            discMap.setMarkedForDeletion(frameNumber);
        }
    }

    // Flush the frames marked for deletion
    discMap.flush();

    qInfo() << "Deletion successful";
}

// Method to save the current disc map
bool DiscMapper::saveDiscMap(DiscMap &discMap)
{
    // Open the input video file
    SourceVideo sourceVideo;
    sourceVideo.open(inputFileInfo.filePath(), discMap.getVideoFieldLength());

    // Open the output video file
    QFile targetVideo(outputFileInfo.filePath());
    if (!targetVideo.open(QIODevice::WriteOnly)) {
        // Could not open target video file
        qInfo() << "Cannot open target video file:" << outputFileInfo.filePath();
        sourceVideo.close();
        return false;
    }

    // Initialise the input audio file
    SourceAudio sourceAudio;
    QFile targetAudio;

    if (!noAudio) {
        // Open the input audio file
        if (!sourceAudio.open(inputFileInfo)) {
            // Could not open input audio file
            qInfo() << "Cannot open source audio file:" << inputFileInfo.absolutePath() + "/" + inputFileInfo.completeBaseName() + ".pcm";
            sourceVideo.close();
            sourceAudio.close();
            return false;
        }

        // Open the output audio file
        targetAudio.setFileName(outputFileInfo.absolutePath() + "/" + outputFileInfo.completeBaseName() + ".pcm");
        if (targetAudio.exists()) {
            qInfo() << "Target audio file already exists:" << targetAudio.fileName() << "- Cannot proceed!";
            sourceVideo.close();
            sourceAudio.close();
            return false;
        }
        if (!targetAudio.open(QIODevice::WriteOnly)) {
            // Could not open target audio file
            qInfo() << "Cannot open target audio file:" << targetAudio.fileName();
            sourceVideo.close();
            sourceAudio.close();
            return false;
        }
    }

    // Make a dummy video field to use when outputting padded frames
    SourceVideo::Data missingFieldData;
    missingFieldData.fill(0, discMap.getVideoFieldLength());

    // Make a dummy audio field to use when outputting padded frames
    SourceAudio::Data missingFieldAudioData;
    missingFieldAudioData.fill(0, discMap.getApproximateAudioFieldLength());

    // Create the output video file
    SourceVideo::Data sourceFirstField;
    SourceVideo::Data sourceSecondField;
    SourceAudio::Data sourceAudioFirstField;
    SourceAudio::Data sourceAudioSecondField;

    qInfo() << "Saving target video frames...";
    qint32 notifyInterval = discMap.numberOfFrames() / 50;
    if (notifyInterval < 1) notifyInterval = 1;

    for (qint32 frameNumber = 0; frameNumber < discMap.numberOfFrames(); frameNumber++) {
        bool writeFail = false;

        // Is the current frameNumber a real frame or a padded frame?
        if (!discMap.isPadded(frameNumber)) {
            // Real frame
            qint32 firstFieldNumber = discMap.getFirstFieldNumber(frameNumber);
            qint32 secondFieldNumber = discMap.getSecondFieldNumber(frameNumber);
            sourceFirstField = sourceVideo.getVideoField(firstFieldNumber);
            sourceSecondField = sourceVideo.getVideoField(secondFieldNumber);

            // Write the fields into the output TBC file in the same order as the source file
            if (firstFieldNumber < secondFieldNumber) {
                // Save the first field and then second field to the output file
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceFirstField.data()),
                                       sourceFirstField.size() * 2)) writeFail = true;
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceSecondField.data()),
                                       sourceSecondField.size() * 2)) writeFail = true;
            } else {
                // Save the second field and then first field to the output file
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceSecondField.data()),
                                       sourceSecondField.size() * 2)) writeFail = true;
                if (!targetVideo.write(reinterpret_cast<const char *>(sourceFirstField.data()),
                                       sourceFirstField.size() * 2)) writeFail = true;
            }

            // Save the audio (not field order dependent)
            if (!noAudio) {
                // Ensure there is audio to read from the first and second fields
                if ((discMap.getFirstFieldAudioDataLength(frameNumber) > 0) &&
                        (discMap.getSecondFieldAudioDataLength(frameNumber) > 0)) {
                    // Read the audio
                    sourceAudioFirstField = sourceAudio.getAudioData(discMap.getFirstFieldAudioDataStart(frameNumber),
                                                                     discMap.getFirstFieldAudioDataLength(frameNumber));
                    sourceAudioSecondField = sourceAudio.getAudioData(discMap.getSecondFieldAudioDataStart(frameNumber),
                                                                      discMap.getSecondFieldAudioDataLength(frameNumber));

                    // Write the audio
                    if (!targetAudio.write(reinterpret_cast<const char *>(sourceAudioFirstField.data()),
                                           sourceAudioFirstField.size() * 2)) writeFail = true;
                    if (!targetAudio.write(reinterpret_cast<const char *>(sourceAudioSecondField.data()),
                                           sourceAudioSecondField.size() * 2)) writeFail = true;
                } else {
                    if (discMap.getFirstFieldAudioDataLength(frameNumber) < 1) {
                        qInfo() << "Warning: Input file seems to have zero audio data in the first field of frame number #" << frameNumber;
                        qInfo() << "The audio output might be corrupt.";
                    }
                    if (discMap.getSecondFieldAudioDataLength(frameNumber) < 1) {
                        qInfo() << "Warning: Input file seems to have zero audio data in the second field of frame number #" << frameNumber;
                        qInfo() << "The audio output might be corrupt.";
                    }
                }
            }
        } else {
            // Padded frame - write two dummy fields
            if (!targetVideo.write(reinterpret_cast<const char *>(missingFieldData.data()),
                                   missingFieldData.size() * 2)) writeFail = true;
            if (!targetVideo.write(reinterpret_cast<const char *>(missingFieldData.data()),
                                   missingFieldData.size() * 2)) writeFail = true;

            if (!noAudio) {
                // Write the padded audio
                if (!targetAudio.write(reinterpret_cast<const char *>(missingFieldAudioData.data()),
                                       missingFieldAudioData.size() * 2)) writeFail = true;
                if (!targetAudio.write(reinterpret_cast<const char *>(missingFieldAudioData.data()),
                                       missingFieldAudioData.size() * 2)) writeFail = true;
            }
        }

        // Notify user
        if (frameNumber % notifyInterval == 0) {
            qInfo() << "Written frame" << frameNumber << "of" << discMap.numberOfFrames();
        }

        // Was the write successful?
        if (writeFail) {
            // Could not write to target TBC file
            qWarning() << "Writing fields to the target TBC file failed on frame number" << frameNumber;
            targetVideo.close();
            sourceVideo.close();
            return false;
        }
    }
    qInfo() << discMap.numberOfFrames() << "video frames saved";

    // Close the source and target video files
    targetVideo.close();
    sourceVideo.close();

    // Close the source and target audio files
    if (!noAudio) {
        qInfo() << "Target audio frames saved";
        targetAudio.close();
        sourceAudio.close();
    }

    // Now save the metadata
    qInfo() << "Saving target video metadata...";
    QFileInfo outputMetadataFileInfo(outputFileInfo.filePath() + ".json");
    if (!discMap.saveTargetMetadata(outputMetadataFileInfo)) {
        qInfo() << "Writing target metadata failed!";
        return false;
    }
    qInfo() << "Target video metadata saved";
    return true;
}











