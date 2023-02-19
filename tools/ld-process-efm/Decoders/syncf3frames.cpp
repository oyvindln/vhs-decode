/************************************************************************

    syncf3frames.cpp

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

#include "syncf3frames.h"

// Note: This class ensures that the F3 Frame output is synchronised with
// the subcode sections.  This is required for audio processing as, without
// the subcode metadata, it's not possible to resync audio data if the input
// data is corrupt.  This wasn't an issue on real players, as the play back
// would just "start again" - however, here the audio must keep sync with the
// video output from ld-decode, so the sample gaps caused by corruption must
// be replaced with exact 'gaps' that can only be calculated if we keep the
// subcode metadata and F3 frames synchronised throughout the decoding process.
//
// This sync isn't required for data-only EFM (as the metadata and sectors
// are contained in the same stream of data).

SyncF3Frames::SyncF3Frames()
{
    debugOn = false;
    reset();
}

// Public methods -----------------------------------------------------------------------------------------------------

// Main processing method
const std::vector<F3Frame> &SyncF3Frames::process(const std::vector<F3Frame> &f3FramesIn, bool debugState)
{
    debugOn = debugState;

    // Clear the output buffer
    f3FramesOut.clear();

    if (f3FramesIn.empty()) return f3FramesOut;

    // Append input data to the processing buffer
    statistics.totalF3Frames += f3FramesIn.size();
    f3FrameBuffer.insert(f3FrameBuffer.end(), f3FramesIn.begin(), f3FramesIn.end());

    waitingForData = false;
    while (!waitingForData) {
        currentState = nextState;

        switch (currentState) {
        case state_initial:
            nextState = sm_state_initial();
            break;
        case state_findInitialSync0:
            nextState = sm_state_findInitialSync0();
            break;
        case state_findNextSync:
            nextState = sm_state_findNextSync();
            break;
        case state_syncRecovery:
            nextState = sm_state_syncRecovery();
            break;
        case state_syncLost:
            nextState = sm_state_syncLost();
            break;
        case state_processSection:
            nextState = sm_state_processSection();
            break;
        }
    }

    return f3FramesOut;
}

// Get method - retrieve statistics
const SyncF3Frames::Statistics &SyncF3Frames::getStatistics() const
{
    return statistics;
}

// Method to report decoding statistics to qInfo
void SyncF3Frames::reportStatistics() const
{
    qInfo() << "";
    qInfo() << "F3 Frame synchronisation:";
    qInfo() << "   Total input F3 Frames:" << statistics.totalF3Frames;
    qInfo() << "        Discarded Frames:" << statistics.discardedFrames;
    qInfo() << "    Total valid sections:" << statistics.totalSections << "(" << statistics.totalSections * 98 << "F3 Frames )";
}

// Method to reset the class
void SyncF3Frames::reset()
{
    // Initialise the state-machine
    f3FrameBuffer.clear();
    currentState = state_initial;
    nextState = currentState;
    waitingForData = false;
    syncRecoveryAttempts = 0;

    clearStatistics();
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to clear the statistics counters
void SyncF3Frames::clearStatistics()
{
    statistics.totalF3Frames = 0;
    statistics.discardedFrames = 0;
    statistics.totalSections = 0;
}

// Processing state machine methods -----------------------------------------------------------------------------------

// Initial state machine state
SyncF3Frames::StateMachine SyncF3Frames::sm_state_initial()
{
    if (debugOn) qDebug() << "SyncF3Frames::sm_state_initial(): Called";
    return state_findInitialSync0;
}

// Find initial subcode sync
SyncF3Frames::StateMachine SyncF3Frames::sm_state_findInitialSync0()
{
    //if (debugOn) qDebug() << "SyncF3Frames::sm_state_findInitialSync0(): Called";

    qint32 i = 0;
    const qint32 bufferSize = static_cast<qint32>(f3FrameBuffer.size());
    for (i = 0; i < bufferSize - 1; i++) {
        if (f3FrameBuffer[i].isSubcodeSync0() || f3FrameBuffer[i+1].isSubcodeSync1()) break;
    }

    // Did we find a sync0 or sync1?
    if (!f3FrameBuffer[i].isSubcodeSync0() && !f3FrameBuffer[i+1].isSubcodeSync1()) {
        // Not found
        statistics.discardedFrames += f3FrameBuffer.size();

        if (debugOn) qDebug() << "SyncF3Frames::sm_state_findInitialSync0(): No initial sync0 found in buffer - discarding" << f3FrameBuffer.size() << "frames";
        waitingForData = true;

        f3FrameBuffer.clear();
        return state_findInitialSync0;
    } else {
        // Found, discard frames up to initial sync
        f3FrameBuffer.erase(f3FrameBuffer.begin(), f3FrameBuffer.begin() + i);
        statistics.discardedFrames += i;
        if (debugOn) qDebug() << "SyncF3Frames::sm_state_findInitialSync0(): Found initial sync0 - discarding" << i << "frames";
    }

    return state_findNextSync;
}

// Find next subcode sync
SyncF3Frames::StateMachine SyncF3Frames::sm_state_findNextSync()
{
    // Ensure we have enough data
    if (f3FrameBuffer.size() < 99) {
        waitingForData = true;
        return state_findNextSync;
    }

    // If we identify the end of the section, process it
    if (f3FrameBuffer[98].isSubcodeSync0()) {
        return state_processSection;
    }

    // Sync0 was missing... look for sync1
    if (f3FrameBuffer[99].isSubcodeSync1()) {
        return state_processSection;
    }

    // Sync is missing, attempt recovery
    if (debugOn) qDebug() << "SyncF3Frames::sm_state_syncRecovery(): F3 subcode sync0 and sync1 missing";
    syncRecoveryAttempts = 0;
    return state_syncRecovery;
}

// Subcode sync recovery state
SyncF3Frames::StateMachine SyncF3Frames::sm_state_syncRecovery()
{
    // Sync0 and sync 1 are missing; so we need to look ahead over another
    // section to see if a sync is present.  If it is, then it's very likely
    // the missing section sync is simple corruption, so we can assume its
    // position.  If 5 sets of sync0 and sync1 are missing in a row, its
    // likely that the EFM signal is simply invalid, so we flag lost sync

    qint32 requiredF3Frames = 98 * (syncRecoveryAttempts + 2);

    // Ensure we have enough data to see the next section
    if (static_cast<qint32>(f3FrameBuffer.size()) < (requiredF3Frames + 2)) {
        waitingForData = true;
        return state_syncRecovery;
    }

    // This section's sync0 should be at 98 (with sync1 at 99),
    // the next section's sync0 should be at 98+98 = 196
    bool nextSectionSyncFound = false;

    // If we identify the end of the section, process it
    if (f3FrameBuffer[98 + (syncRecoveryAttempts * 98)].isSubcodeSync0()) {
        nextSectionSyncFound = true;
    }

    // Sync0 was missing... look for sync1
    if (f3FrameBuffer[99 + (syncRecoveryAttempts * 98)].isSubcodeSync1()) {
        nextSectionSyncFound = true;
    }

    if (nextSectionSyncFound) {
        if (debugOn) qDebug() << "SyncF3Frames::sm_state_syncRecovery(): Lost sync recovered on attempt" << syncRecoveryAttempts;
        syncRecoveryAttempts = 0;
        return state_processSection;
    }

    // Give up - make another attempt?
    if (debugOn) qDebug() << "SyncF3Frames::sm_state_syncRecovery(): Failed to find sync on attempt" << syncRecoveryAttempts;

    // Try recovery 5 times...
    syncRecoveryAttempts++;
    if (syncRecoveryAttempts > 5) {
        // Too many attempts
        if (debugOn) qDebug() << "SyncF3Frames::sm_state_syncRecovery(): Too many sync recovery attempts (" << syncRecoveryAttempts - 1 << ") - giving up";
        syncRecoveryAttempts = 0;
        return state_syncLost;
    }

    return state_syncRecovery;
}

// Subcode sync lost state
SyncF3Frames::StateMachine SyncF3Frames::sm_state_syncLost()
{
    if (debugOn) qDebug() << "SyncF3Frames::sm_state_syncLost(): Called";

    // We have lost sync; clear the buffer and go back to looking for an initial sync
    f3FrameBuffer.erase(f3FrameBuffer.begin(), f3FrameBuffer.begin() + 98);
    statistics.discardedFrames += 98;
    if (debugOn) qDebug() << "SyncF3Frames::sm_state_findNextSync(): Sync lost! - discarding 98 frames";

    if (f3FrameBuffer.size() < 98) {
        waitingForData = true;
    }

    return state_findInitialSync0;
}

// Process completed F3 frame
SyncF3Frames::StateMachine SyncF3Frames::sm_state_processSection()
{
    //if (debugOn) qDebug() << "SyncF3Frames::sm_state_processSection(): Called";

    // Write the complete section of 98 F3 frames to the output buffer
    f3FramesOut.insert(f3FramesOut.end(), f3FrameBuffer.begin(), f3FrameBuffer.begin() + 98);
    statistics.totalSections++;

    // Remove the processed section from the F3 frame buffer
    f3FrameBuffer.erase(f3FrameBuffer.begin(), f3FrameBuffer.begin() + 98);

    return state_findNextSync;
}
