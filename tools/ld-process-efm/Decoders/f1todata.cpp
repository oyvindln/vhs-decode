/************************************************************************

    f1todata.cpp

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

#include "f1todata.h"

F1ToData::F1ToData()
{
    debugOn = false;
    reset();

    // Set the sector sync pattern
    syncPattern.clear();
    syncPattern.append(static_cast<char>(0x00));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0xFF));
    syncPattern.append(static_cast<char>(0x00));
}

// Public methods -----------------------------------------------------------------------------------------------------

// Method to feed the sector processing state-machine with F1 frames
QByteArray F1ToData::process(const std::vector<F1Frame> &f1FramesIn, bool debugState)
{
    debugOn = debugState;

    // Clear the output buffer
    dataOutputBuffer.clear();

    if (f1FramesIn.empty()) return dataOutputBuffer;

    // Append input data to the processing buffer
    for (const F1Frame &f1Frame: f1FramesIn) {
        f1DataBuffer.append(reinterpret_cast<const char *>(f1Frame.getDataSymbols()), 24);

        // Each validity flag covers 24 bytes of data symbols
        for (qint32 p = 0; p < 24; p++) {
            f1IsCorruptBuffer.append(f1Frame.isCorrupt());
            f1IsMissingBuffer.append(f1Frame.isMissing());
        }
    }

    waitingForData = false;
    while (!waitingForData) {
        currentState = nextState;

        switch (currentState) {
        case state_initial:
            nextState = sm_state_initial();
            break;
        case state_getInitialSync:
            nextState = sm_state_getInitialSync();
            break;
        case state_getNextSync:
            nextState = sm_state_getNextSync();
            break;
        case state_processFrame:
            nextState = sm_state_processFrame();
            break;
        case state_noSync:
            nextState = sm_state_noSync();
            break;
        }
    }

    return dataOutputBuffer;
}

// Get method - retrieve statistics
const F1ToData::Statistics &F1ToData::getStatistics() const
{
    return statistics;
}

// Method to report decoding statistics to qInfo
void F1ToData::reportStatistics() const
{
    qInfo()           << "";
    qInfo()           << "F1 Frames to Data:";
    qInfo()           << "         Valid sectors:" << statistics.validSectors;
    qInfo()           << "       Invalid sectors:" << statistics.invalidSectors;
    qInfo()           << "       Missing sectors:" << statistics.missingSectors;
    qInfo()           << "         Total sectors:" << statistics.totalSectors;

    qInfo()           << "";
    qInfo()           << "  Sectors missing sync:" << statistics.missingSync;

    qInfo()           << "";
    qInfo().noquote() << "         Start address:" << statistics.startAddress.getTimeAsQString();
    qInfo().noquote() << "       Current address:" << statistics.currentAddress.getTimeAsQString();
}

// Reset the object
void F1ToData::reset()
{
    f1DataBuffer.clear();
    f1IsCorruptBuffer.clear();
    f1IsMissingBuffer.clear();
    dataOutputBuffer.clear();

    waitingForData = false;
    currentState = state_initial;
    nextState = currentState;

    missingSyncCount = 0;

    clearStatistics();
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to clear the statistics counters
void F1ToData::clearStatistics()
{
    statistics.validSectors = 0;
    statistics.invalidSectors = 0;
    statistics.missingSectors = 0;
    statistics.totalSectors = 0;
    statistics.missingSync = 0;

    statistics.startAddress.setTime(0, 0, 0);
    statistics.currentAddress.setTime(0, 0, 0);
}

// State-machine methods ----------------------------------------------------------------------------------------------

F1ToData::StateMachine F1ToData::sm_state_initial()
{
    if (debugOn) qDebug() << "F1ToData::sm_state_initial(): Called";

    // Set initial disc time to 00:00.00
    statistics.startAddress.setTime(0, 0, 0);
    lastAddress.setTime(0, 0, 0);

    return state_getInitialSync;
}

// Find the initial sector sync pattern
F1ToData::StateMachine F1ToData::sm_state_getInitialSync()
{
    // Look for the sector sync pattern in the F1 frame data
    qint32 syncPosition = f1DataBuffer.indexOf(syncPattern);

    // Was a sync pattern found?
    if (syncPosition == -1) {
        // No sync found
        f1DataBuffer.clear();
        f1IsCorruptBuffer.clear();
        f1IsMissingBuffer.clear();
        waitingForData = true;
        //if (debugOn) qDebug() << "F1ToData::sm_state_getInitialSync(): No sync found";
        return state_getInitialSync;
    }

    if (debugOn) qDebug() << "F1ToData::sm_state_getInitialSync(): Initial sync found at position" << syncPosition;
    f1DataBuffer.remove(0, syncPosition);
    return state_processFrame;
}

// Find the next sector sync pattern
F1ToData::StateMachine F1ToData::sm_state_getNextSync()
{
    // Ensure we have enough data to detect a sync
    if (f1DataBuffer.size() < 12) {
        // We need more data
        waitingForData = true;
        return state_getNextSync;
    }

    // Once the initial sync is found and the buffer is aligned, the sync should always
    // be at the start of the input buffer
    qint32 syncPosition = f1DataBuffer.indexOf(syncPattern);
    if (syncPosition != 0) {
        // Sector has no sync pattern
        return state_noSync;
    }

    return state_processFrame;
}

// Process a sector into data
F1ToData::StateMachine F1ToData::sm_state_processFrame()
{
    // Ensure we have enough data to process an entire sector
    if (f1DataBuffer.size() < 2352) {
        // We need more data
        waitingForData = true;
        return state_processFrame;
    }

    // Create a sector object from the sector data
    bool sectorValidity = true;
    bool sectorBufferCorrupt = false;
    bool sectorBufferMissing = false;
    for (qint32 i = 0; i < 2352; i++) {
        if (f1IsCorruptBuffer[i] == static_cast<char>(0)) sectorBufferCorrupt = true;
        if (f1IsMissingBuffer[i] == static_cast<char>(0)) sectorBufferMissing = true;
        if (sectorBufferCorrupt || sectorBufferMissing) break;
    }
    if (sectorBufferCorrupt || sectorBufferMissing) sectorValidity = false;
    Sector sector(f1DataBuffer.mid(0, 2352), sectorValidity);

    // Remove the sector data from the input F1 buffer
    f1DataBuffer.remove(0, 2352);
    f1IsCorruptBuffer.remove(0, 2352);
    f1IsMissingBuffer.remove(0, 2352);

    // Verify the sector is valid
    if (!sector.isValid()) {
        // Sector is not valid, set to zero and force address
        if (debugOn) qDebug() << "F1ToData::sm_state_processFrame(): Current frame is invalid, setting user data to null";
        lastAddress.addFrames(1);
        statistics.currentAddress = lastAddress;
        sector.setAsNull(statistics.currentAddress);

        if (debugOn && sectorBufferCorrupt) qDebug() << "F1ToData::sm_state_processFrame(): Sector invalid - Buffer contained corrupt F1 data";
        if (debugOn && sectorBufferMissing) qDebug() << "F1ToData::sm_state_processFrame(): Sector invalid - Buffer contained missing F1 data (padded)";

        statistics.invalidSectors++;
        statistics.totalSectors++;
    } else {
        // Sector is valid
        statistics.currentAddress = sector.getAddress();
        statistics.validSectors++;
        statistics.totalSectors++;
    }

    // Sector will now have a valid address, check for gaps and pad if required, then write the current sector
    qint32 sectorAddressGap = sector.getAddress().getDifference(lastAddress.getTime());
    if (sectorAddressGap > 1) {
        if (debugOn) qDebug().noquote() << "F1ToData::sm_state_processFrame(): Sector address gap - Adding" <<
                                           sectorAddressGap - 1 << "sectors(s) of padding - Last sector address was" <<
                                           lastAddress.getTimeAsQString() <<
                                           " - current sector address is" << sector.getAddress().getTimeAsQString();

        // If we're not at the start of the disc, add one to avoid writing the same
        // address twice
        if (lastAddress.getFrames() != 0) {
            lastAddress.addFrames(1);
            sectorAddressGap--;
        }

        Sector paddingSector;
        for (qint32 p = 0; p < sectorAddressGap; p++) {
            paddingSector.setAsNull(lastAddress);

            dataOutputBuffer.append(paddingSector.getUserData());
            //if (debugOn) qDebug().noquote() << "F1ToData::sm_state_processFrame(): Padding sector with address" << lastAddress.getTimeAsQString();

            lastAddress.addFrames(1);

            statistics.missingSectors++;
            statistics.totalSectors++;
        }
    }

    // Write out the new sector
    dataOutputBuffer.append(sector.getUserData());
    lastAddress = statistics.currentAddress;
    //if (debugOn) qDebug().noquote() << "F1ToData::sm_state_processFrame(): Writing data sector with address" << statistics.currentAddress.getTimeAsQString();

    return state_getNextSync;
}

// Sector sync has been lost
F1ToData::StateMachine F1ToData::sm_state_noSync()
{
    statistics.missingSync++;

    // The current sector has no sync.  Here we need to determine if the sector is corrupt, or if it's just missing
    // (due to a gap in the EFM rather than errors in the EFM)

    // If the f1DataBuffer is less than the minium sector size, we have to resize it to avoid
    // unwanted errors
    if (f1DataBuffer.size() < 2352) f1DataBuffer.resize(2352);
    Sector sector(f1DataBuffer.mid(0, 2352), true);

    if (sector.isMissing()) {
        if (debugOn) qDebug() << "F1ToData::sm_state_syncLost(): Sector sync has been lost and sector looks like it's missing.  Hunting for next valid sync";

        // Remove the sector data from the input F1 buffer
        f1DataBuffer.remove(0, 2352);
        f1IsCorruptBuffer.remove(0, 2352);
        f1IsMissingBuffer.remove(0, 2352);

        return state_getInitialSync;
    }

    if (debugOn) qDebug() << "F1ToData::sm_state_syncLost(): Sector is missing sync pattern, but looks like it should be a valid sector - continuing";
    return state_processFrame;
}
