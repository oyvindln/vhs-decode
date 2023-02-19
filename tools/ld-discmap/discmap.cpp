/************************************************************************

    discmap.cpp

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

#include "discmap.h"

DiscMap::DiscMap(const QFileInfo &metadataFileInfo, const bool reverseFieldOrder,
                 const bool noStrict)
            : m_metadataFileInfo(metadataFileInfo), m_reverseFieldOrder(reverseFieldOrder),
              m_noStrict(noStrict)
{
    m_tbcValid = true;
    ldDecodeMetaData = new LdDecodeMetaData;

    // Open the TBC metadata file
    if (!ldDecodeMetaData->read(metadataFileInfo.filePath())) {
        // Open failed
        qDebug() << "Cannot load JSON metadata from" << metadataFileInfo.filePath();
        m_tbcValid = false;
        return;
    }

    // If source is reverse-field order, set it up
    if (m_reverseFieldOrder) ldDecodeMetaData->setIsFirstFieldFirst(false);
    else ldDecodeMetaData->setIsFirstFieldFirst(true);

    // Get the number of available frames
    m_numberOfFrames = ldDecodeMetaData->getNumberOfFrames();

    if (m_numberOfFrames < 2) {
        qDebug() << "JSON metadata contains only" << m_numberOfFrames << "frames - too small";
        m_tbcValid = false;
        return;
    }

    if (m_numberOfFrames > 108000) {
        qDebug() << "JSON metadata contains" << m_numberOfFrames << "frames - too big";
        m_tbcValid = false;
        return;
    }

    // Set the video field length
    m_videoFieldLength = ldDecodeMetaData->getVideoParameters().fieldWidth *
            ldDecodeMetaData->getVideoParameters().fieldHeight;

    // Resize the frame store
    m_frames.resize(m_numberOfFrames);

    // Decode the VBI information for the TBC and initialise the frame object
    VbiDecoder vbiDecoder;
    QVector<VbiDecoder::Vbi> vbiData(m_numberOfFrames);
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        // Store the original sequential frame number and the fields
        m_frames[frameNumber].seqFrameNumber(frameNumber + 1);
        m_frames[frameNumber].firstField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1));
        m_frames[frameNumber].secondField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1));

        // Get the VBI data and then decode (frames are indexed from 1)
        auto vbi1 = ldDecodeMetaData->getFieldVbi(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).vbiData;
        auto vbi2 = ldDecodeMetaData->getFieldVbi(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).vbiData;
        vbiData[frameNumber] = vbiDecoder.decodeFrame(vbi1[0], vbi1[1], vbi1[2], vbi2[0], vbi2[1], vbi2[2]);

        if (vbiData[frameNumber].leadIn || vbiData[frameNumber].leadOut) m_frames[frameNumber].isLeadInOrOut(true);
        else m_frames[frameNumber].isLeadInOrOut(false);
    }

    // Get the source format (PAL/NTSC)
    m_videoSystemDescription = ldDecodeMetaData->getVideoSystemDescription();
    if (ldDecodeMetaData->getVideoParameters().system == PAL) m_isDiscPal = true;
    else if (ldDecodeMetaData->getVideoParameters().system == NTSC) m_isDiscPal = false;
    else {
        qDebug() << "Input TBC video system" << m_videoSystemDescription << "is not supported";
        qCritical("Video system must be PAL or NTSC");
    }

    // Set the audio field length
    if (m_isDiscPal) {
        // Disc is PAL:
        // 44,100 samples per second
        // 50 fields per second
        // 44,100 / 50 = 882 * 2 = 1764
        // L/R channels = 1764 * 2 =
        m_audioFieldByteLength = 3528;
        m_audioFieldSampleLength = 882;
    } else {
        // Disc is NTSC:
        // 44,100 samples per second
        // 60000/1001 fields per second
        // 44100 / (60000/1001) = 735.735 * 2 = 1472
        // L/R channels = 1472 * 2 =
        m_audioFieldByteLength = 2944;
        m_audioFieldSampleLength = 736;
    }

    // Get the disc type (CAV/CLV)
    qint32 framesToCheck = 100;
    if (m_numberOfFrames < framesToCheck) framesToCheck = m_numberOfFrames;
    qDebug() << "Checking first" << framesToCheck << "sequential frames for disc CAV/CLV type determination";

    qint32 cavCount = 0;
    qint32 clvCount = 0;
    // Count how many frames are marked as CAV or CLV in the metadata
    for (qint32 frameNumber = 0; frameNumber < framesToCheck; frameNumber++) {
        // Look for a complete, valid CAV picture number or CLV time-code
        if (vbiData[frameNumber].picNo > 0) cavCount++;
        if (vbiData[frameNumber].clvHr != -1 && vbiData[frameNumber].clvMin != -1 &&
                vbiData[frameNumber].clvSec != -1 && vbiData[frameNumber].clvPicNo != -1) clvCount++;
    }

    // If the metadata has no picture numbers or time-codes, we cannot use the source
    if (cavCount == 0 && clvCount == 0) {
        qDebug() << "Source does not seem to contain valid CAV picture numbers or CLV time-codes - cannot map";
        m_tbcValid = false;
        return;
    }

    // Determine disc type
    if (cavCount > clvCount) {
        m_isDiscCav = true;
        m_discType = "CAV";
        qDebug() << "Got" << cavCount << "valid CAV picture numbers from" << framesToCheck << "frames - source disc type is CAV";
    } else {
        m_isDiscCav = false;
        m_discType = "CLV";
        qDebug() << "Got" << clvCount << "valid CLV picture numbers from" << framesToCheck << "frames - source disc type is CLV";
    }

    // If the disc type is CLV, convert the timecodes into frame numbers and update the stored VBI
    // otherwise just store the CAV picture numbers
    if (m_isDiscCav)  qDebug() << "Storing VBI CAV picture numbers as frame numbers";
    else qDebug() << "Converting VBI CLV timecodes into frame numbers";
    qint32 iecOffset = -1;
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        if (!m_isDiscCav) {
            // Attempt to translate the CLV timecode into a frame number
            LdDecodeMetaData::ClvTimecode clvTimecode;
            clvTimecode.hours = vbiData[frameNumber].clvHr;
            clvTimecode.minutes = vbiData[frameNumber].clvMin;
            clvTimecode.seconds = vbiData[frameNumber].clvSec;
            clvTimecode.pictureNumber = vbiData[frameNumber].clvPicNo;
            m_frames[frameNumber].vbiFrameNumber(ldDecodeMetaData->convertClvTimecodeToFrameNumber(clvTimecode));

            // Check for CLV timecode offset frame (actually, this marks the frame
            // that preceeds the jump)
            // There will be a one frame time-code jump after each frame marked
            // by this check
            if (!m_isDiscPal) {
                if (isNtscAmendment2ClvFrameNumber(m_frames[frameNumber].vbiFrameNumber() - iecOffset)) {
                    m_frames[frameNumber].isClvOffset(true);
                    iecOffset++;
                    //qDebug() << "CLV offset set for frame" << m_frames[frameNumber].seqFrameNumber() << "with VBI of" << m_frames[frameNumber].vbiFrameNumber();
                }
            }
        } else {
            m_frames[frameNumber].vbiFrameNumber(vbiData[frameNumber].picNo);
        }
    }

    // Check for the presence of pull-down frames (if NTSC CAV)
    m_numberOfPulldowns = false;
    if (!m_isDiscPal && m_isDiscCav) {
        qDebug() << "Disc type is NTSC CAV - checking for pull-down frames";

        for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
            bool isPulldown = false;

            // Does the current frame have a frame number (and is not lead in/out)?
            if (m_frames[frameNumber].vbiFrameNumber() == -1 && !m_frames[frameNumber].isLeadInOrOut()) {
                // Get the phaseID of the preceeding frame (with underflow protection)
                qint32 lastPhase2 = -1;
                if (frameNumber > 0) lastPhase2 = ldDecodeMetaData->getField(
                            ldDecodeMetaData->getSecondFieldNumber(frameNumber)).fieldPhaseID; // -1

                // Get the phaseID of the current frame
                qint32 currentPhase1 = ldDecodeMetaData->getField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).fieldPhaseID;
                qint32 currentPhase2 = ldDecodeMetaData->getField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).fieldPhaseID;

                // Get the phaseID of the following frame (with overflow protection)
                qint32 nextPhase1 = -1;
                if (frameNumber < m_numberOfFrames - 1) nextPhase1 = ldDecodeMetaData->getField(
                            ldDecodeMetaData->getFirstFieldNumber(frameNumber + 2)).fieldPhaseID; // +1

                // Work out what the preceeding phase is expected to be
                qint32 expectedLastPhase;
                qint32 expectedNextPhase;
                qint32 expectedIntraPhase;

                if (!m_reverseFieldOrder) {
                    // Normal field order
                    expectedLastPhase = currentPhase1 - 1;
                    if (expectedLastPhase == 0) expectedLastPhase = 4;

                    // Work out what the subsequent phase is expected to be
                    expectedNextPhase = currentPhase2 + 1;
                    if (expectedNextPhase == 5) expectedNextPhase = 1;

                    // Work out what the infra-frame phase is expected to be
                    expectedIntraPhase = currentPhase1 + 1;
                    if (expectedIntraPhase == 5) expectedIntraPhase = 1;
                } else {
                    // Reversed field order
                    expectedLastPhase = currentPhase1 + 1;
                    if (expectedLastPhase == 5) expectedLastPhase = 1;

                    // Work out what the subsequent phase is expected to be
                    expectedNextPhase = currentPhase2 - 1;
                    if (expectedNextPhase == 0) expectedNextPhase = 4;

                    // Work out what the infra-frame phase is expected to be
                    expectedIntraPhase = currentPhase1 - 1;
                    if (expectedIntraPhase == 0) expectedIntraPhase = 4;
                }

                // Now confirm everything is sane
                if (currentPhase2 == expectedIntraPhase) {
                    if (lastPhase2 == expectedLastPhase || lastPhase2 == -1) {
                         if (nextPhase1 == expectedNextPhase || nextPhase1 == -1) {
                             // It's very likely this is a pulldown frame
                             isPulldown = true;
                         } else {
                             // Probably not a pull-down frame
                             qDebug() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() << "is not in phase sequence with the subsequent frame!";
                         }
                    } else {
                        // Probably not a pull-down frame
                        qDebug() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() << "is not in phase sequence with the preceeding frame!";
                    }
                } else {
                    // Probably not a pull-down frame
                    qDebug() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() << "has an incorrect intra-frame phaseID!";
                }

                if (isPulldown) {
                    // If we have a possible pull-down, perform an additional check based on the VBI numbering
                    // If it's really a pull-down, then the VBI frame numbers should be missing 5 frames before
                    // and after the current frame...
                    qint32 doubleCheckCounter = 0;
                    if (frameNumber > 5) {
                        if (vbiData[frameNumber - 5].picNo == -1) doubleCheckCounter++;
                    }
                    if (frameNumber < m_numberOfFrames - 5) {
                        if (vbiData[frameNumber + 5].picNo == -1) doubleCheckCounter++;
                    }

                    if (doubleCheckCounter < 1) {
                        if (!m_noStrict) {
                            qDebug() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() <<
                                        "looks like a pull-down, but there is no pull-down sequence in the surrounding frames - marking as false-positive";
                            isPulldown = false;
                        } else {
                            qDebug() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber() <<
                                        "looks like a pull-down, but there is no pull-down sequence in the surrounding frames" <<
                                        "- strict checking is disabled, so marking as pulldown anyway";
                            isPulldown = true;
                        }
                    }

                    m_frames[frameNumber].isPullDown(isPulldown);
                    if (m_frames[frameNumber].isPullDown()) {
                        //qDebug() << "Seq. frame" << m_frames[frameNumber].seqFrameNumber << "marked as pulldown";
                        m_numberOfPulldowns++;
                    }
                }
            }
        }
    }

    // Measure and record a quality value for each frame
    qDebug() << "Performing a frame quality analysis for each frame";
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        // If the frame following the current one has a lower VBI number, give the current
        // frame a quality penalty as the likelyhood the player skipped is higher
        double penaltyPercent = 0;
        if (frameNumber < m_numberOfFrames - 1) {
            if (vbiData[frameNumber + 1].picNo < vbiData[frameNumber].picNo) penaltyPercent = 80.0;
            else penaltyPercent = 100.0;
        }

        // Add the Black SNR to the quality value
        // Get the average bPSNR for both fields
        double bsnr = (ldDecodeMetaData->getFieldVitsMetrics(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).bPSNR +
                ldDecodeMetaData->getFieldVitsMetrics(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).bPSNR) / 2.0;

        // Convert logarithmic to linear and then into percentage
        double blackSnrLinear = pow(bsnr / 20, 10);
        double snrReferenceLinear = pow(43.0 / 20, 10); // Note: 43 dB is the expected maximum
        double bsnrPercent = (100.0 / snrReferenceLinear) * blackSnrLinear;
        if (bsnrPercent > 100.0) bsnrPercent = 100.0;

        // Calculate the cumulative length of all the dropouts in the frame (by summing both fields)
        qint32 totalDotsInFrame = (ldDecodeMetaData->getVideoParameters().fieldHeight * 2) + ldDecodeMetaData->getVideoParameters().fieldWidth;
        DropOuts dropOuts1 = ldDecodeMetaData->getFieldDropOuts(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1));
        DropOuts dropOuts2 = ldDecodeMetaData->getFieldDropOuts(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1));

        qint32 frameDoLength = 0;
        for (qint32 i = 0; i < dropOuts1.size(); i++) {
            frameDoLength += dropOuts1.endx(i) - dropOuts1.startx(i);
        }

        for (qint32 i = 0; i < dropOuts2.size(); i++) {
            frameDoLength += dropOuts2.endx(i) - dropOuts2.startx(i);
        }

        double frameDoPercent = 100.0 - (static_cast<double>(frameDoLength) / static_cast<double>(totalDotsInFrame));

        // Include the sync confidence in the quality value (this is 100% where each measurement is 50% of the total)
        qint32 syncConfPercent = (ldDecodeMetaData->getField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).syncConf +
                                  ldDecodeMetaData->getField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).syncConf) / 2;

        m_frames[frameNumber].frameQuality((bsnrPercent + penaltyPercent + static_cast<double>(syncConfPercent) + (frameDoPercent * 1000.0)) / 1004.0);
        //qDebug() << "Frame:" << frameNumber << bsnrPercent << penaltyPercent << syncConfPercent << frameDoPercent << "quality =" << m_frames[frameNumber].frameQuality();
    }

    // Record the phase for both fields of each frame
    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        m_frames[frameNumber].firstFieldPhase(ldDecodeMetaData->getField(ldDecodeMetaData->getFirstFieldNumber(frameNumber + 1)).fieldPhaseID);
        m_frames[frameNumber].secondFieldPhase(ldDecodeMetaData->getField(ldDecodeMetaData->getSecondFieldNumber(frameNumber + 1)).fieldPhaseID);
    }

}

DiscMap::~DiscMap()
{
    delete ldDecodeMetaData;
}

// Custom streaming operator (for debug)
QDebug operator<<(QDebug dbg, const DiscMap &discMap)
{
    dbg.nospace().noquote() << "DiscMap(Frames " << discMap.numberOfFrames() <<
                               ", disc type is " << discMap.discType() << ", video format is " <<
                               discMap.discFormat() << ", detected " <<
                               discMap.numberOfPulldowns() << " pulldown frames)";

    return dbg.maybeSpace();
}

// Get methods

// Get the metadata file name for the TBC
QString DiscMap::filename() const
{
    return m_metadataFileInfo.filePath();
}

// Get the validity flag
bool DiscMap::valid() const
{
    return m_tbcValid;
}

// Get the available number of frames
qint32 DiscMap::numberOfFrames() const
{
    return m_numberOfFrames;
}

// Get the disc type
bool DiscMap::isDiscCav() const
{
    return m_isDiscCav;
}

// Get the disc video format
bool DiscMap::isDiscPal() const
{
    return m_isDiscPal;
}

// Method to return the disc type as a string
QString DiscMap::discType() const
{
    return m_discType;
}

// Method to return the disc format as a string
QString DiscMap::discFormat() const
{
    return m_videoSystemDescription;
}

// Method to return the VBI frame number
qint32 DiscMap::vbiFrameNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "vbiFrameNumber out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].vbiFrameNumber();
}

// Method to set the VBI frame number
void DiscMap::setVbiFrameNumber(qint32 frameNumber, qint32 vbiFrameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "setVbiFrameNumber out of frameNumber range";
        return;
    }
    m_frames[frameNumber].vbiFrameNumber(vbiFrameNumber);
}

// Method to return the original sequential frame number (which maps to the lddecodemetadata VBI)
qint32 DiscMap::seqFrameNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "seqFrameNumber out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].seqFrameNumber();
}

// Get the pulldown flag for a frame
bool DiscMap::isPulldown(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isPulldown out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isPullDown();
}

// Get the picture stop flag for a frame
bool DiscMap::isPictureStop(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isPictureStop out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isPictureStop();
}

// Get the number of pulldown frames on the disc
qint32 DiscMap::numberOfPulldowns() const
{
    return m_numberOfPulldowns;
}

// Get the lead in/out flag for a frame
bool DiscMap::isLeadInOut(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isLeadInOut out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isLeadInOrOut();
}

// Get the frame quality
double DiscMap::frameQuality(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "frameQuality out of frameNumber range";
        return -1;
    }
    return m_frames[frameNumber].frameQuality();
}

// Get the isPadded flag
bool DiscMap::isPadded(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isPadded out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isPadded();
}

// Set a frame as marked for deletion
void DiscMap::setMarkedForDeletion(qint32 frameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "setMarkedForDeletion out of frameNumber range";
        return;
    }
    m_frames[frameNumber].isMarkedForDeletion(true);
}

// Get the isClvOffset flag
bool DiscMap::isClvOffset(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isClvOffset out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].isClvOffset();
}

// Return true if the phase of the frame is correct according to the leading and trailing frames
// Note: This checks that:
//  The second field of the preceeding frame phase is -1 from the first field of the current frame
//  The second field of the current frame is -1 from the first field of the following frame
bool DiscMap::isPhaseCorrect(qint32 frameNumber) const
{
    qint32 expectedNextPhase = -1;

    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isPhaseCorrect out of frameNumber range";
        return false;
    }

    // Check that the phase of the preceeding field and the first field
    // of the current frame are in sequence
    if (frameNumber > 0) { // not the first frame
        expectedNextPhase = m_frames[frameNumber - 1].secondFieldPhase() + 1;
        if (m_isDiscPal && expectedNextPhase == 9) expectedNextPhase = 1;
        if (!m_isDiscPal && expectedNextPhase == 5) expectedNextPhase = 1;
        if (m_frames[frameNumber].firstFieldPhase() != expectedNextPhase) {
            qDebug() << "Frame number" << frameNumber << "phase sequence does not match preceeding frame! -"
            << expectedNextPhase << "expected but got" << m_frames[frameNumber].firstFieldPhase();
            return false;
        }
    }

    // Check that the phase of the second field and the first
    // field of the next frame are in sequence
    if (frameNumber != m_numberOfFrames) { // not the last frame
        expectedNextPhase = m_frames[frameNumber].secondFieldPhase() + 1;
        if (m_isDiscPal && expectedNextPhase == 9) expectedNextPhase = 1;
        if (!m_isDiscPal && expectedNextPhase == 5) expectedNextPhase = 1;
        if (m_frames[frameNumber + 1].firstFieldPhase() != expectedNextPhase) {
            qDebug() << "Frame number" << frameNumber << "phase sequence does not match following frame! -"
            << expectedNextPhase << "expected but got" << m_frames[frameNumber].secondFieldPhase();
            return false;
        }
    }

    return true;
}

// Return true if the phase of the frame is the same as the preceeding frame
bool DiscMap::isPhaseRepeating(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "isPhaseCorrect out of frameNumber range";
        return false;
    }

    if (frameNumber > 0) { // not the first frame
        if ((m_frames[frameNumber].firstFieldPhase() == m_frames[frameNumber - 1].firstFieldPhase()) &&
                (m_frames[frameNumber].secondFieldPhase() == m_frames[frameNumber - 1].secondFieldPhase())) return true;
    } else {
        // Frame number 0 can never be a repeat of the previous frame`
        return true;
    }

    return false;
}

// Flush the frames (delete anything marked for deletion)
// Returns the number of frames deleted
qint32 DiscMap::flush()
{
    qint32 origSize = m_frames.size();

    // Erase
    m_frames.erase(
        std::remove_if(m_frames.begin(), m_frames.end(),
                       [](const Frame &o) { return o.isMarkedForDeletion(); }),
                m_frames.end());

    // Reset the number of available frames
    m_numberOfFrames = m_frames.size();

    return origSize - m_frames.size();
}

// Sort the discmap by frame number (accounting for pull-downs if required)
void DiscMap::sort()
{
    // Here we sort the disc map (m_frames) using frame numbers.  If a frame is NTSC CAV pull-down
    // it will not have a frame number - the only thing we can do is sort it so the
    // pull-downs are sorted following the preceeding numbered frame (which should keep
    // them in the right place).
    //
    // Note that the stuct has an overloaded < operator that controls the sort comparison
    // (see the .h file)
    std::sort(m_frames.begin(), m_frames.end());

    m_numberOfFrames = m_frames.size();
}

// Method to output frame debug for a frame number in the disc map
void DiscMap::debugFrameDetails(qint32 frameNumber)
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "debugFrameDetails out of frameNumber range";
        return;
    }
    qDebug() << m_frames[frameNumber];
}

// Check if frame number matches IEC 60857-1986 LaserVision NTSC Amendment 2
// clause 10.1.10 CLV time-code skip frame number sequence
bool DiscMap::isNtscAmendment2ClvFrameNumber(qint32 frameNumber)
{
    // l < 14 gives a maximum frame number of 124,974 (71 minutes)
    for (qint32 l = 0; l < 14; l++) {
        for (qint32 m = 1; m <= 9; m++) {
            qint32 n = 8991 * l + 899 * m;
            if (n == frameNumber) return true;
            if (n > frameNumber) return false;
        }
    }

    return false;
}

// Method to add padding frames to the disc map
// Note: padding is appended to the end of the disc map - so the
// disc map must be sorted afterwards.
void DiscMap::addPadding(qint32 startFrame, qint32 numberOfFrames)
{
    m_frames.reserve(m_frames.size() + numberOfFrames);
    qint32 currentVbi = m_frames[startFrame].vbiFrameNumber() + 1;
    for (qint32 i = 0; i < numberOfFrames; i++) {
        Frame paddingFrame;
        paddingFrame.vbiFrameNumber(currentVbi + i);
        paddingFrame.seqFrameNumber(-1);
        paddingFrame.isPadded(true);

        m_frames.push_back(paddingFrame);
    }

    m_numberOfFrames = m_frames.size();
}

// Method to get the current video field length from the metadata
qint32 DiscMap::getVideoFieldLength()
{
    return m_videoFieldLength;
}

// Method to get the current audio field length (in bytes) from the metadata
// Note: This acutally varies from field to field, so this provides
// a best guess
qint32 DiscMap::getApproximateAudioFieldLength()
{
    return m_audioFieldByteLength / 2; // return in qint16 samples
}

// Get first field number
qint32 DiscMap::getFirstFieldNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getFirstFieldNumber out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].firstField();
}

// Get second field number
qint32 DiscMap::getSecondFieldNumber(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getFirstFieldNumber out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].secondField();
}

// Get first field phase
qint32 DiscMap::getFirstFieldPhase(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getFirstFieldPhase out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].firstFieldPhase();
}

// Get second field phase
qint32 DiscMap::getSecondFieldPhase(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getSecondFieldPhase out of frameNumber range";
        return false;
    }
    return m_frames[frameNumber].secondFieldPhase();
}

// Get first field audio sample start position
qint32 DiscMap::getFirstFieldAudioDataStart(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getFirstFieldAudioDataStart out of frameNumber range";
        return false;
    }
    return ldDecodeMetaData->getFieldPcmAudioStart(m_frames[frameNumber].firstField());
}

// Get first field audio sample length
qint32 DiscMap::getFirstFieldAudioDataLength(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getFirstFieldAudioDataLength out of frameNumber range";
        return false;
    }
    return ldDecodeMetaData->getFieldPcmAudioLength(m_frames[frameNumber].firstField());
}

// Get second field audio sample start position
qint32 DiscMap::getSecondFieldAudioDataStart(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getSecondFieldAudioDataStart out of frameNumber range";
        return false;
    }
    return ldDecodeMetaData->getFieldPcmAudioStart(m_frames[frameNumber].secondField());
}

// Get second field audio sample length
qint32 DiscMap::getSecondFieldAudioDataLength(qint32 frameNumber) const
{
    if (frameNumber < 0 || frameNumber >= m_numberOfFrames) {
        qDebug() << "getSecondFieldAudioDataLength out of frameNumber range";
        return false;
    }
    return ldDecodeMetaData->getFieldPcmAudioLength(m_frames[frameNumber].secondField());
}

// Save the target metadata from the disc map
bool DiscMap::saveTargetMetadata(QFileInfo outputFileInfo)
{
    qint32 notifyInterval = m_numberOfFrames / 50;
    if (notifyInterval < 1) notifyInterval = 1;

    LdDecodeMetaData targetMetadata;
    LdDecodeMetaData::VideoParameters sourceVideoParameters = ldDecodeMetaData->getVideoParameters();

    // Indicate that the source has been mapped
    sourceVideoParameters.isMapped = true;
    targetMetadata.setVideoParameters(sourceVideoParameters);

    // Store the PCM audio parameters
    targetMetadata.setPcmAudioParameters(ldDecodeMetaData->getPcmAudioParameters());

    // Set the number of sequential fields
    targetMetadata.setNumberOfFields(m_numberOfFrames * 2);

    // Make a VBI Decoder object for verifying generated VBI
    VbiDecoder vbiDecoder;

    for (qint32 frameNumber = 0; frameNumber < m_numberOfFrames; frameNumber++) {
        if (!m_frames[frameNumber].isPadded()) {
            // Normal frame metadata

            // Normal frame - get the data from the source video
            qint32 firstFieldNumber = m_frames[frameNumber].firstField();
            qint32 secondFieldNumber = m_frames[frameNumber].secondField();

            // Get the source metadata for the fields
            LdDecodeMetaData::Field firstSourceMetadata = ldDecodeMetaData->getField(firstFieldNumber);
            LdDecodeMetaData::Field secondSourceMetadata = ldDecodeMetaData->getField(secondFieldNumber);

            // Generate new VBI data for the frame
            if (m_isDiscCav) {
                // Disc is CAV - add a frame number
                // The frame number is hex 0xF12345 (where 1,2,3,4,5 are hex digits 0-9)
                // inserted into VBI lines 17 and 18 of the first field
                if (!firstSourceMetadata.vbi.inUse) {
                    firstSourceMetadata.vbi.inUse = true;
                    firstSourceMetadata.vbi.vbiData[0] = 0;
                }

                firstSourceMetadata.vbi.vbiData[1] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());

                // Note: Because only 2 lines of VBI are replaced here, its possible that corruption
                // in the unmodified line causes the resulting VBI to be invalid - so we need to check
                // for that here
                VbiDecoder::Vbi vbi = vbiDecoder.decodeFrame(
                        firstSourceMetadata.vbi.vbiData[0], firstSourceMetadata.vbi.vbiData[1], firstSourceMetadata.vbi.vbiData[2],
                        secondSourceMetadata.vbi.vbiData[0], secondSourceMetadata.vbi.vbiData[1], secondSourceMetadata.vbi.vbiData[2]);

                if (vbi.picNo != m_frames[frameNumber].vbiFrameNumber()) {
                    qInfo() << "Warning: Updated VBI frame number for frame" << m_frames[frameNumber].vbiFrameNumber() << "has been corrupted by exisiting VBI data - overwriting all VBI for frame";
                    firstSourceMetadata.vbi.vbiData[0] = 0;
                }
            } else {
                // Disc is CLV - add a timecode
                if (!firstSourceMetadata.vbi.inUse) {
                    firstSourceMetadata.vbi.inUse = true;
                }
                firstSourceMetadata.vbi.vbiData[0] = convertFrameToClvPicNo(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
            }

            // Append the fields to the metadata
            targetMetadata.appendField(firstSourceMetadata);
            targetMetadata.appendField(secondSourceMetadata);
        } else {
            // Padded frame metadata

            // Generate dummy target field metadata
            LdDecodeMetaData::Field firstSourceMetadata;
            LdDecodeMetaData::Field secondSourceMetadata;
            firstSourceMetadata.isFirstField = true;
            secondSourceMetadata.isFirstField = false;
            firstSourceMetadata.pad = true;
            secondSourceMetadata.pad = true;

            // Generate VBI data for the padded (dummy) output frame
            // Also add the padded size of the audio sample data
            if (m_isDiscCav) {
                // CAV
                firstSourceMetadata.vbi.inUse = true;
                firstSourceMetadata.vbi.vbiData[0] = 0;
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToVbi(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.audioSamples = m_audioFieldSampleLength;

                secondSourceMetadata.vbi.inUse = true;
                secondSourceMetadata.vbi.vbiData[0] = 0;
                secondSourceMetadata.vbi.vbiData[1] = 0;
                secondSourceMetadata.vbi.vbiData[2] = 0;
                secondSourceMetadata.audioSamples = m_audioFieldSampleLength;
            } else {
                // CLV
                firstSourceMetadata.vbi.inUse = true;
                firstSourceMetadata.vbi.vbiData[0] = convertFrameToClvPicNo(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[1] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.vbi.vbiData[2] = convertFrameToClvTimeCode(m_frames[frameNumber].vbiFrameNumber());
                firstSourceMetadata.audioSamples = m_audioFieldSampleLength;

                secondSourceMetadata.vbi.inUse = true;
                secondSourceMetadata.vbi.vbiData[0] = 0;
                secondSourceMetadata.vbi.vbiData[1] = 0;
                secondSourceMetadata.vbi.vbiData[2] = 0;
                secondSourceMetadata.audioSamples = m_audioFieldSampleLength;
            }

            // Append the fields to the metadata
            targetMetadata.appendField(firstSourceMetadata);
            targetMetadata.appendField(secondSourceMetadata);
        }

        // Notify the user
        if (frameNumber % notifyInterval == 0) {
            qInfo() << "Created metadata for frame" << frameNumber << "of" << m_numberOfFrames;
        }
    }

    // Save the target video metadata
    qInfo() << "Writing target metadata to disc...";
    targetMetadata.write(outputFileInfo.filePath());
    qInfo() << "Target metadata written";

    return true;
}

// Convert a frame number to the VBI hex frame number representation
// See the IEC specification for details of the VBI format
qint32 DiscMap::convertFrameToVbi(qint32 frameNumber)
{
    // Generate a string containing the required number
    QString number = "00F" + QString("%1").arg(frameNumber, 5, 10, QChar('0'));
    bool ok;
    qint32 returnValue = number.toInt(&ok, 16);
    if (!ok) returnValue = 0;

    return returnValue;
}

// Convert a frame number to a VBI CLV picture number
// See the IEC specification for details of the VBI format
qint32 DiscMap::convertFrameToClvPicNo(qint32 frameNumber)
{
    // Convert the frame number into a CLV timecode
    LdDecodeMetaData::ClvTimecode timecode = ldDecodeMetaData->convertFrameNumberToClvTimecode(frameNumber);

    // Generate the seconds
    qint32 secondsX1;
    if (timecode.seconds % 10 == 0) secondsX1 = timecode.seconds;
    else secondsX1 = (timecode.seconds - (timecode.seconds % 10));
    qint32 secondsX3 = timecode.seconds - secondsX1;
    secondsX1 = ((secondsX1 + 10) / 10) + 9;

    // Generate a string containing the required number
    QString number = "008" + QString("%1").arg(secondsX1, 1, 16, QChar('0')) + "E" +
            QString("%1").arg(secondsX3, 1, 10, QChar('0')) +
            QString("%1").arg(timecode.pictureNumber, 2, 10, QChar('0'));
    bool ok;
    qint32 returnValue = number.toInt(&ok, 16);
    if (!ok) returnValue = 0;

    return returnValue;
}

// Convert a frame number to a CLV programme time code
// See the IEC specification for details of the VBI format
qint32 DiscMap::convertFrameToClvTimeCode(qint32 frameNumber)
{
    // Convert the frame number into a CLV timecode
    LdDecodeMetaData::ClvTimecode timecode = ldDecodeMetaData->convertFrameNumberToClvTimecode(frameNumber);

    // Generate a string containing the required number
    QString number = "00F" + QString("%1").arg(timecode.hours, 1, 10, QChar('0')) + "DD" +
            QString("%1").arg(timecode.minutes, 2, 10, QChar('0'));
    bool ok;
    qint32 returnValue = number.toInt(&ok, 16);
    if (!ok) returnValue = 0;

    return returnValue;
}
