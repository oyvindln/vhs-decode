/************************************************************************

    lddecodemetadata.cpp

    ld-decode-tools TBC library
    Copyright (C) 2018-2020 Simon Inns
    Copyright (C) 2022 Ryan Holtz
    Copyright (C) 2022-2023 Adam Sampson

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

#include "lddecodemetadata.h"

#include "jsonio.h"

#include <cassert>
#include <fstream>

// Default values used when configuring VideoParameters for a particular video system.
// See the comments in VideoParameters for the meanings of these values.
// For descriptions of the systems, see ITU BT.1700.
struct VideoSystemDefaults {
    VideoSystem system;
    const char *name;
    double fSC;
    qint32 minActiveFrameLine;
    qint32 firstActiveFieldLine;
    qint32 lastActiveFieldLine;
    qint32 firstActiveFrameLine;
    qint32 lastActiveFrameLine;
};

static constexpr VideoSystemDefaults palDefaults {
    PAL,
    "PAL",
    (283.75 * 15625) + 25,
    2,
    22, 308,
    // Interlaced line 44 is PAL line 23 (the first active half-line)
    // Interlaced line 619 is PAL line 623 (the last active half-line)
    44, 620,
};

static constexpr VideoSystemDefaults ntscDefaults {
    NTSC,
    "NTSC",
    315.0e6 / 88.0,
    1,
    20, 259,
    // Interlaced line 40 is NTSC line 21 (the closed-caption line before the first active half-line)
    // Interlaced line 524 is NTSC line 263 (the last active half-line)
    40, 525,
};

static constexpr VideoSystemDefaults palMDefaults {
    PAL_M,
    "PAL-M",
    5.0e6 * (63.0 / 88.0) * (909.0 / 910.0),
    ntscDefaults.minActiveFrameLine,
    ntscDefaults.firstActiveFieldLine, ntscDefaults.lastActiveFieldLine,
    ntscDefaults.firstActiveFrameLine, ntscDefaults.lastActiveFrameLine,
};

static constexpr VideoSystemDefaults secamDefaults {
    SECAM,
    "SECAM",
    (283.75 * 15625) + 25, // TODO FIX!!!
    palDefaults.minActiveFrameLine,
    palDefaults.firstActiveFieldLine, palDefaults.lastActiveFieldLine,
    palDefaults.firstActiveFrameLine, palDefaults.lastActiveFrameLine,
};

// These must be in the same order as enum VideoSystem
static constexpr VideoSystemDefaults VIDEO_SYSTEM_DEFAULTS[] = {
    palDefaults,
    ntscDefaults,
    palMDefaults,
    secamDefaults,
};

// Return appropriate defaults for the selected video system
static const VideoSystemDefaults &getSystemDefaults(const LdDecodeMetaData::VideoParameters &videoParameters)
{
    return VIDEO_SYSTEM_DEFAULTS[videoParameters.system];
}

// Look up a video system by name.
// Return true and set system if found; if not found, return false.
bool parseVideoSystemName(QString name, VideoSystem &system)
{
    // Search VIDEO_SYSTEM_DEFAULTS for a matching name
    for (const auto &defaults: VIDEO_SYSTEM_DEFAULTS) {
        if (name == defaults.name) {
            system = defaults.system;
            return true;
        }
    }
    return false;
}

// Read Vbi from JSON
void LdDecodeMetaData::Vbi::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "vbiData") {
            reader.beginArray();

            // There should be exactly 3 values, but handle more or less
            unsigned int i = 0;
            while (reader.readElement()) {
                int value;
                reader.read(value);

                if (i < vbiData.size()) vbiData[i++] = value;
            }
            while (i < vbiData.size()) vbiData[i++] = 0;

            reader.endArray();
        } else {
            reader.discard();
        }
    }

    reader.endObject();

    inUse = true;
}

// Write Vbi to JSON
void LdDecodeMetaData::Vbi::write(JsonWriter &writer) const
{
    assert(inUse);

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("vbiData");
    writer.beginArray();
    for (auto value : vbiData) {
        writer.writeElement();
        writer.write(value);
    }
    writer.endArray();

    writer.endObject();
}

// Read VideoParameters from JSON
void LdDecodeMetaData::VideoParameters::read(JsonReader &reader)
{
    bool isSourcePal = false;
    std::string systemString = "";

    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "activeVideoEnd") reader.read(activeVideoEnd);
        else if (member == "activeVideoStart") reader.read(activeVideoStart);
        else if (member == "black16bIre") reader.read(black16bIre);
        else if (member == "colourBurstEnd") reader.read(colourBurstEnd);
        else if (member == "colourBurstStart") reader.read(colourBurstStart);
        else if (member == "fieldHeight") reader.read(fieldHeight);
        else if (member == "fieldWidth") reader.read(fieldWidth);
        else if (member == "gitBranch") reader.read(gitBranch);
        else if (member == "gitCommit") reader.read(gitCommit);
        else if (member == "isMapped") reader.read(isMapped);
        else if (member == "isSourcePal") reader.read(isSourcePal); // obsolete
        else if (member == "isSubcarrierLocked") reader.read(isSubcarrierLocked);
        else if (member == "isWidescreen") reader.read(isWidescreen);
        else if (member == "numberOfSequentialFields") reader.read(numberOfSequentialFields);
        else if (member == "sampleRate") reader.read(sampleRate);
        else if (member == "system") reader.read(systemString);
        else if (member == "white16bIre") reader.read(white16bIre);
        else reader.discard();
    }

    reader.endObject();

    // Work out which video system is being used
    if (systemString == "") {
        // Not specified -- detect based on isSourcePal and fieldHeight
        if (isSourcePal) {
            if (fieldHeight < 300) system = PAL_M;
            else system = PAL;
        } else system = NTSC;
    } else if (!parseVideoSystemName(QString::fromStdString(systemString), system)) {
        reader.throwError("unknown value for videoParameters.system");
    }

    isValid = true;
}

// Write VideoParameters to JSON
void LdDecodeMetaData::VideoParameters::write(JsonWriter &writer) const
{
    assert(isValid);

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("activeVideoEnd", activeVideoEnd);
    writer.writeMember("activeVideoStart", activeVideoStart);
    writer.writeMember("black16bIre", black16bIre);
    writer.writeMember("colourBurstEnd", colourBurstEnd);
    writer.writeMember("colourBurstStart", colourBurstStart);
    writer.writeMember("fieldHeight", fieldHeight);
    writer.writeMember("fieldWidth", fieldWidth);
    if (gitBranch != "") {
        writer.writeMember("gitBranch", gitBranch);
    }
    if (gitCommit != "") {
        writer.writeMember("gitCommit", gitCommit);
    }
    writer.writeMember("isMapped", isMapped);
    writer.writeMember("isSubcarrierLocked", isSubcarrierLocked);
    writer.writeMember("isWidescreen", isWidescreen);
    writer.writeMember("numberOfSequentialFields", numberOfSequentialFields);
    writer.writeMember("sampleRate", sampleRate);
    writer.writeMember("system", VIDEO_SYSTEM_DEFAULTS[system].name);
    writer.writeMember("white16bIre", white16bIre);

    writer.endObject();
}

// Read VitsMetrics from JSON
void LdDecodeMetaData::VitsMetrics::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "bPSNR") reader.read(bPSNR);
        else if (member == "wSNR") reader.read(wSNR);
        else reader.discard();
    }

    reader.endObject();

    inUse = true;
}

// Write VitsMetrics to JSON
void LdDecodeMetaData::VitsMetrics::write(JsonWriter &writer) const
{
    assert(inUse);

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("bPSNR", bPSNR);
    writer.writeMember("wSNR", wSNR);

    writer.endObject();
}

// Read Ntsc from JSON
void LdDecodeMetaData::Ntsc::read(JsonReader &reader, ClosedCaption &closedCaption)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "isFmCodeDataValid") reader.read(isFmCodeDataValid);
        else if (member == "fmCodeData") reader.read(fmCodeData);
        else if (member == "fieldFlag") reader.read(fieldFlag);
        else if (member == "isVideoIdDataValid") reader.read(isVideoIdDataValid);
        else if (member == "videoIdData") reader.read(videoIdData);
        else if (member == "whiteFlag") reader.read(whiteFlag);
        else if (member == "ccData0") {
            // rev7 and earlier put ccData0/1 here rather than in cc
            reader.read(closedCaption.data0);
            closedCaption.inUse = true;
        } else if (member == "ccData1") {
            reader.read(closedCaption.data1);
            closedCaption.inUse = true;
        } else {
            reader.discard();
        }
    }

    reader.endObject();

    inUse = true;
}

// Write Ntsc to JSON
void LdDecodeMetaData::Ntsc::write(JsonWriter &writer) const
{
    assert(inUse);

    writer.beginObject();

    // Keep members in alphabetical order
    if (isFmCodeDataValid) {
        writer.writeMember("fieldFlag", fieldFlag);
    }
    if (isFmCodeDataValid) {
        writer.writeMember("fmCodeData", fmCodeData);
    }
    writer.writeMember("isFmCodeDataValid", isFmCodeDataValid);
    if (isVideoIdDataValid) {
        writer.writeMember("videoIdData", videoIdData);
    }
    writer.writeMember("isVideoIdDataValid", isVideoIdDataValid);
    if (whiteFlag) {
        writer.writeMember("whiteFlag", whiteFlag);
    }

    writer.endObject();
}

// Read Vitc from JSON
void LdDecodeMetaData::Vitc::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "vitcData") {
            reader.beginArray();

            // There should be exactly 8 values, but handle more or less
            unsigned int i = 0;
            while (reader.readElement()) {
                int value;
                reader.read(value);

                if (i < vitcData.size()) vitcData[i++] = value;
            }
            while (i < vitcData.size()) vitcData[i++] = 0;

            reader.endArray();
        } else {
            reader.discard();
        }
    }

    reader.endObject();

    inUse = true;
}

// Write Vitc to JSON
void LdDecodeMetaData::Vitc::write(JsonWriter &writer) const
{
    assert(inUse);

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("vitcData");
    writer.beginArray();
    for (auto value : vitcData) {
        writer.writeElement();
        writer.write(value);
    }
    writer.endArray();

    writer.endObject();
}

// Read ClosedCaption from JSON
void LdDecodeMetaData::ClosedCaption::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "data0") reader.read(data0);
        else if (member == "data1") reader.read(data1);
        else reader.discard();
    }

    reader.endObject();

    inUse = true;
}

// Write ClosedCaption to JSON
void LdDecodeMetaData::ClosedCaption::write(JsonWriter &writer) const
{
    assert(inUse);

    writer.beginObject();

    // Keep members in alphabetical order
    if (data0 != -1) {
        writer.writeMember("data0", data0);
    }
    if (data1 != -1) {
        writer.writeMember("data1", data1);
    }

    writer.endObject();
}

// Read PcmAudioParameters from JSON
void LdDecodeMetaData::PcmAudioParameters::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "bits") reader.read(bits);
        else if (member == "isLittleEndian") reader.read(isLittleEndian);
        else if (member == "isSigned") reader.read(isSigned);
        else if (member == "sampleRate") reader.read(sampleRate);
        else reader.discard();
    }

    reader.endObject();

    isValid = true;
}

// Write PcmAudioParameters to JSON
void LdDecodeMetaData::PcmAudioParameters::write(JsonWriter &writer) const
{
    assert(isValid);

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("bits", bits);
    writer.writeMember("isLittleEndian", isLittleEndian);
    writer.writeMember("isSigned", isSigned);
    writer.writeMember("sampleRate", sampleRate);

    writer.endObject();
}

// Read Field from JSON
void LdDecodeMetaData::Field::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "audioSamples") reader.read(audioSamples);
        else if (member == "cc") closedCaption.read(reader);
        else if (member == "decodeFaults") reader.read(decodeFaults);
        else if (member == "diskLoc") reader.read(diskLoc);
        else if (member == "dropOuts") dropOuts.read(reader);
        else if (member == "efmTValues") reader.read(efmTValues);
        else if (member == "fieldPhaseID") reader.read(fieldPhaseID);
        else if (member == "fileLoc") reader.read(fileLoc);
        else if (member == "isFirstField") reader.read(isFirstField);
        else if (member == "medianBurstIRE") reader.read(medianBurstIRE);
        else if (member == "ntsc") ntsc.read(reader, closedCaption);
        else if (member == "pad") reader.read(pad);
        else if (member == "seqNo") reader.read(seqNo);
        else if (member == "syncConf") reader.read(syncConf);
        else if (member == "vbi") vbi.read(reader);
        else if (member == "vitc") vitc.read(reader);
        else if (member == "vitsMetrics") vitsMetrics.read(reader);
        else reader.discard();
    }

    reader.endObject();
}

// Write Field to JSON
void LdDecodeMetaData::Field::write(JsonWriter &writer) const
{
    writer.beginObject();

    // Keep members in alphabetical order
    if (audioSamples != -1) {
        writer.writeMember("audioSamples", audioSamples);
    }
    if (closedCaption.inUse) {
        writer.writeMember("cc");
        closedCaption.write(writer);
    }
    if (decodeFaults != -1) {
        writer.writeMember("decodeFaults", decodeFaults);
    }
    if (diskLoc != -1) {
        writer.writeMember("diskLoc", diskLoc);
    }
    if (!dropOuts.empty()) {
        writer.writeMember("dropOuts");
        dropOuts.write(writer);
    }
    if (efmTValues != -1) {
        writer.writeMember("efmTValues", efmTValues);
    }
    if (fieldPhaseID != -1) {
        writer.writeMember("fieldPhaseID", fieldPhaseID);
    }
    if (fileLoc != -1) {
        writer.writeMember("fileLoc", fileLoc);
    }
    writer.writeMember("isFirstField", isFirstField);
    writer.writeMember("medianBurstIRE", medianBurstIRE);
    if (ntsc.inUse) {
        writer.writeMember("ntsc");
        ntsc.write(writer);
    }
    writer.writeMember("pad", pad);
    writer.writeMember("seqNo", seqNo);
    writer.writeMember("syncConf", syncConf);
    if (vbi.inUse) {
        writer.writeMember("vbi");
        vbi.write(writer);
    }
    if (vitc.inUse) {
        writer.writeMember("vitc");
        vitc.write(writer);
    }
    if (vitsMetrics.inUse) {
        writer.writeMember("vitsMetrics");
        vitsMetrics.write(writer);
    }

    writer.endObject();
}

LdDecodeMetaData::LdDecodeMetaData()
{
    clear();
}

// Reset the metadata to the defaults
void LdDecodeMetaData::clear()
{
    // Default to the standard still-frame field order (of first field first)
    isFirstFieldFirst = true;

    // Reset the parameters to their defaults
    videoParameters = VideoParameters();
    pcmAudioParameters = PcmAudioParameters();

    fields.clear();
}

// Read all metadata from a JSON file
bool LdDecodeMetaData::read(QString fileName)
{
    std::ifstream jsonFile(fileName.toStdString());
    if (jsonFile.fail()) {
        qCritical("Opening JSON input file failed: JSON file cannot be opened/does not exist");
        return false;
    }

    clear();

    JsonReader reader(jsonFile);

    try {
        reader.beginObject();

        std::string member;
        while (reader.readMember(member)) {
            if (member == "fields") readFields(reader);
            else if (member == "pcmAudioParameters") pcmAudioParameters.read(reader);
            else if (member == "videoParameters") videoParameters.read(reader);
            else reader.discard();
        }

        reader.endObject();
    } catch (JsonReader::Error &error) {
        qCritical() << "Parsing JSON file failed:" << error.what();
        return false;
    }

    jsonFile.close();

    // Check we saw VideoParameters - if not, we can't do anything useful!
    if (!videoParameters.isValid) {
        qCritical("JSON file invalid: videoParameters object is not defined");
        return false;
    }

    // Check numberOfSequentialFields is consistent
    if (videoParameters.numberOfSequentialFields != fields.size()) {
        qCritical("JSON file invalid: numberOfSequentialFields does not match fields array");
        return false;
    }

    // Now we know the video system, initialise the rest of VideoParameters
    initialiseVideoSystemParameters();

    // Generate the PCM audio map based on the field metadata
    generatePcmAudioMap();

    return true;
}

// Write all metadata out to a JSON file
bool LdDecodeMetaData::write(QString fileName) const
{
    std::ofstream jsonFile(fileName.toStdString());
    if (jsonFile.fail()) {
        qCritical("Opening JSON output file failed");
        return false;
    }

    JsonWriter writer(jsonFile);

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("fields");
    writeFields(writer);
    if (pcmAudioParameters.isValid) {
        writer.writeMember("pcmAudioParameters");
        pcmAudioParameters.write(writer);
    }
    writer.writeMember("videoParameters");
    videoParameters.write(writer);

    writer.endObject();

    jsonFile.close();

    return true;
}

// Read array of Fields from JSON
void LdDecodeMetaData::readFields(JsonReader &reader)
{
    reader.beginArray();

    while (reader.readElement()) {
        Field field;
        field.read(reader);
        fields.push_back(field);
    }

    reader.endArray();
}

// Write array of Fields to JSON
void LdDecodeMetaData::writeFields(JsonWriter &writer) const
{
    writer.beginArray();

    for (const Field &field : fields) {
        writer.writeElement();
        field.write(writer);
    }

    writer.endArray();
}

// This method returns the videoParameters metadata
const LdDecodeMetaData::VideoParameters &LdDecodeMetaData::getVideoParameters()
{
    assert(videoParameters.isValid);
    return videoParameters;
}

// This method sets the videoParameters metadata
void LdDecodeMetaData::setVideoParameters(const LdDecodeMetaData::VideoParameters &_videoParameters)
{
    videoParameters = _videoParameters;
    videoParameters.isValid = true;
}

// This method returns the pcmAudioParameters metadata
const LdDecodeMetaData::PcmAudioParameters &LdDecodeMetaData::getPcmAudioParameters()
{
    assert(pcmAudioParameters.isValid);
    return pcmAudioParameters;
}

// This method sets the pcmAudioParameters metadata
void LdDecodeMetaData::setPcmAudioParameters(const LdDecodeMetaData::PcmAudioParameters &_pcmAudioParameters)
{
    pcmAudioParameters = _pcmAudioParameters;
    pcmAudioParameters.isValid = true;
}

// Based on the video system selected, set default values for the members of
// VideoParameters that aren't obtained from the JSON.
void LdDecodeMetaData::initialiseVideoSystemParameters()
{
    const VideoSystemDefaults &defaults = getSystemDefaults(videoParameters);
    videoParameters.fSC = defaults.fSC;

    // Set default LineParameters
    LdDecodeMetaData::LineParameters lineParameters;
    processLineParameters(lineParameters);
}

// Validate LineParameters and apply them to the VideoParameters
void LdDecodeMetaData::processLineParameters(LdDecodeMetaData::LineParameters &lineParameters)
{
    lineParameters.applyTo(videoParameters);
}

// Validate and apply to a set of VideoParameters
void LdDecodeMetaData::LineParameters::applyTo(LdDecodeMetaData::VideoParameters &videoParameters)
{
    const bool firstFieldLineExists = firstActiveFieldLine != -1;
    const bool lastFieldLineExists = lastActiveFieldLine != -1;
    const bool firstFrameLineExists = firstActiveFrameLine != -1;
    const bool lastFrameLineExists = lastActiveFrameLine != -1;

    const VideoSystemDefaults &defaults = getSystemDefaults(videoParameters);
    const qint32 minFirstFrameLine = defaults.minActiveFrameLine;
    const qint32 defaultFirstFieldLine = defaults.firstActiveFieldLine;
    const qint32 defaultLastFieldLine = defaults.lastActiveFieldLine;
    const qint32 defaultFirstFrameLine = defaults.firstActiveFrameLine;
    const qint32 defaultLastFrameLine = defaults.lastActiveFrameLine;

    // Validate and potentially fix the first active field line.
    if (firstActiveFieldLine < 1 || firstActiveFieldLine > defaultLastFieldLine) {
        if (firstFieldLineExists) {
            qInfo().nospace() << "Specified first active field line " << firstActiveFieldLine << " out of bounds (1 to "
                              << defaultLastFieldLine << "), resetting to default (" << defaultFirstFieldLine << ").";
        }
        firstActiveFieldLine = defaultFirstFieldLine;
    }

    // Validate and potentially fix the last active field line.
    if (lastActiveFieldLine < 1 || lastActiveFieldLine > defaultLastFieldLine) {
        if (lastFieldLineExists) {
            qInfo().nospace() << "Specified last active field line " << lastActiveFieldLine << " out of bounds (1 to "
                              << defaultLastFieldLine << "), resetting to default (" << defaultLastFieldLine << ").";
        }
        lastActiveFieldLine = defaultLastFieldLine;
    }

    // Range-check the first and last active field lines.
    if (firstActiveFieldLine > lastActiveFieldLine) {
       qInfo().nospace() << "Specified last active field line " << lastActiveFieldLine << " is before specified first active field line"
                         << firstActiveFieldLine << ", resetting to defaults (" << defaultFirstFieldLine << "-" << defaultLastFieldLine << ").";
        firstActiveFieldLine = defaultFirstFieldLine;
        lastActiveFieldLine = defaultLastFieldLine;
    }

    // Validate and potentially fix the first active frame line.
    if (firstActiveFrameLine < minFirstFrameLine || firstActiveFrameLine > defaultLastFrameLine) {
        if (firstFrameLineExists) {
            qInfo().nospace() << "Specified first active frame line " << firstActiveFrameLine << " out of bounds (" << minFirstFrameLine << " to "
                              << defaultLastFrameLine << "), resetting to default (" << defaultFirstFrameLine << ").";
        }
        firstActiveFrameLine = defaultFirstFrameLine;
    }

    // Validate and potentially fix the last active frame line.
    if (lastActiveFrameLine < minFirstFrameLine || lastActiveFrameLine > defaultLastFrameLine) {
        if (lastFrameLineExists) {
            qInfo().nospace() << "Specified last active frame line " << lastActiveFrameLine << " out of bounds (" << minFirstFrameLine << " to "
                              << defaultLastFrameLine << "), resetting to default (" << defaultLastFrameLine << ").";
        }
        lastActiveFrameLine = defaultLastFrameLine;
    }

    // Range-check the first and last active frame lines.
    if (firstActiveFrameLine > lastActiveFrameLine) {
        qInfo().nospace() << "Specified last active frame line " << lastActiveFrameLine << " is before specified first active frame line"
                          << firstActiveFrameLine << ", resetting to defaults (" << defaultFirstFrameLine << "-" << defaultLastFrameLine << ").";
        firstActiveFrameLine = defaultFirstFrameLine;
        lastActiveFrameLine = defaultLastFrameLine;
    }

    // Store the new values back into videoParameters
    videoParameters.firstActiveFieldLine = firstActiveFieldLine;
    videoParameters.lastActiveFieldLine = lastActiveFieldLine;
    videoParameters.firstActiveFrameLine = firstActiveFrameLine;
    videoParameters.lastActiveFrameLine = lastActiveFrameLine;
}

// This method gets the metadata for the specified sequential field number (indexed from 1 (not 0!))
const LdDecodeMetaData::Field &LdDecodeMetaData::getField(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getField(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber];
}

// This method gets the VITS metrics metadata for the specified sequential field number
const LdDecodeMetaData::VitsMetrics &LdDecodeMetaData::getFieldVitsMetrics(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber].vitsMetrics;
}

// This method gets the VBI metadata for the specified sequential field number
const LdDecodeMetaData::Vbi &LdDecodeMetaData::getFieldVbi(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldVbi(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber].vbi;
}

// This method gets the NTSC metadata for the specified sequential field number
const LdDecodeMetaData::Ntsc &LdDecodeMetaData::getFieldNtsc(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldNtsc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber].ntsc;
}

// This method gets the VITC metadata for the specified sequential field number
const LdDecodeMetaData::Vitc &LdDecodeMetaData::getFieldVitc(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldVitc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber].vitc;
}

// This method gets the Closed Caption metadata for the specified sequential field number
const LdDecodeMetaData::ClosedCaption &LdDecodeMetaData::getFieldClosedCaption(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldClosedCaption(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber].closedCaption;
}

// This method gets the drop-out metadata for the specified sequential field number
const DropOuts &LdDecodeMetaData::getFieldDropOuts(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    return fields[fieldNumber].dropOuts;
}

// This method sets the field metadata for a field
void LdDecodeMetaData::updateField(const LdDecodeMetaData::Field &field, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber] = field;
}

// This method sets the field VBI metadata for a field
void LdDecodeMetaData::updateFieldVitsMetrics(const LdDecodeMetaData::VitsMetrics &vitsMetrics, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].vitsMetrics = vitsMetrics;
}

// This method sets the field VBI metadata for a field
void LdDecodeMetaData::updateFieldVbi(const LdDecodeMetaData::Vbi &vbi, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldVbi(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].vbi = vbi;
}

// This method sets the field NTSC metadata for a field
void LdDecodeMetaData::updateFieldNtsc(const LdDecodeMetaData::Ntsc &ntsc, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldNtsc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].ntsc = ntsc;
}

// This method sets the VITC metadata for a field
void LdDecodeMetaData::updateFieldVitc(const LdDecodeMetaData::Vitc &vitc, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldVitc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].vitc = vitc;
}

// This method sets the Closed Caption metadata for a field
void LdDecodeMetaData::updateFieldClosedCaption(const LdDecodeMetaData::ClosedCaption &closedCaption, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldClosedCaption(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].closedCaption = closedCaption;
}

// This method sets the field dropout metadata for a field
void LdDecodeMetaData::updateFieldDropOuts(const DropOuts &dropOuts, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::updateFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].dropOuts = dropOuts;
}

// This method clears the field dropout metadata for a field
void LdDecodeMetaData::clearFieldDropOuts(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;
    if (fieldNumber < 0 || fieldNumber >= getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::clearFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    fields[fieldNumber].dropOuts.clear();
}

// This method appends a new field to the existing metadata
void LdDecodeMetaData::appendField(const LdDecodeMetaData::Field &field)
{
    fields.append(field);

    videoParameters.numberOfSequentialFields = fields.size();
}

// Method to get the available number of fields (according to the metadata)
qint32 LdDecodeMetaData::getNumberOfFields()
{
    return fields.size();
}

// Method to set the available number of fields
// XXX This is unnecessary given appendField
void LdDecodeMetaData::setNumberOfFields(qint32 numberOfFields)
{
    videoParameters.numberOfSequentialFields = numberOfFields;
}

// A note about fields, frames and still-frames:
//
// There is a lot of confusing terminology around fields and the order in which
// they should be combined to make a 'frame'.  Basically, (taking NTSC as an example)
// a frame consists of frame lines numbered from 1 to 525.  A frame is made from two
// fields, one field contains field lines 1 to 262.5 and another 263 to 525 (although
// for convenence the 'half-lines' are usually treated as one full line and ignored
// so both fields contain a total of 263 lines of which 1 is ignored).
//
// When a frame is created, the field containing field lines 1-263 is interlaced
// with the field containing 263-525 creating a frame with field lines 1, 263, 2, 264
// and so on.  This 'frame' is then considered to contain frame lines 1-525.
//
// The field containing the first line of the frame is called the 'first field' and
// the field containing the second line of the frame is called the 'second field'.
//
// However, other names exist:
//
// Even/Odd - where the 'odd' field contains the odd line numbers 1, 3, 5, etc. This is
// the same as the 'first' field so odd = first and even = second.
//
// Upper/lower - where the 'upper' field contains the upper-part of each combination.
// This is the same as the first field so upper = first and lower = second.
//
// With a standard TV, as long as one field is first and the other is second, the only
// thing a TV requires is that the sequence of fields is constant.  They are simply
// displayed one set of fields after another to form a frame which is part of a
// moving image.
//
// This is an issue for 'still-frames' as, if the video sequence consists of
// still images (rather than motion), pausing at any given point can result in a
// frame containing a first field from one image and a second field from another
// as there is no concept of 'frame' in the video (just a sequence of first and
// second fields).
//
// Since digital formats are frame based (not field) this is an issue, as there
// is no way (from the video data) to tell how to combine fields into a
// still-frame (rather than just 'a frame').  The LaserDisc mastering could be
// in the order first field/second field = still-frame or second field/first field
// = still frame.
//
// This is why the following methods use the "isFirstFieldFirst" flag (which is
// a little confusing in itself).
//
// There are two ways to determine the 'isFirstFieldFirst'.  The first method is by
// user observation (it's pretty clear on a still-frame when it is wrong), the other
// (used by LaserDisc players) is to look for a CAV picture number in the VBI data
// of the field.  The IEC specification states that the picture number should only be
// in the first field of a frame. (Note: CLV discs don't really have to follow this
// as there are no 'still-frames' allowed by the original format).
//
// This gets even more confusing for NTSC discs using pull-down, where the sequence
// of fields making up the frames isn't even, so some field-pairs aren't considered
// to contain the first field of a still-frame (and when pausing the LaserDisc
// player will never use certain fields to render the still-frame).  Wikipedia is
// your friend if you want to learn more about it.
//
// Determining the correct setting of 'isFirstFieldFirst' is therefore outside of
// the shared-library scope.

// Method to get the available number of still-frames
qint32 LdDecodeMetaData::getNumberOfFrames()
{
    qint32 frameOffset = 0;

    // If the first field in the TBC input isn't the expected first field,
    // skip it when counting the number of still-frames
    if (isFirstFieldFirst) {
        // Expecting first field first
        if (!getField(1).isFirstField) frameOffset = 1;
    } else {
        // Expecting second field first
        if (getField(1).isFirstField) frameOffset = 1;
    }

    return (getNumberOfFields() / 2) - frameOffset;
}

// Method to get the first and second field numbers based on the frame number
// If field = 1 return the firstField, otherwise return second field
qint32 LdDecodeMetaData::getFieldNumber(qint32 frameNumber, qint32 field)
{
    qint32 firstFieldNumber = 0;
    qint32 secondFieldNumber = 0;

    // Verify the frame number
    if (frameNumber < 1) {
        qCritical() << "Invalid frame number, cannot determine fields";
        return -1;
    }

    // Calculate the first and last fields based on the position in the TBC
    if (isFirstFieldFirst) {
        // Expecting TBC file to provide still-frames as first field / second field
        firstFieldNumber = (frameNumber * 2) - 1;
        secondFieldNumber = firstFieldNumber + 1;
    } else {
        // Expecting TBC file to provide still-frames as second field / first field
        secondFieldNumber = (frameNumber * 2) - 1;
        firstFieldNumber = secondFieldNumber + 1;
    }

    // If the field number pointed to by firstFieldNumber doesn't have
    // isFirstField set, move forward field by field until the current
    // field does
    while (!getField(firstFieldNumber).isFirstField) {
        firstFieldNumber++;
        secondFieldNumber++;

        // Give up if we reach the end of the available fields
        if (firstFieldNumber > getNumberOfFields() || secondFieldNumber > getNumberOfFields()) {
            qCritical() << "Attempting to get field number failed - no isFirstField in JSON before end of file";
            firstFieldNumber = -1;
            secondFieldNumber = -1;
            break;
        }
    }

    // Range check the first field number
    if (firstFieldNumber > getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldNumber(): First field number exceed the available number of fields!";
        firstFieldNumber = -1;
        secondFieldNumber = -1;
    }

    // Range check the second field number
    if (secondFieldNumber > getNumberOfFields()) {
        qCritical() << "LdDecodeMetaData::getFieldNumber(): Second field number exceed the available number of fields!";
        firstFieldNumber = -1;
        secondFieldNumber = -1;
    }

    // Test for a buggy TBC file...
    if (getField(secondFieldNumber).isFirstField) {
        qCritical() << "LdDecodeMetaData::getFieldNumber(): Both of the determined fields have isFirstField set - the TBC source video is probably broken...";
    }

    if (field == 1) return firstFieldNumber; else return secondFieldNumber;
}

// Method to get the first field number based on the frame number
qint32 LdDecodeMetaData::getFirstFieldNumber(qint32 frameNumber)
{
    return getFieldNumber(frameNumber, 1);
}

// Method to get the second field number based on the frame number
qint32 LdDecodeMetaData::getSecondFieldNumber(qint32 frameNumber)
{
    return getFieldNumber(frameNumber, 2);
}

// Method to set the isFirstFieldFirst flag
void LdDecodeMetaData::setIsFirstFieldFirst(bool flag)
{
    isFirstFieldFirst = flag;
}

// Method to get the isFirstFieldFirst flag
bool LdDecodeMetaData::getIsFirstFieldFirst()
{
    return isFirstFieldFirst;
}

// Method to convert a CLV time code into an equivalent frame number (to make
// processing the timecodes easier)
qint32 LdDecodeMetaData::convertClvTimecodeToFrameNumber(LdDecodeMetaData::ClvTimecode clvTimeCode)
{
    // Calculate the frame number
    qint32 frameNumber = 0;
    VideoParameters videoParameters = getVideoParameters();

    // Check for invalid CLV timecode
    if (clvTimeCode.hours == -1 || clvTimeCode.minutes == -1 || clvTimeCode.seconds == -1 || clvTimeCode.pictureNumber == -1) {
        return -1;
    }

    if (clvTimeCode.hours != -1) {
        if (videoParameters.system == PAL) frameNumber += clvTimeCode.hours * 3600 * 25;
        else frameNumber += clvTimeCode.hours * 3600 * 30;
    }

    if (clvTimeCode.minutes != -1) {
        if (videoParameters.system == PAL) frameNumber += clvTimeCode.minutes * 60 * 25;
        else frameNumber += clvTimeCode.minutes * 60 * 30;
    }

    if (clvTimeCode.seconds != -1) {
        if (videoParameters.system == PAL) frameNumber += clvTimeCode.seconds * 25;
        else frameNumber += clvTimeCode.seconds * 30;
    }

    if (clvTimeCode.pictureNumber != -1) {
        frameNumber += clvTimeCode.pictureNumber;
    }

    return frameNumber;
}

// Method to convert a frame number into an equivalent CLV timecode
LdDecodeMetaData::ClvTimecode LdDecodeMetaData::convertFrameNumberToClvTimecode(qint32 frameNumber)
{
    ClvTimecode clvTimecode;

    clvTimecode.hours = 0;
    clvTimecode.minutes = 0;
    clvTimecode.seconds = 0;
    clvTimecode.pictureNumber = 0;

    if (getVideoParameters().system == PAL) {
        clvTimecode.hours = frameNumber / (3600 * 25);
        frameNumber -= clvTimecode.hours * (3600 * 25);

        clvTimecode.minutes = frameNumber / (60 * 25);
        frameNumber -= clvTimecode.minutes * (60 * 25);

        clvTimecode.seconds = frameNumber / 25;
        frameNumber -= clvTimecode.seconds * 25;

        clvTimecode.pictureNumber = frameNumber;
    } else {
        clvTimecode.hours = frameNumber / (3600 * 30);
        frameNumber -= clvTimecode.hours * (3600 * 30);

        clvTimecode.minutes = frameNumber / (60 * 30);
        frameNumber -= clvTimecode.minutes * (60 * 30);

        clvTimecode.seconds = frameNumber / 30;
        frameNumber -= clvTimecode.seconds * 30;

        clvTimecode.pictureNumber = frameNumber;
    }

    return clvTimecode;
}

// Method to return a description string for the current video format
QString LdDecodeMetaData::getVideoSystemDescription() const
{
    return getSystemDefaults(videoParameters).name;
}

// Private method to generate a map of the PCM audio data (used by the sourceAudio library)
// Note: That the map unit is "stereo sample pairs"; so each unit represents 2 16-bit samples
// for a total of 4 bytes per unit.
void LdDecodeMetaData::generatePcmAudioMap()
{
    pcmAudioFieldStartSampleMap.clear();
    pcmAudioFieldLengthMap.clear();

    qDebug() << "LdDecodeMetaData::generatePcmAudioMap(): Generating PCM audio map...";

    // Get the number of fields and resize the maps
    qint32 numberOfFields = getVideoParameters().numberOfSequentialFields;
    pcmAudioFieldStartSampleMap.resize(numberOfFields + 1);
    pcmAudioFieldLengthMap.resize(numberOfFields + 1);

    for (qint32 fieldNo = 0; fieldNo < numberOfFields; fieldNo++) {
        // Each audio sample is 16 bit - and there are 2 samples per stereo pair
        pcmAudioFieldLengthMap[fieldNo] = static_cast<qint32>(getField(fieldNo+1).audioSamples);

        if (fieldNo == 0) {
            // First field starts at 0 units
            pcmAudioFieldStartSampleMap[fieldNo] = 0;
        } else {
            // Every following field's start position is the start+length of the previous
            pcmAudioFieldStartSampleMap[fieldNo] = pcmAudioFieldStartSampleMap[fieldNo - 1] + pcmAudioFieldLengthMap[fieldNo - 1];
        }
    }
}

// Method to get the start sample location of the specified sequential field number
qint32 LdDecodeMetaData::getFieldPcmAudioStart(qint32 sequentialFieldNumber)
{
    if (pcmAudioFieldStartSampleMap.size() < sequentialFieldNumber) return -1;
    // Field numbers are 1 indexed, but our map is 0 indexed
    return pcmAudioFieldStartSampleMap[sequentialFieldNumber - 1];
}

// Method to get the sample length of the specified sequential field number
qint32 LdDecodeMetaData::getFieldPcmAudioLength(qint32 sequentialFieldNumber)
{
    if (pcmAudioFieldLengthMap.size() < sequentialFieldNumber) return -1;
    // Field numbers are 1 indexed, but our map is 0 indexed
    return pcmAudioFieldLengthMap[sequentialFieldNumber - 1];
}
