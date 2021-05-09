/************************************************************************

    lddecodemetadata.cpp

    ld-decode-tools TBC library
    Copyright (C) 2018-2020 Simon Inns

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

LdDecodeMetaData::LdDecodeMetaData()
{
    // Set defaults
    isFirstFieldFirst = false;
}

// This method opens the JSON metadata file and reads the content into the
// metadata structure read for use
bool LdDecodeMetaData::read(QString fileName)
{
    // Open the JSON file
    qDebug() << "LdDecodeMetaData::read(): Loading JSON file" << fileName;
    if (!json.loadFile(fileName)) {
        qCritical() << "JSON wax library error:" << json.errorMsg();
        qCritical("Opening JSON file failed: JSON file cannot be opened/does not exist");

        return false;
    }

    // Default to the standard still-frame field order (of first field first)
    isFirstFieldFirst = true;

    // Generate the PCM audio map based on the field metadata
    generatePcmAudioMap();

    return true;
}

// This method copies the metadata structure into a JSON metadata file
bool LdDecodeMetaData::write(QString fileName)
{
    // Write the JSON object
    qDebug() << "LdDecodeMetaData::write(): Writing JSON metadata to:" << fileName;
    if (!json.saveAs(fileName, JsonWax::Compact)) {
        qCritical("Writing JSON metadata file failed!");
        return false;
    }

    return true;
}

// This method returns the videoParameters metadata
LdDecodeMetaData::VideoParameters LdDecodeMetaData::getVideoParameters()
{
    VideoParameters videoParameters;

    // Read the video paramters
    if (json.size({"videoParameters"}) > 0) {
        videoParameters.numberOfSequentialFields = json.value({"videoParameters", "numberOfSequentialFields"}).toInt();
        videoParameters.isSourcePal = json.value({"videoParameters", "isSourcePal"}).toBool();
        videoParameters.isSourcePalM = json.value({"videoParameters", "isSourcePalM"}, false).toBool();
        videoParameters.isSubcarrierLocked = json.value({"videoParameters", "isSubcarrierLocked"}).toBool();
        videoParameters.isWidescreen = json.value({"videoParameters", "isWidescreen"}).toBool();

        videoParameters.colourBurstStart = json.value({"videoParameters", "colourBurstStart"}).toInt();
        videoParameters.colourBurstEnd = json.value({"videoParameters", "colourBurstEnd"}).toInt();
        videoParameters.activeVideoStart = json.value({"videoParameters", "activeVideoStart"}).toInt();
        videoParameters.activeVideoEnd = json.value({"videoParameters", "activeVideoEnd"}).toInt();

        videoParameters.white16bIre = json.value({"videoParameters", "white16bIre"}).toInt();
        videoParameters.black16bIre = json.value({"videoParameters", "black16bIre"}).toInt();

        videoParameters.fieldWidth = json.value({"videoParameters", "fieldWidth"}).toInt();
        videoParameters.fieldHeight = json.value({"videoParameters", "fieldHeight"}).toInt();
        videoParameters.sampleRate = json.value({"videoParameters", "sampleRate"}).toInt();
        videoParameters.fsc = json.value({"videoParameters", "fsc"}).toInt();

        videoParameters.isMapped = json.value({"videoParameters", "isMapped"}).toBool();
    } else {
        qCritical("JSON file invalid: videoParameters object is not defined");
        return videoParameters;
    }

    // Add in the active field line range psuedo-metadata
    if (videoParameters.isSourcePal && !videoParameters.isSourcePalM) {
        // PAL
        videoParameters.firstActiveFieldLine = 22;
        videoParameters.lastActiveFieldLine = 308;

        // Interlaced line 44 is PAL line 23 (the first active half-line)
        videoParameters.firstActiveFrameLine = 44;
        // Interlaced line 619 is PAL line 623 (the last active half-line)
        videoParameters.lastActiveFrameLine = 620;
    } else {
        // NTSC
        videoParameters.firstActiveFieldLine = 20;
        videoParameters.lastActiveFieldLine = 259;

        // Interlaced line 40 is NTSC line 21 (the closed-caption line before the first active half-line)
        videoParameters.firstActiveFrameLine = 40;
        // Interlaced line 524 is NTSC line 263 (the last active half-line).
        videoParameters.lastActiveFrameLine = 525;
    }

    return videoParameters;
}

// This method sets the videoParameters metadata
void LdDecodeMetaData::setVideoParameters (LdDecodeMetaData::VideoParameters _videoParameters)
{
    // Write the video parameters
    json.setValue({"videoParameters", "numberOfSequentialFields"}, getNumberOfFields());
    json.setValue({"videoParameters", "isSourcePal"}, _videoParameters.isSourcePal);
    json.setValue({"videoParameters", "isSubcarrierLocked"}, _videoParameters.isSubcarrierLocked);
    json.setValue({"videoParameters", "isWidescreen"}, _videoParameters.isWidescreen);

    json.setValue({"videoParameters", "colourBurstStart"}, _videoParameters.colourBurstStart);
    json.setValue({"videoParameters", "colourBurstEnd"}, _videoParameters.colourBurstEnd);
    json.setValue({"videoParameters", "activeVideoStart"}, _videoParameters.activeVideoStart);
    json.setValue({"videoParameters", "activeVideoEnd"}, _videoParameters.activeVideoEnd);

    json.setValue({"videoParameters", "white16bIre"}, _videoParameters.white16bIre);
    json.setValue({"videoParameters", "black16bIre"}, _videoParameters.black16bIre);

    json.setValue({"videoParameters", "fieldWidth"}, _videoParameters.fieldWidth);
    json.setValue({"videoParameters", "fieldHeight"}, _videoParameters.fieldHeight);
    json.setValue({"videoParameters", "sampleRate"}, _videoParameters.sampleRate);
    json.setValue({"videoParameters", "fsc"}, _videoParameters.fsc);

    json.setValue({"videoParameters", "isMapped"}, _videoParameters.isMapped);
}

// This method returns the pcmAudioParameters metadata
LdDecodeMetaData::PcmAudioParameters LdDecodeMetaData::getPcmAudioParameters()
{
    PcmAudioParameters pcmAudioParameters;

    if (json.size({"pcmAudioParameters"}) > 0) {
        // Read the PCM audio data
        pcmAudioParameters.sampleRate = json.value({"pcmAudioParameters", "sampleRate"}).toInt();
        pcmAudioParameters.isLittleEndian = json.value({"pcmAudioParameters", "isLittleEndian"}).toBool();
        pcmAudioParameters.isSigned = json.value({"pcmAudioParameters", "isSigned"}).toBool();
        pcmAudioParameters.bits = json.value({"pcmAudioParameters", "bits"}).toInt();
    } else {
        qCritical("JSON file invalid: pcmAudioParameters is not defined");
        return pcmAudioParameters;
    }

    return pcmAudioParameters;
}

// This method sets the pcmAudioParameters metadata
void LdDecodeMetaData::setPcmAudioParameters(LdDecodeMetaData::PcmAudioParameters _pcmAudioParam)
{
    json.setValue({"pcmAudioParameters", "sampleRate"}, _pcmAudioParam.sampleRate);
    json.setValue({"pcmAudioParameters", "isLittleEndian"}, _pcmAudioParam.isLittleEndian);
    json.setValue({"pcmAudioParameters", "isSigned"}, _pcmAudioParam.isSigned);
    json.setValue({"pcmAudioParameters", "bits"}, _pcmAudioParam.bits);
}

// This method gets the metadata for the specified sequential field number (indexed from 1 (not 0!))
LdDecodeMetaData::Field LdDecodeMetaData::getField(qint32 sequentialFieldNumber)
{
    Field field;
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::getField(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    // Primary field values
    field.seqNo = json.value({"fields", fieldNumber, "seqNo"}).toInt();
    field.isFirstField = json.value({"fields", fieldNumber, "isFirstField"}).toBool();
    field.syncConf = json.value({"fields", fieldNumber, "syncConf"}).toInt();
    field.medianBurstIRE = json.value({"fields", fieldNumber, "medianBurstIRE"}).toDouble();
    field.fieldPhaseID = json.value({"fields", fieldNumber, "fieldPhaseID"}).toInt();
    field.audioSamples = json.value({"fields", fieldNumber, "audioSamples"}).toInt();

    // VITS metrics values
    field.vitsMetrics = getFieldVitsMetrics(sequentialFieldNumber);

    // VBI values
    field.vbi = getFieldVbi(sequentialFieldNumber);

    // NTSC values
    field.ntsc = getFieldNtsc(sequentialFieldNumber);

    // dropOuts values
    field.dropOuts = getFieldDropOuts(sequentialFieldNumber);

    // Padding flag
    field.pad = json.value({"fields", fieldNumber, "pad"}).toBool();

    return field;
}

// This method gets the VITS metrics metadata for the specified sequential field number
LdDecodeMetaData::VitsMetrics LdDecodeMetaData::getFieldVitsMetrics(qint32 sequentialFieldNumber)
{
    VitsMetrics vitsMetrics;
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::getFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (json.size({"fields", fieldNumber, "vitsMetrics"}) > 0) {
        vitsMetrics.inUse = true;
        vitsMetrics.wSNR = json.value({"fields", fieldNumber, "vitsMetrics", "wSNR"}).toReal();
        vitsMetrics.bPSNR = json.value({"fields", fieldNumber, "vitsMetrics", "bPSNR"}).toReal();
    } else {
        // Mark VITS metrics as undefined
        vitsMetrics.inUse = false;
    }

    return vitsMetrics;
}

// This method gets the VBI metadata for the specified sequential field number
LdDecodeMetaData::Vbi LdDecodeMetaData::getFieldVbi(qint32 sequentialFieldNumber)
{
    Vbi vbi;
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::getFieldVbi(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (json.size({"fields", fieldNumber, "vbi"}) > 0) {
        // Mark VBI as in use
        vbi.inUse = true;

        vbi.vbiData.append(json.value({"fields", fieldNumber, "vbi", "vbiData", 0}).toInt()); // Line 16
        vbi.vbiData.append(json.value({"fields", fieldNumber, "vbi", "vbiData", 1}).toInt()); // Line 17
        vbi.vbiData.append(json.value({"fields", fieldNumber, "vbi", "vbiData", 2}).toInt()); // Line 18
    } else {
        // Mark VBI as undefined
        vbi.inUse = false;

        // Resize the VBI data fields to prevent assert issues downstream
        vbi.vbiData.resize(3);
    }

    return vbi;
}

// This method gets the NTSC metadata for the specified sequential field number
LdDecodeMetaData::Ntsc LdDecodeMetaData::getFieldNtsc(qint32 sequentialFieldNumber)
{
    Ntsc ntsc;
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::getFieldNtsc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (json.size({"fields", fieldNumber, "ntsc"}) > 0) {
        // Mark as in use
        ntsc.inUse = true;

        ntsc.isFmCodeDataValid = json.value({"fields", fieldNumber, "ntsc", "isFmCodeDataValid"}).toBool();
        ntsc.fmCodeData = json.value({"fields", fieldNumber, "ntsc", "fmCodeData"}).toInt();
        ntsc.fieldFlag = json.value({"fields", fieldNumber, "ntsc", "fieldFlag"}).toBool();
        ntsc.whiteFlag = json.value({"fields", fieldNumber, "ntsc", "whiteFlag"}).toBool();
        ntsc.ccData0 = json.value({"fields", fieldNumber, "ntsc", "ccData0"}).toInt();
        ntsc.ccData1 = json.value({"fields", fieldNumber, "ntsc", "ccData1"}).toInt();
    } else {
        // Mark ntscSpecific as undefined
        ntsc.inUse = false;
    }

    return ntsc;
}

// This method gets the drop-out metadata for the specified sequential field number
DropOuts LdDecodeMetaData::getFieldDropOuts(qint32 sequentialFieldNumber)
{
    DropOuts dropOuts;
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::getFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    // Get the JSON array sizes
    qint32 startxSize = json.size({"fields", fieldNumber, "dropOuts", "startx"});
    qint32 endxSize = json.size({"fields", fieldNumber, "dropOuts", "endx"});
    qint32 fieldLinesSize = json.size({"fields", fieldNumber, "dropOuts", "fieldLines"});

    // Ensure that all three objects are the same size
    if (startxSize != endxSize && startxSize != fieldLinesSize) {
        qCritical("JSON file is invalid: Dropouts object is illegal");
    }

    if (startxSize > 0) {
        for (qint32 doCounter = 0; doCounter < startxSize; doCounter++) {
            dropOuts.append(json.value({"fields", fieldNumber, "dropOuts", "startx", doCounter}).toInt(),
                            json.value({"fields", fieldNumber, "dropOuts", "endx", doCounter}).toInt(),
                            json.value({"fields", fieldNumber, "dropOuts", "fieldLine", doCounter}).toInt());
        }
    } else {
        dropOuts.clear();
    }

    return dropOuts;
}

// This method sets the field metadata for a field
void LdDecodeMetaData::updateField(LdDecodeMetaData::Field _field, qint32 sequentialFieldNumber)
{
    if (sequentialFieldNumber < 1) {
        qCritical() << "LdDecodeMetaData::updateFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    qint32 fieldNumber = sequentialFieldNumber - 1;

    // Write the field data
    json.setValue({"fields", fieldNumber, "seqNo"}, sequentialFieldNumber);
    json.setValue({"fields", fieldNumber, "isFirstField"}, _field.isFirstField);
    json.setValue({"fields", fieldNumber, "syncConf"}, _field.syncConf);
    json.setValue({"fields", fieldNumber, "medianBurstIRE"}, _field.medianBurstIRE);
    json.setValue({"fields", fieldNumber, "fieldPhaseID"}, _field.fieldPhaseID);
    json.setValue({"fields", fieldNumber, "audioSamples"}, _field.audioSamples);

    // Write the VITS metrics data if in use
    updateFieldVitsMetrics(_field.vitsMetrics, sequentialFieldNumber);

    // Write the VBI data if in use
    updateFieldVbi(_field.vbi, sequentialFieldNumber);

    // Write the NTSC specific record if in use
    updateFieldNtsc(_field.ntsc, sequentialFieldNumber);

    // Write the drop-out records
    updateFieldDropOuts(_field.dropOuts, sequentialFieldNumber);

    // Padding flag
    json.setValue({"fields", fieldNumber, "pad"}, _field.pad);
}

// This method sets the field VBI metadata for a field
void LdDecodeMetaData::updateFieldVitsMetrics(LdDecodeMetaData::VitsMetrics _vitsMetrics, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() + 1 || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::updateFieldVitsMetrics(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (_vitsMetrics.inUse) {
        json.setValue({"fields", fieldNumber, "vitsMetrics", "wSNR"}, _vitsMetrics.wSNR);
        json.setValue({"fields", fieldNumber, "vitsMetrics", "bPSNR"}, _vitsMetrics.bPSNR);
    }
}

// This method sets the field VBI metadata for a field
void LdDecodeMetaData::updateFieldVbi(LdDecodeMetaData::Vbi _vbi, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() + 1 || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::updateFieldVbi(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (_vbi.inUse) {
        // Validate the VBI data array
        if (_vbi.vbiData.size() != 3) {
            qDebug() << "LdDecodeMetaData::write(): Invalid vbiData array!  Setting to -1";
            _vbi.vbiData.resize(3);
            _vbi.vbiData[0] = -1;
            _vbi.vbiData[1] = -1;
            _vbi.vbiData[2] = -1;
        }

        json.setValue({"fields", fieldNumber, "vbi", "vbiData", 0}, _vbi.vbiData[0]);
        json.setValue({"fields", fieldNumber, "vbi", "vbiData", 1}, _vbi.vbiData[1]);
        json.setValue({"fields", fieldNumber, "vbi", "vbiData", 2}, _vbi.vbiData[2]);
    }
}

// This method sets the field NTSC metadata for a field
void LdDecodeMetaData::updateFieldNtsc(LdDecodeMetaData::Ntsc _ntsc, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() + 1 || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::updateFieldNtsc(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (_ntsc.inUse) {
        json.setValue({"fields", fieldNumber, "ntsc", "isFmCodeDataValid"}, _ntsc.isFmCodeDataValid);
        if (_ntsc.isFmCodeDataValid)
            json.setValue({"fields", fieldNumber, "ntsc", "fmCodeData"}, _ntsc.fmCodeData);
        else json.setValue({"fields", fieldNumber, "ntsc", "fmCodeData"}, -1);
        json.setValue({"fields", fieldNumber, "ntsc", "fieldFlag"}, _ntsc.fieldFlag);
        json.setValue({"fields", fieldNumber, "ntsc", "whiteFlag"}, _ntsc.whiteFlag);
        json.setValue({"fields", fieldNumber, "ntsc", "ccData0"}, _ntsc.ccData0);
        json.setValue({"fields", fieldNumber, "ntsc", "ccData1"}, _ntsc.ccData1);
    }
}

// This method sets the field dropout metadata for a field
void LdDecodeMetaData::updateFieldDropOuts(DropOuts _dropOuts, qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() + 1 || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::updateFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    if (_dropOuts.size() != 0) {
        // Populate the arrays with the drop out metadata
        for (qint32 doCounter = 0; doCounter < _dropOuts.size(); doCounter++) {
            json.setValue({"fields", fieldNumber, "dropOuts", "startx", doCounter}, _dropOuts.startx(doCounter));
            json.setValue({"fields", fieldNumber, "dropOuts", "endx", doCounter}, _dropOuts.endx(doCounter));
            json.setValue({"fields", fieldNumber, "dropOuts", "fieldLine", doCounter}, _dropOuts.fieldLine(doCounter));
        }
    } else {
        // If updated dropouts is empty, clear the field's dropouts
        clearFieldDropOuts(sequentialFieldNumber);
    }
}

// This method clears the field dropout metadata for a field
void LdDecodeMetaData::clearFieldDropOuts(qint32 sequentialFieldNumber)
{
    qint32 fieldNumber = sequentialFieldNumber - 1;

    if (fieldNumber >= getNumberOfFields() + 1 || fieldNumber < 0) {
        qCritical() << "LdDecodeMetaData::updateFieldDropOuts(): Requested field number" << sequentialFieldNumber << "out of bounds!";
    }

    json.remove({"fields", fieldNumber, "dropOuts", "startx"});
    json.remove({"fields", fieldNumber, "dropOuts", "endx"});
    json.remove({"fields", fieldNumber, "dropOuts", "fieldLine"});
}

// This method appends a new field to the existing metadata
void LdDecodeMetaData::appendField(LdDecodeMetaData::Field _field)
{
    // Get the field number (and adjust to 0 if there are no fields in the JSON)
    qint32 fieldNumber = getNumberOfFields();
    if (fieldNumber < 0) fieldNumber = 0;

    updateField(_field, fieldNumber + 1);
}

// Method to get the available number of fields (according to the metadata)
qint32 LdDecodeMetaData::getNumberOfFields()
{
    return json.size({"fields"});
}

// Method to set the available number of fields
void LdDecodeMetaData::setNumberOfFields(qint32 numberOfFields)
{
    json.setValue({"videoParameters", "numberOfSequentialFields"}, numberOfFields);
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
        if (videoParameters.isSourcePal) frameNumber += clvTimeCode.hours * 3600 * 25;
        else frameNumber += clvTimeCode.hours * 3600 * 30;
    }

    if (clvTimeCode.minutes != -1) {
        if (videoParameters.isSourcePal) frameNumber += clvTimeCode.minutes * 60 * 25;
        else frameNumber += clvTimeCode.minutes * 60 * 30;
    }

    if (clvTimeCode.seconds != -1) {
        if (videoParameters.isSourcePal) frameNumber += clvTimeCode.seconds * 25;
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

    if (getVideoParameters().isSourcePal) {
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
            pcmAudioFieldStartSampleMap[fieldNo] = pcmAudioFieldStartSampleMap[fieldNo - 1] + pcmAudioFieldLengthMap[fieldNo];
        }
    }
}

// Method to get the start sample location of the specified sequential field number
qint32 LdDecodeMetaData::getFieldPcmAudioStart(qint32 sequentialFieldNumber)
{
    if (pcmAudioFieldStartSampleMap.size() < sequentialFieldNumber) return -1;
    return pcmAudioFieldStartSampleMap[sequentialFieldNumber];
}

// Method to get the sample length of the specified sequential field number
qint32 LdDecodeMetaData::getFieldPcmAudioLength(qint32 sequentialFieldNumber)
{
    if (pcmAudioFieldLengthMap.size() < sequentialFieldNumber) return -1;
    return pcmAudioFieldLengthMap[sequentialFieldNumber];
}
