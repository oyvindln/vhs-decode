/************************************************************************

    lddecodemetadata.h

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

#ifndef LDDECODEMETADATA_H
#define LDDECODEMETADATA_H

#include <QString>
#include <QVector>
#include <QTemporaryFile>
#include <QDebug>
#include <array>

#include "vbidecoder.h"
#include "dropouts.h"

class JsonReader;
class JsonWriter;

// The video system (combination of a line standard and a colour standard)
// Note: If you update this, be sure to update VIDEO_SYSTEM_DEFAULTS also
enum VideoSystem {
    PAL = 0,    // 625-line PAL
    NTSC,       // 525-line NTSC
    PAL_M,      // 525-line PAL
};

bool parseVideoSystemName(QString name, VideoSystem &system);

class LdDecodeMetaData
{

public:
    // VBI Metadata definition
    struct Vbi {
        bool inUse = false;
        std::array<qint32, 3> vbiData { 0, 0, 0 };

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // Video metadata definition
    struct VideoParameters {
        // -- Members stored in the JSON metadata --

        qint32 numberOfSequentialFields = -1;

        VideoSystem system = NTSC;
        bool isSubcarrierLocked = false;
        bool isWidescreen = false;

        qint32 colourBurstStart = -1;
        qint32 colourBurstEnd = -1;
        qint32 activeVideoStart = -1;
        qint32 activeVideoEnd = -1;

        qint32 white16bIre = -1;
        qint32 black16bIre = -1;

        qint32 fieldWidth = -1;
        qint32 fieldHeight = -1;
        double sampleRate = -1.0;

        bool isMapped = false;

        QString gitBranch;
        QString gitCommit;

        // -- Members set by the library --

        // Colour subcarrier frequency in Hz
        double fSC = -1.0;

        // The range of active lines within a frame.
        // This is the same information represented in two different ways, for
        // field- and frame-based processing respectively; the field range
        // should cover the active lines in both fields of a frame.
        // These are half-open ranges, where lines are numbered sequentially
        // from 1 within each field or interlaced frame.
        qint32 firstActiveFieldLine = -1;
        qint32 lastActiveFieldLine = -1;
        qint32 firstActiveFrameLine = -1;
        qint32 lastActiveFrameLine = -1;

        // Flags if our data has been initialized yet
        bool isValid = false;

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // Specification for customising the range of active lines in VideoParameters.
    // -1 for any of these means to use the default for the standard.
    struct LineParameters {
        qint32 firstActiveFieldLine = -1;
        qint32 lastActiveFieldLine = -1;
        qint32 firstActiveFrameLine = -1;
        qint32 lastActiveFrameLine = -1;

        void applyTo(VideoParameters &videoParameters);
    };

    // VITS metrics metadata definition
    struct VitsMetrics {
        bool inUse = false;
        double wSNR = 0.0;
        double bPSNR = 0.0;

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // NTSC Specific metadata definition
    struct ClosedCaption;
    struct Ntsc {
        bool inUse = false;
        bool isFmCodeDataValid = false;
        qint32 fmCodeData = 0;
        bool fieldFlag = false;
        bool whiteFlag = false;

        void read(JsonReader &reader, ClosedCaption &closedCaption);
        void write(JsonWriter &writer) const;
    };

    // VITC timecode definition
    struct Vitc {
        bool inUse = false;

        // Just the VITC data, without the sync bits or CRC.
        // vitcData[0]'s LSB is bit 2; vitcData[7]'s MSB is bit 79.
        std::array<qint32, 8> vitcData;

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // Closed Caption definition
    struct ClosedCaption {
        bool inUse = false;

        qint32 data0 = -1;
        qint32 data1 = -1;

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // PCM sound metadata definition
    struct PcmAudioParameters {
        double sampleRate = -1.0;
        bool isLittleEndian = false;
        bool isSigned = false;
        qint32 bits = -1;

        // Flags if our data has been initialized yet
        bool isValid = false;

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // Field metadata definition
    struct Field {
        qint32 seqNo = 0;   // Note: This is the unique primary-key
        bool isFirstField = false;
        qint32 syncConf = 0;
        double medianBurstIRE = 0.0;
        qint32 fieldPhaseID = -1;
        qint32 audioSamples = -1;

        VitsMetrics vitsMetrics;
        Vbi vbi;
        Ntsc ntsc;
        Vitc vitc;
        ClosedCaption closedCaption;
        DropOuts dropOuts;
        bool pad = false;

        qint32 diskLoc = -1;
        qint32 fileLoc = -1;
        qint32 decodeFaults = -1;
        qint32 efmTValues = -1;

        void read(JsonReader &reader);
        void write(JsonWriter &writer) const;
    };

    // CLV timecode (used by frame number conversion methods)
    struct ClvTimecode {
        qint32 hours;
        qint32 minutes;
        qint32 seconds;
        qint32 pictureNumber;
    };

    LdDecodeMetaData();

    // Prevent copying or assignment
    LdDecodeMetaData(const LdDecodeMetaData &) = delete;
    LdDecodeMetaData& operator=(const LdDecodeMetaData &) = delete;

    void clear();
    bool read(QString fileName);
    bool write(QString fileName) const;
    void readFields(JsonReader &reader);
    void writeFields(JsonWriter &writer) const;

    const VideoParameters &getVideoParameters();
    void setVideoParameters(const VideoParameters &videoParameters);

    const PcmAudioParameters &getPcmAudioParameters();
    void setPcmAudioParameters(const PcmAudioParameters &pcmAudioParam);

    // Handle line parameters
    void processLineParameters(LdDecodeMetaData::LineParameters &_lineParameters);

    // Get field metadata
    const Field &getField(qint32 sequentialFieldNumber);
    const VitsMetrics &getFieldVitsMetrics(qint32 sequentialFieldNumber);
    const Vbi &getFieldVbi(qint32 sequentialFieldNumber);
    const Ntsc &getFieldNtsc(qint32 sequentialFieldNumber);
    const Vitc &getFieldVitc(qint32 sequentialFieldNumber);
    const ClosedCaption &getFieldClosedCaption(qint32 sequentialFieldNumber);
    const DropOuts &getFieldDropOuts(qint32 sequentialFieldNumber);

    // Set field metadata
    void updateField(const Field &field, qint32 sequentialFieldNumber);
    void updateFieldVitsMetrics(const LdDecodeMetaData::VitsMetrics &vitsMetrics, qint32 sequentialFieldNumber);
    void updateFieldVbi(const LdDecodeMetaData::Vbi &vbi, qint32 sequentialFieldNumber);
    void updateFieldNtsc(const LdDecodeMetaData::Ntsc &ntsc, qint32 sequentialFieldNumber);
    void updateFieldVitc(const LdDecodeMetaData::Vitc &vitc, qint32 sequentialFieldNumber);
    void updateFieldClosedCaption(const LdDecodeMetaData::ClosedCaption &closedCaption, qint32 sequentialFieldNumber);
    void updateFieldDropOuts(const DropOuts &dropOuts, qint32 sequentialFieldNumber);
    void clearFieldDropOuts(qint32 sequentialFieldNumber);

    void appendField(const Field &field);

    void setNumberOfFields(qint32 numberOfFields);
    qint32 getNumberOfFields();
    qint32 getNumberOfFrames();
    qint32 getFirstFieldNumber(qint32 frameNumber);
    qint32 getSecondFieldNumber(qint32 frameNumber);

    void setIsFirstFieldFirst(bool flag);
    bool getIsFirstFieldFirst();

    qint32 convertClvTimecodeToFrameNumber(LdDecodeMetaData::ClvTimecode clvTimeCode);
    LdDecodeMetaData::ClvTimecode convertFrameNumberToClvTimecode(qint32 clvFrameNumber);

    // PCM Analogue audio helper methods
    qint32 getFieldPcmAudioStart(qint32 sequentialFieldNumber);
    qint32 getFieldPcmAudioLength(qint32 sequentialFieldNumber);

    // Video system helper methods
    QString getVideoSystemDescription() const;

private:
    bool isFirstFieldFirst;
    VideoParameters videoParameters;
    PcmAudioParameters pcmAudioParameters;
    QVector<Field> fields;
    QVector<qint32> pcmAudioFieldStartSampleMap;
    QVector<qint32> pcmAudioFieldLengthMap;

    void initialiseVideoSystemParameters();
    qint32 getFieldNumber(qint32 frameNumber, qint32 field);
    void generatePcmAudioMap();
};

#endif // LDDECODEMETADATA_H
