/************************************************************************

    lddecodemetadata.h

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

#ifndef LDDECODEMETADATA_H
#define LDDECODEMETADATA_H

#include <QVector>
#include <QTemporaryFile>
#include <QDebug>

#include "../JsonWax/JsonWax.h"
#include "vbidecoder.h"
#include "dropouts.h"

class LdDecodeMetaData
{

public:

    // VBI Metadata definition
    struct Vbi {
        Vbi() : inUse(false) {}

        bool inUse;
        QVector<qint32> vbiData;
    };

    // Video metadata definition
    struct VideoParameters {
        qint32 numberOfSequentialFields;

        bool isSourcePal;
        bool isSubcarrierLocked;
        bool isWidescreen;

        qint32 colourBurstStart;
        qint32 colourBurstEnd;
        qint32 activeVideoStart;
        qint32 activeVideoEnd;

        qint32 white16bIre;
        qint32 black16bIre;

        qint32 fieldWidth;
        qint32 fieldHeight;
        qint32 sampleRate;
        qint32 fsc;

        bool isMapped;

        // Note: These are psuedo metadata items - The values are populated by the library
        // These are half-open ranges, where lines are numbered sequentially from 0 within
        // each field or interlaced frame
        qint32 firstActiveFieldLine;
        qint32 lastActiveFieldLine;
        qint32 firstActiveFrameLine;
        qint32 lastActiveFrameLine;
    };

    // VITS metrics metadata definition
    struct VitsMetrics {
        VitsMetrics() : inUse(false), wSNR(0), bPSNR(0) {}

        bool inUse;
        qreal wSNR;
        qreal bPSNR;
    };

    // NTSC Specific metadata definition
    struct Ntsc {
        Ntsc() : inUse(false), isFmCodeDataValid(false), fmCodeData(0), fieldFlag(false),
            whiteFlag(false), ccData0(0), ccData1(0) {}

        bool inUse;
        bool isFmCodeDataValid;
        qint32 fmCodeData;
        bool fieldFlag;
        bool whiteFlag;
        qint32 ccData0;
        qint32 ccData1;
    };

    // PCM sound metadata definition
    struct PcmAudioParameters {
        qint32 sampleRate;
        bool isLittleEndian;
        bool isSigned;
        qint32 bits;
    };

    // Field metadata definition
    struct Field {
        Field() : seqNo(0), isFirstField(false), syncConf(0), medianBurstIRE(0),
            fieldPhaseID(0), audioSamples(0), pad(false) {}

        qint32 seqNo;       // Note: This is the unique primary-key
        bool isFirstField;
        qint32 syncConf;
        qreal medianBurstIRE;
        qint32 fieldPhaseID;
        qint32 audioSamples;

        VitsMetrics vitsMetrics;
        Vbi vbi;
        Ntsc ntsc;
        DropOuts dropOuts;
        bool pad;
    };

    // Overall metadata definition
    struct MetaData {
        VideoParameters videoParameters;
        PcmAudioParameters pcmAudioParameters;
        QVector<Field> fields;
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

    bool read(QString fileName);
    bool write(QString fileName);

    VideoParameters getVideoParameters();
    void setVideoParameters (VideoParameters _videoParameters);

    PcmAudioParameters getPcmAudioParameters();
    void setPcmAudioParameters(PcmAudioParameters _pcmAudioParam);

    // Get field metadata
    Field getField(qint32 sequentialFieldNumber);
    VitsMetrics getFieldVitsMetrics(qint32 sequentialFieldNumber);
    Vbi getFieldVbi(qint32 sequentialFieldNumber);
    Ntsc getFieldNtsc(qint32 sequentialFieldNumber);
    DropOuts getFieldDropOuts(qint32 sequentialFieldNumber);

    // Set field metadata
    void updateField(Field _field, qint32 sequentialFieldNumber);
    void updateFieldVitsMetrics(LdDecodeMetaData::VitsMetrics _vitsMetrics, qint32 sequentialFieldNumber);
    void updateFieldVbi(LdDecodeMetaData::Vbi _vbi, qint32 sequentialFieldNumber);
    void updateFieldNtsc(LdDecodeMetaData::Ntsc _ntsc, qint32 sequentialFieldNumber);
    void updateFieldDropOuts(DropOuts _dropOuts, qint32 sequentialFieldNumber);
    void clearFieldDropOuts(qint32 sequentialFieldNumber);

    void appendField(Field _field);

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

private:
    JsonWax json;
    bool isFirstFieldFirst;
    QVector<qint32> pcmAudioFieldStartSampleMap;
    QVector<qint32> pcmAudioFieldLengthMap;

    qint32 getFieldNumber(qint32 frameNumber, qint32 field);
    void generatePcmAudioMap();
};

#endif // LDDECODEMETADATA_H
