/************************************************************************

    stackingpool.cpp

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
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

#ifndef STACKINGPOOL_H
#define STACKINGPOOL_H

#include <QObject>
#include <QAtomicInt>
#include <QElapsedTimer>
#include <QMutex>
#include <QThread>

#include "sourcevideo.h"
#include "lddecodemetadata.h"
#include "stacker.h"

class StackingPool : public QObject
{
    Q_OBJECT
public:
    explicit StackingPool(QString _outputFilename, QString _outputJsonFilename,
                           qint32 _maxThreads, QVector<LdDecodeMetaData *> &_ldDecodeMetaData, QVector<SourceVideo *> &_sourceVideos,
                           bool _reverse, bool _noDiffDod, bool _passThrough, QObject *parent = nullptr);

    bool process();

    // Member functions used by worker threads
    bool getInputFrame(qint32& frameNumber,
                       QVector<qint32> &firstFieldNumber, QVector<SourceVideo::Data> &firstFieldVideoData, QVector<LdDecodeMetaData::Field> &firstFieldMetadata,
                       QVector<qint32> &secondFieldNumber, QVector<SourceVideo::Data> &secondFieldVideoData, QVector<LdDecodeMetaData::Field> &secondFieldMetadata,
                       QVector<LdDecodeMetaData::VideoParameters> &videoParameters,
                       bool& _reverse, bool &_noDiffDod, bool &_passThrough, QVector<qint32> &availableSourcesForFrame);

    bool setOutputFrame(qint32 frameNumber,
                        SourceVideo::Data firstTargetFieldData, SourceVideo::Data secondTargetFieldData,
                        qint32 firstFieldSeqNo, qint32 secondFieldSeqNo,
                        DropOuts firstTargetFieldDropOuts, DropOuts secondTargetFieldDropouts);

private:
    QString outputFilename;
    QString outputJsonFilename;
    qint32 maxThreads;
    bool reverse;
    bool noDiffDod;
    bool passThrough;
    QElapsedTimer totalTimer;

    // Atomic abort flag shared by worker threads; workers watch this, and shut
    // down as soon as possible if it becomes true
    QAtomicInt abort;

    // Input stream information (all guarded by inputMutex while threads are running)
    QMutex inputMutex;
    qint32 inputFrameNumber;
    qint32 lastFrameNumber;
    QVector<LdDecodeMetaData *> &ldDecodeMetaData;
    QVector<SourceVideo *> &sourceVideos;

    // Output stream information (all guarded by outputMutex while threads are running)
    QMutex outputMutex;

    struct OutputFrame {
        SourceVideo::Data firstTargetFieldData;
        SourceVideo::Data secondTargetFieldData;
        qint32 firstFieldSeqNo;
        qint32 secondFieldSeqNo;
        DropOuts firstTargetFieldDropOuts;
        DropOuts secondTargetFieldDropOuts;
    };

    qint32 outputFrameNumber;
    QMap<qint32, OutputFrame> pendingOutputFrames;
    QFile targetVideo;

    // Local source information
    QVector<bool> sourceDiscTypeCav;
    QVector<qint32> sourceMinimumVbiFrame;
    QVector<qint32> sourceMaximumVbiFrame;

    bool setMinAndMaxVbiFrames();
    qint32 convertSequentialFrameNumberToVbi(qint32 sequentialFrameNumber, qint32 sourceNumber);
    qint32 convertVbiFrameNumberToSequential(qint32 vbiFrameNumber, qint32 sourceNumber);
    QVector<qint32> getAvailableSourcesForFrame(qint32 vbiFrameNumber);
    bool writeOutputField(const SourceVideo::Data &fieldData);
};

#endif // STACKINGPOOL_H
