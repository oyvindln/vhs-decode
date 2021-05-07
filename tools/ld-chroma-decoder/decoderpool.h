/************************************************************************

    decoderpool.h

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
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

#ifndef DECODERPOOL_H
#define DECODERPOOL_H

#include <QObject>
#include <QAtomicInt>
#include <QElapsedTimer>
#include <QMap>
#include <QMutex>
#include <QThread>
#include <QVector>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

#include "decoder.h"
#include "outputwriter.h"
#include "sourcefield.h"

class DecoderPool
{
public:
    explicit DecoderPool(Decoder &decoder, QString inputFileName,
                         LdDecodeMetaData &ldDecodeMetaData,
                         OutputWriter::Configuration &outputConfig, QString outputFileName,
                         qint32 startFrame, qint32 length, qint32 maxThreads);

    // Decode fields to frames as specified by the constructor args.
    // Returns true on success; on failure, prints a message and returns false.
    bool process();

    // For worker threads: get the configured OutputWriter
    OutputWriter &getOutputWriter() {
        return outputWriter;
    }

    // For worker threads: get the next batch of data from the input file.
    //
    // fields will be resized and filled with pairs of SourceFields; entries
    // from startIndex to endIndex are those that should be processed into
    // output frames, with startIndex corresponding to the first field of frame
    // startFrameNumber.
    //
    // If the Decoder requested lookahead or lookbehind, an appropriate number
    // of additional fields will be provided before startIndex and after
    // endIndex. Dummy black frames (with metadata copied from a real frame)
    // will be provided when going beyond the bounds of the input file.
    //
    // Returns true if a frame was returned, false if the end of the input has
    // been reached.
    bool getInputFrames(qint32 &startFrameNumber, QVector<SourceField> &fields, qint32 &startIndex, qint32 &endIndex);

    // For worker threads: return decoded frames to write to the output file.
    //
    // outputFrames should contain RGB48, YUV444P16, or GRAY16 output frames,
    // with the first frame being startFrameNumber.
    //
    // Returns true on success, false on failure.
    bool putOutputFrames(qint32 startFrameNumber, const QVector<OutputFrame> &outputFrames);

private:
    bool putOutputFrame(qint32 frameNumber, const OutputFrame &outputFrame);

    // Default batch size, in frames
    static constexpr qint32 DEFAULT_BATCH_SIZE = 16;

    // Parameters
    Decoder &decoder;
    QString inputFileName;
    OutputWriter::Configuration outputConfig;
    QString outputFileName;
    qint32 startFrame;
    qint32 length;
    qint32 maxThreads;

    // Atomic abort flag shared by worker threads; workers watch this, and shut
    // down as soon as possible if it becomes true
    QAtomicInt abort;

    // Input stream information (all guarded by inputMutex while threads are running)
    QMutex inputMutex;
    qint32 decoderLookBehind;
    qint32 decoderLookAhead;
    qint32 inputFrameNumber;
    qint32 lastFrameNumber;
    LdDecodeMetaData &ldDecodeMetaData;
    SourceVideo sourceVideo;

    // Output stream information (all guarded by outputMutex while threads are running)
    QMutex outputMutex;
    qint32 outputFrameNumber;
    QMap<qint32, OutputFrame> pendingOutputFrames;
    OutputWriter outputWriter;
    QFile targetVideo;
    QElapsedTimer totalTimer;
};

#endif // DECODERPOOL_H
