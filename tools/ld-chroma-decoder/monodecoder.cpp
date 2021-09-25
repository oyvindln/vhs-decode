/************************************************************************

    monodecoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019-2021 Adam Sampson

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

#include "monodecoder.h"

#include "comb.h"
#include "decoderpool.h"
#include "palcolour.h"

bool MonoDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {
    // This decoder works for both PAL and NTSC.

    config.videoParameters = videoParameters;

    return true;
}

QThread *MonoDecoder::makeThread(QAtomicInt& abort, DecoderPool& decoderPool) {
    return new MonoThread(abort, decoderPool, config);
}

MonoThread::MonoThread(QAtomicInt& _abort, DecoderPool& _decoderPool,
                       const MonoDecoder::Configuration &_config, QObject *parent)
    : DecoderThread(_abort, _decoderPool, parent), config(_config)
{
}

void MonoThread::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                              QVector<ComponentFrame> &componentFrames)
{
    for (qint32 fieldIndex = startIndex, frameIndex = 0; fieldIndex < endIndex; fieldIndex += 2, frameIndex++) {
        decodeFrame(inputFields[fieldIndex], inputFields[fieldIndex + 1], componentFrames[frameIndex]);
    }
}

void MonoThread::decodeFrame(const SourceField &firstField, const SourceField &secondField, ComponentFrame &componentFrame)
{
    const LdDecodeMetaData::VideoParameters &videoParameters = config.videoParameters;

    bool ignoreUV = decoderPool.getOutputWriter().getPixelFormat() == OutputWriter::PixelFormat::GRAY16;

    // Initialise and clear the component frame
    // Ignore UV if we're doing Grayscale output.
    // TODO: Fix so we don't need U/V vectors for RGB and YUV output either.
    componentFrame.init(videoParameters, ignoreUV);

    // Interlace the active lines of the two input fields to produce a component frame
    for (qint32 y = videoParameters.firstActiveFrameLine; y < videoParameters.lastActiveFrameLine; y++) {
        const SourceVideo::Data &inputFieldData = (y % 2) == 0 ? firstField.data : secondField.data;
        const quint16 *inputLine = inputFieldData.data() + ((y / 2) * videoParameters.fieldWidth);

        // Copy the whole composite signal to Y (leaving U and V blank)
        double *outY = componentFrame.y(y);
        for (qint32 x = videoParameters.activeVideoStart; x < videoParameters.activeVideoEnd; x++) {
            outY[x] = inputLine[x];
        }
    }
}
