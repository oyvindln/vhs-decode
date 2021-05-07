/************************************************************************

    paldecoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns
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

#include "paldecoder.h"

#include "decoderpool.h"

PalDecoder::PalDecoder(const PalColour::Configuration &palConfig)
{
    config.pal = palConfig;
}

bool PalDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {
    // Ensure the source video is PAL
    if (!videoParameters.isSourcePal) {
        qCritical() << "This decoder is for PAL video sources only";
        return false;
    }

    config.videoParameters = videoParameters;

    return true;
}

qint32 PalDecoder::getLookBehind() const
{
    return config.pal.getLookBehind();
}

qint32 PalDecoder::getLookAhead() const
{
    return config.pal.getLookAhead();
}

QThread *PalDecoder::makeThread(QAtomicInt& abort, DecoderPool& decoderPool) {
    return new PalThread(abort, decoderPool, config);
}

PalThread::PalThread(QAtomicInt& _abort, DecoderPool& _decoderPool,
                     const PalDecoder::Configuration &_config, QObject *parent)
    : DecoderThread(_abort, _decoderPool, parent), config(_config)
{
    // Configure PALcolour
    palColour.updateConfiguration(config.videoParameters, config.pal);
}

void PalThread::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                             QVector<ComponentFrame> &componentFrames)
{
    palColour.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
}
