/************************************************************************

    paldecoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2019 Adam Sampson
    Copyright (C) 2021 Phillip Blucas

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
    config.outputYCbCr = palConfig.outputYCbCr;
    config.outputY4m = palConfig.outputY4m;
    config.pixelFormat = palConfig.pixelFormat;
}

bool PalDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {
    // Ensure the source video is PAL
    if (!videoParameters.isSourcePal) {
        qCritical() << "This decoder is for PAL video sources only";
        return false;
    }

    // Compute cropping parameters
    setVideoParameters(config, videoParameters);

    return true;
}

const char *PalDecoder::getPixelName() const
{
    return config.outputYCbCr ?
           config.pal.chromaGain > 0 ? "YUV444P16" : "GRAY16" : "RGB48";
}

bool PalDecoder::isOutputY4m()
{
    return config.outputY4m;
}

QString PalDecoder::getHeaders() const
{
    QString y4mHeader;
    qint32 rateN = 25;
    qint32 rateD = 1;
    qint32 width = config.videoParameters.activeVideoEnd - config.videoParameters.activeVideoStart;
    qint32 height = config.topPadLines + config.bottomPadLines +
                    config.videoParameters.lastActiveFrameLine - config.videoParameters.firstActiveFrameLine;
    QTextStream(&y4mHeader) << "YUV4MPEG2 W" << width << " H" << height << " F" << rateN << ":" << rateD
                            << " I" << y4mFieldOrder << " A" << (config.videoParameters.isWidescreen ? Y4M_PAR_PAL_169 : Y4M_PAR_PAL_43)
                            << (config.pixelFormat == YUV444P16 ? Y4M_CS_YUV444P16 : Y4M_CS_GRAY16);
    return y4mHeader;
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
                             QVector<OutputFrame> &outputFrames)
{
    QVector<OutputFrame> decodedFrames(outputFrames.size());

    // Perform the PALcolour filtering
    palColour.decodeFrames(inputFields, startIndex, endIndex, decodedFrames);

    for (qint32 i = 0; i < outputFrames.size(); i++) {
        // Crop the frame to just the active area
        outputFrames[i] = PalDecoder::cropOutputFrame(config, decodedFrames[i]);
    }
}
