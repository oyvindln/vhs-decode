/************************************************************************

    ntscdecoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018 Chad Page
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2019-2020 Adam Sampson
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

#include "ntscdecoder.h"

#include "decoderpool.h"

NtscDecoder::NtscDecoder(const Comb::Configuration &combConfig)
{
    config.combConfig = combConfig;
    config.outputYCbCr = combConfig.outputYCbCr;
    config.outputY4m = combConfig.outputY4m;
    config.pixelFormat = combConfig.pixelFormat;
}

bool NtscDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {
    // Ensure the source video is NTSC
    if (videoParameters.isSourcePal) {
        qCritical() << "This decoder is for NTSC video sources only";
        return false;
    }

    // Compute cropping parameters
    setVideoParameters(config, videoParameters);

    return true;
}

const char *NtscDecoder::getPixelName() const
{
    return config.outputYCbCr ?
           config.combConfig.chromaGain > 0 ? "YUV444P16" : "GRAY16" : "RGB48";
}

bool NtscDecoder::isOutputY4m()
{
    return config.outputY4m;
}

QString NtscDecoder::getHeaders() const
{
    QString y4mHeader;
    qint32 rateN = 30000;
    qint32 rateD = 1001;
    qint32 width = config.videoParameters.activeVideoEnd - config.videoParameters.activeVideoStart;
    qint32 height = config.topPadLines + config.bottomPadLines +
                    config.videoParameters.lastActiveFrameLine - config.videoParameters.firstActiveFrameLine;
    QTextStream(&y4mHeader) << "YUV4MPEG2 W" << width << " H" << height << " F" << rateN << ":" << rateD
                            << " I" << y4mFieldOrder << " A" << (config.videoParameters.isWidescreen ? Y4M_PAR_NTSC_169 : Y4M_PAR_NTSC_43)
                            << (config.pixelFormat == YUV444P16 ? Y4M_CS_YUV444P16 : Y4M_CS_GRAY16);
    return y4mHeader;
}

qint32 NtscDecoder::getLookBehind() const
{
    return config.combConfig.getLookBehind();
}

qint32 NtscDecoder::getLookAhead() const
{
    return config.combConfig.getLookAhead();
}

QThread *NtscDecoder::makeThread(QAtomicInt& abort, DecoderPool& decoderPool)
{
    return new NtscThread(abort, decoderPool, config);
}

NtscThread::NtscThread(QAtomicInt& _abort, DecoderPool &_decoderPool,
                       const NtscDecoder::Configuration &_config, QObject *parent)
    : DecoderThread(_abort, _decoderPool, parent), config(_config)
{
    // Configure NTSC decoder
    comb.updateConfiguration(config.videoParameters, config.combConfig);
}

void NtscThread::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                              QVector<OutputFrame> &outputFrames)
{
    QVector<OutputFrame> decodedFrames(outputFrames.size());

    // Decode fields to frames
    comb.decodeFrames(inputFields, startIndex, endIndex, decodedFrames);

    for (qint32 i = 0; i < outputFrames.size(); i++) {
        // Crop the frame to just the active area
        outputFrames[i] = NtscDecoder::cropOutputFrame(config, decodedFrames[i]);
    }
}
