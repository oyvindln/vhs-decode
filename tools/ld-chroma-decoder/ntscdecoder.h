/************************************************************************

    ntscdecoder.h

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

#ifndef NTSCDECODER_H
#define NTSCDECODER_H

#include <QObject>
#include <QAtomicInt>
#include <QThread>
#include <QDebug>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

#include "comb.h"
#include "decoder.h"
#include "sourcefield.h"

class DecoderPool;

// 2D/3D NTSC decoder using Comb
class NtscDecoder : public Decoder {
public:
    NtscDecoder(const Comb::Configuration &combConfig);
    bool configure(const LdDecodeMetaData::VideoParameters &videoParameters) override;
    const char *getPixelName() const override;
    bool isOutputY4m() override;
    QString getHeaders() const override;
    qint32 getLookBehind() const override;
    qint32 getLookAhead() const override;
    QThread *makeThread(QAtomicInt& abort, DecoderPool& decoderPool) override;

    // Parameters used by NtscDecoder and NtscThread
    struct Configuration : public Decoder::Configuration {
        Comb::Configuration combConfig;
    };

private:
    Configuration config;
};

class NtscThread : public DecoderThread
{
    Q_OBJECT
public:
    explicit NtscThread(QAtomicInt &abort, DecoderPool &decoderPool,
                        const NtscDecoder::Configuration &config,
                        QObject *parent = nullptr);

protected:
    void decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                      QVector<OutputFrame> &outputFrames) override;

private:
    // Settings
    const NtscDecoder::Configuration &config;

    // NTSC decoder
    Comb comb;
};

#endif // NTSCDECODER_H
