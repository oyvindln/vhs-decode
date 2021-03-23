/************************************************************************

    paldecoder.h

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

#ifndef PALDECODER_H
#define PALDECODER_H

#include <QObject>
#include <QAtomicInt>
#include <QThread>
#include <QDebug>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

#include "decoder.h"
#include "palcolour.h"
#include "sourcefield.h"

class DecoderPool;

// 2D PAL decoder using PALcolour
class PalDecoder : public Decoder {
public:
    PalDecoder(const PalColour::Configuration &palConfig);
    bool configure(const LdDecodeMetaData::VideoParameters &videoParameters) override;
    const char *getPixelName() const override;
    bool isOutputY4m() override;
    QString getHeaders() const override;
    qint32 getLookBehind() const override;
    qint32 getLookAhead() const override;
    QThread *makeThread(QAtomicInt& abort, DecoderPool& decoderPool) override;

    // Parameters used by PalDecoder and PalThread
    struct Configuration : public Decoder::Configuration {
        PalColour::Configuration pal;
    };

private:
    Configuration config;
};

class PalThread : public DecoderThread
{
    Q_OBJECT
public:
    explicit PalThread(QAtomicInt &abort, DecoderPool &decoderPool,
                       const PalDecoder::Configuration &config,
                       QObject *parent = nullptr);

protected:
    void decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                      QVector<OutputFrame> &outputFrames) override;

private:
    // Settings
    const PalDecoder::Configuration &config;

    // PAL colour object
    PalColour palColour;
};

#endif // PALDECODER
