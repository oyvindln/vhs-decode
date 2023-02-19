/************************************************************************

    vbilinedecoder.h

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-vbi is free software: you can redistribute it and/or
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

#ifndef VBILINEDECODER_H
#define VBILINEDECODER_H

#include <QObject>
#include <QAtomicInt>
#include <QThread>
#include <QDebug>

#include "lddecodemetadata.h"
#include "sourcevideo.h"

class DecoderPool;

class VbiLineDecoder : public QThread {
    Q_OBJECT

public:
    explicit VbiLineDecoder(QAtomicInt& _abort, DecoderPool& _decoderPool, QObject *parent = nullptr);

    // The range of field lines needed from the input file (1-based, inclusive)
    static constexpr qint32 startFieldLine = 6;
    static constexpr qint32 endFieldLine = 22;

protected:
    void run() override;

private:
    // Decoder pool
    QAtomicInt& abort;
    DecoderPool& decoderPool;

    SourceVideo::Data getFieldLine(const SourceVideo::Data& sourceField, qint32 fieldLine,
                                   const LdDecodeMetaData::VideoParameters& videoParameters);
};

#endif // VBILINEDECODER_H
