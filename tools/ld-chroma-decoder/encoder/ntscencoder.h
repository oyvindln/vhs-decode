/************************************************************************

    ntscencoder.h

    ld-chroma-encoder - Composite video encoder
    Copyright (C) 2019-2022 Adam Sampson
    Copyright (C) 2022 Phillip Blucas

    This file is part of ld-decode-tools.

    ld-chroma-encoder is free software: you can redistribute it and/or
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

#ifndef NTSCENCODER_H
#define NTSCENCODER_H

#include <QFile>
#include <vector>

#include "encoder.h"

#include "lddecodemetadata.h"

enum ChromaMode {
    WIDEBAND_YUV = 0,   // Y'UV
    WIDEBAND_YIQ,       // Y'IQ
    NARROWBAND_Q        // Y'IQ with Q low-passed
};

class NTSCEncoder : public Encoder
{
public:
    NTSCEncoder(QFile &inputFile, QFile &tbcFile, QFile &chromaFile, LdDecodeMetaData &metaData,
                int fieldOffset, bool isComponent, ChromaMode chromaMode, bool addSetup);

protected:
    virtual void getFieldMetadata(qint32 fieldNo, LdDecodeMetaData::Field &fieldData);
    virtual void encodeLine(qint32 fieldNo, qint32 frameLine, const quint16 *inputData,
                            std::vector<double> &outputC, std::vector<double> &outputVBS);

    const qint32 blankingIre = 0x3C00;
    const qint32 setupIreOffset = 0x0A80; // 10.5 * 256

    ChromaMode chromaMode;
    bool addSetup;

    std::vector<double> Y;
    std::vector<double> C1;
    std::vector<double> C2;
};

#endif
