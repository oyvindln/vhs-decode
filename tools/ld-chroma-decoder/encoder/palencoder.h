/************************************************************************

    palencoder.h

    ld-chroma-encoder - Composite video encoder
    Copyright (C) 2019-2022 Adam Sampson

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

#ifndef PALENCODER_H
#define PALENCODER_H

#include <QFile>
#include <vector>

#include "lddecodemetadata.h"

#include "encoder.h"

class PALEncoder : public Encoder
{
public:
    PALEncoder(QFile &inputFile, QFile &tbcFile, QFile &chromaFile, LdDecodeMetaData &metaData,
               int fieldOffset, bool isComponent, bool scLocked);

private:
    virtual void getFieldMetadata(qint32 fieldNo, LdDecodeMetaData::Field &fieldData);
    virtual void encodeLine(qint32 fieldNo, qint32 frameLine, const quint16 *inputData,
                            std::vector<double> &outputC, std::vector<double> &outputVBS);

    bool scLocked;

    std::vector<double> Y;
    std::vector<double> U;
    std::vector<double> V;
};

#endif
