/************************************************************************

    biphasecode.h

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

#ifndef BIPHASECODE_H
#define BIPHASECODE_H

#include "lddecodemetadata.h"
#include "sourcevideo.h"

#include <QVector>

// Decoder for PAL/NTSC LaserDisc biphase code lines.
// Specified in IEC 60586-1986 section 10.1 (PAL) and IEC 60587-1986 section 10.1 (NTSC).
class BiphaseCode {
public:
    bool decodeLines(const SourceVideo::Data& line16Data, const SourceVideo::Data& line17Data,
                     const SourceVideo::Data& line18Data,
                     const LdDecodeMetaData::VideoParameters& videoParameters,
                     LdDecodeMetaData::Field& fieldMetadata);
    bool decodeLine(qint32 lineIndex, const SourceVideo::Data& lineData,
                    const LdDecodeMetaData::VideoParameters& videoParameters,
                    LdDecodeMetaData::Field& fieldMetadata);

private:
    qint32 manchesterDecoder(const SourceVideo::Data& lineData, qint32 zcPoint,
                             LdDecodeMetaData::VideoParameters videoParameters);
};

#endif // BIPHASECODE_H
