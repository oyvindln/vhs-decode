/************************************************************************

    vitccode.h

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2022 Adam Sampson

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

#ifndef VITCCODE_H
#define VITCCODE_H

#include "sourcevideo.h"
#include "lddecodemetadata.h"

#include <vector>

class VitcCode
{
public:
    bool decodeLine(const SourceVideo::Data &lineData,
                    const LdDecodeMetaData::VideoParameters& videoParameters,
                    LdDecodeMetaData::Field& fieldMetadata);

    std::vector<qint32> getLineNumbers(const LdDecodeMetaData::VideoParameters& videoParameters);
};

#endif // VITCCODE_H
