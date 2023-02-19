/************************************************************************

    vitcdecoder.cpp

    ld-decode-tools TBC library
    Copyright (C) 2022 Adam Sampson

    This file is part of ld-decode-tools.

    ld-decode-tools is free software: you can redistribute it and/or
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

#ifndef VITCDECODER_H
#define VITCDECODER_H

#include <QtGlobal>
#include <array>

#include "lddecodemetadata.h"

class VitcDecoder
{
public:
    struct Vitc {
        bool isValid = false;
        qint32 hour = -1;
        qint32 minute = -1;
        qint32 second = -1;
        qint32 frame = -1;
        bool isDropFrame = false;
        bool isColFrame = false;
        bool isFieldMark = false;
        qint32 binaryGroupFlags = 0;
        std::array<qint32, 8> binaryGroups { 0, 0, 0, 0, 0, 0, 0, 0 };
    };

    static Vitc decode(const std::array<qint32, 8>& vitcData, VideoSystem system);

private:
    static void decodeBCD(qint32 tens, qint32 units, qint32& output, bool& isValid);
};

#endif // VITCDECODER_H
