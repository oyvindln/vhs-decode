/************************************************************************

    componentframe.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2021 Adam Sampson

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

#include "componentframe.h"

ComponentFrame::ComponentFrame()
    : width(-1), height(-1)
{
}

void ComponentFrame::init(const LdDecodeMetaData::VideoParameters &videoParameters)
{
    width = videoParameters.fieldWidth;
    height = (videoParameters.fieldHeight * 2) - 1;

    const qint32 size = width * height;

    yData.resize(size);
    yData.fill(0.0);

    uData.resize(size);
    uData.fill(0.0);

    vData.resize(size);
    vData.fill(0.0);
}
