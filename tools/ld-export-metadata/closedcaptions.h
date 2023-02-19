/************************************************************************

    closedcaptions.cpp

    ld-export-metadata - Export JSON metadata into other formats
    Copyright (C) 2019-2020 Adam Sampson
    Copyright (C) 2021 Simon Inns

    This file is part of ld-decode-tools.

    ld-export-metadata is free software: you can redistribute it and/or
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

#ifndef CLOSEDCAPTIONS_H
#define CLOSEDCAPTIONS_H

#include <QString>

#include "lddecodemetadata.h"

QString generateTimeStamp(qint32 fieldIndex);
qint32 sanityCheckData(qint32 dataByte);
bool writeClosedCaptions(LdDecodeMetaData &metaData, const QString &fileName);

#endif // CLOSEDCAPTIONS_H
