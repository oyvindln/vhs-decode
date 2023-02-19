/************************************************************************

    audacity.h

    ld-export-metadata - Export JSON metadata into other formats
    Copyright (C) 2023 Adam Sampson

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

#ifndef AUDACITY_H
#define AUDACITY_H

#include <QString>

#include "lddecodemetadata.h"

/*!
    Write an Audacity labels file containing navigation information.

    Format description: <https://manual.audacityteam.org/man/importing_and_exporting_labels.html>

    Returns true on success, false on failure.
*/
bool writeAudacityLabels(LdDecodeMetaData &metaData, const QString &fileName);

#endif
