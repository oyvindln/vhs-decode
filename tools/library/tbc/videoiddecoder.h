/************************************************************************

    videoiddecoder.cpp

    ld-decode-tools TBC library
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2023 Phillip Blucas

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

#ifndef VIDEOIDDECODER_H
#define VIDEOIDDECODER_H

#include <QDebug>

class VideoIdDecoder
{
public:
    // VIDEO ID Aspect Ratio
    enum VIdAspectRatio {
        fourByThree,      // 0
        sixteenByNine,    // 1
        letterBox,        // 2
        undefined,        // 3
    };

    // VIDEO ID CGMS-A
    enum VIdCgms {
        copyFreely,    // 0
        notUsed,       // 1
        copyOnce,      // 2
        copyNever,     // 3
    };

    // VIDEO ID APS trigger bits
    enum VIdAps {
        pspOff,        // 0
        pspOn,         // 1
        pspOn2Line,    // 2
        pspOn4Line,    // 3
    };

    // Overall container struct for VIDEO ID information, with default values
    struct VideoId {
        qint32 videoIdData = -1;
        VIdAspectRatio vIdAspectRatio = fourByThree;
        VIdCgms vIdCgms = copyFreely;
        VIdAps vIdAps = pspOff;
        bool analoguePreRecorded = false;
    };

    VideoId decodeFrame(qint32 videoData_1, qint32 videoData_2);
    VideoId decode(qint32 videoData);

};

#endif // VIDEOIDDECODER_H
