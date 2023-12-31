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

#include "videoiddecoder.h"

// Method to decode 2 fields (a frame) of VIDEO ID and combine them into a single VIDEO ID response
VideoIdDecoder::VideoId VideoIdDecoder::decodeFrame(qint32 videoData_1, qint32 videoData_2)
{
    // Data from both fields should match (line 20 and line 283)
    if (videoData_1 != videoData_2) {
        VideoId videoid;
        return videoid;
    }

    return decode(videoData_1);
}

// Method to decode VIDEO ID for a field
VideoIdDecoder::VideoId VideoIdDecoder::decode(qint32 videoData)
{
    VideoId videoid;

    if (videoData == -1) return videoid;

    // IEC 61880-1998 - VIDEO ID --------------------------------------------------------------------------------------

    // 14-bit raw data
    videoid.videoIdData = videoData;

    // 4.1 Aspect ratio and display format
    videoid.vIdAspectRatio = static_cast<VideoIdDecoder::VIdAspectRatio>(videoData >> 12);

    // B.2 CGMS-A information identifier
    videoid.vIdCgms = static_cast<VideoIdDecoder::VIdCgms>((videoData >> 6) & 3);

    // B.3 APS trigger bits
    videoid.vIdAps = static_cast<VideoIdDecoder::VIdAps>((videoData >> 4) & 3);

    // B.4 Analogue source bit
    videoid.analoguePreRecorded = (videoData >> 3) & 1;

    return videoid;
}
