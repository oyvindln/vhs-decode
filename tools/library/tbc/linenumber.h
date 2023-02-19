/************************************************************************

    linenumber.h

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

#ifndef LINENUMBER_H
#define LINENUMBER_H

#include <QtGlobal>
#include <cassert>

#include "lddecodemetadata.h"

/*
The lines in a ComponentField or OutputField are numbered as follows:

For 525-line standards: [Poynton p500 table 41.1]

frame0  field0  first?  standard
0       0       1       1           first of 5 lines of equalisation pulses
1       0       0       264
2       1       1       2
3       1       0       265
...
523     261     0       525
524     262     1       263         last half-line of active area + half-line of equalisation pulses

For 625-line standards: [Poynton p520 table 43.1]

frame0  field0  first?  standard
0       0       1       1           first of 4 lines of broad pulses
1       0       0       314
2       1       1       2
3       1       0       315
...
623     311     0       625         last of 4 lines of equalisation pulses
624     312     1       313         half-line of equalisation pulses + half-line of broad pulses

All fields in a TBC file have the same size, so the second field has an extra
line of padding at the end -- this is not included in the output.

In 625-line standards, line 313 is treated by ld-decode as being part of the
first field, so the first field has 313 lines and the second has 312 plus a
padding line. (Poynton says line 313 is part of the second field; EBU Tech 3280
says the field boundary occurs in the middle of line 313.)
*/

// A line number within a video frame in a particular VideoStandard
class LineNumber {
public:
    // Default constructor: initialise to an invalid line number
    LineNumber()
        : frame0Line(-1), firstFieldLines(0)
    {}

    // Return the line number in standard terminology:
    // 1-based, in the order in which lines are transmitted
    qint32 standard() const {
        return (frame0Line / 2) + 1 + (firstFieldLines * (frame0Line % 2));
    }

    // Return true if this line is in the first field in standard terminology:
    // the field containing frame0() == 0 and standard() == 1
    bool isFirstField() const {
        return (frame0Line % 2) == 0;
    }

    // Return 0-based line number within the frame
    qint32 frame0() const {
        return frame0Line;
    }

    // Return 1-based line number within the frame
    qint32 frame1() const {
        return frame0Line + 1;
    }

    // Return 0-based line number within the field
    qint32 field0() const {
        return frame0Line / 2;
    }

    // Return 1-based line number within the field
    qint32 field1() const {
        return field0() + 1;
    }

    // Construct from a standard line number
    static LineNumber fromStandard(qint32 standardLine, VideoSystem system) {
        return LineNumber(standardLine, system, true);
    }

    // Construct from a 0-based line number within the frame
    static LineNumber fromFrame0(qint32 frame0Line, VideoSystem system) {
        return LineNumber(frame0Line, system);
    }

    // Construct from a 1-based line number within the frame
    static LineNumber fromFrame1(qint32 frame1Line, VideoSystem system) {
        return LineNumber(frame1Line - 1, system);
    }

    // Construct from a 0-based line number within the field
    static LineNumber fromField0(qint32 field0Line, bool isFirstField, VideoSystem system) {
        return LineNumber((field0Line * 2) + (isFirstField ? 0 : 1), system);
    }

    // Construct from a 1-based line number within the field
    static LineNumber fromField1(qint32 field1Line, bool isFirstField, VideoSystem system) {
        return fromField0(field1Line - 1, isFirstField, system);
    }

protected:
    // Private constructor used by from* static methods
    LineNumber(qint32 number, VideoSystem system, bool isStandard = false)
    {
        const qint32 numLines = (system == PAL) ? 625 : 525;
        firstFieldLines = (numLines / 2) + 1;

        // In both cases, we allow an extra two lines at the end, so we can
        // represent the padding line in the second field, plus a
        // one-past-the-end value for ranges.

        if (isStandard) {
            // number is in standard form (so we have to know firstFieldLines to convert it)

            assert(number >= 1);
            assert(number <= numLines + 2);

            frame0Line = (((number - 1) % firstFieldLines) * 2) + (number / (firstFieldLines + 1));
        } else {
            // number is in frame0 form

            assert(number >= 0);
            assert(number < numLines + 2);

            frame0Line = number;
        }
    }

private:
    qint32 frame0Line;
    qint32 firstFieldLines;
};

#endif
