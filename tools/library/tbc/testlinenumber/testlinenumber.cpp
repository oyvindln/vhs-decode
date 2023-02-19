/************************************************************************

    testlinenumber.cpp

    Unit tests for metadata classes
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

#include <cassert>
#include <cstdio>

#include "lddecodemetadata.h"
#include "linenumber.h"

void testAllValues(VideoSystem system, qint32 lines) {
    printf("-- %d line --\n", lines);
    printf("%8s %8s %8s %8s %8s %8s\n",
           "std", "frame0", "frame1", "field0", "field1", "isFirst");

    // Check up to the padding line at the end
    for (qint32 i = 0; i <= lines; i++) {
        LineNumber num = LineNumber::fromFrame0(i, system);

        if (i < 6 || i > lines - 6) {
            printf("%8d %8d %8d %8d %8d %8s\n",
                   num.standard(), num.frame0(), num.frame1(),
                   num.field0(), num.field1(),
                   num.isFirstField() ? "true" : "false");
        } else if (i == 6) {
            printf("...\n");
        }

        // Check that we can round-trip all the formats for this line number.
        assert(LineNumber::fromStandard(num.standard(), system).frame0() == i);
        assert(LineNumber::fromFrame0(num.frame0(), system).frame0() == i);
        assert(LineNumber::fromFrame1(num.frame1(), system).frame0() == i);
        assert(LineNumber::fromField0(num.field0(), num.isFirstField(), system).frame0() == i);
        assert(LineNumber::fromField1(num.field1(), num.isFirstField(), system).frame0() == i);
    }
}

int main()
{
    testAllValues(PAL, 625);
    testAllValues(NTSC, 525);

    // PAL bounds
    assert(LineNumber::fromFrame0(0, PAL).standard() == 1);
    assert(LineNumber::fromFrame0(0, PAL).isFirstField());
    assert(LineNumber::fromFrame0(1, PAL).standard() == 314);
    assert(!LineNumber::fromFrame0(1, PAL).isFirstField());
    assert(LineNumber::fromFrame0(623, PAL).standard() == 625);
    assert(LineNumber::fromFrame0(624, PAL).standard() == 313);

    // NTSC bounds
    assert(LineNumber::fromFrame0(0, NTSC).standard() == 1);
    assert(LineNumber::fromFrame0(0, NTSC).isFirstField());
    assert(LineNumber::fromFrame0(1, NTSC).standard() == 264);
    assert(!LineNumber::fromFrame0(1, NTSC).isFirstField());
    assert(LineNumber::fromFrame0(523, NTSC).standard() == 525);
    assert(LineNumber::fromFrame0(524, NTSC).standard() == 263);

    // ld-decode treats the "middle" line as being part of the first field
    assert(LineNumber::fromStandard(313, PAL).isFirstField());
    assert(LineNumber::fromStandard(263, NTSC).isFirstField());

    // Check other systems have the right number of lines
    assert(LineNumber::fromFrame0(524, PAL_M).standard() == 263);

    return 0;
}

