/************************************************************************

    testvitcdecoder.cpp

    Unit tests for VitcDecoder
    Copyright (C) 2022 Adam Sampson

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

#include <cassert>
#include <iostream>

using std::cerr;

#include "vitcdecoder.h"

// Set the time fields in a Vitc struct
void setTime(VitcDecoder::Vitc& vitc, qint32 hour, qint32 minute, qint32 second, qint32 frame)
{
    vitc.hour = hour;
    vitc.minute = minute;
    vitc.second = second;
    vitc.frame = frame;
}

// Check that two Vitc structs are field-by-field identical
void assertSame(const VitcDecoder::Vitc& actual, const VitcDecoder::Vitc& expected)
{
    // Assume we're looking for valid result -- if not we'll test isValid explicitly
    assert(actual.isValid);

    assert(actual.hour == expected.hour);
    assert(actual.minute == expected.minute);
    assert(actual.second == expected.second);
    assert(actual.frame == expected.frame);
    assert(actual.isDropFrame == expected.isDropFrame);
    assert(actual.isColFrame == expected.isColFrame);
    assert(actual.binaryGroupFlags == expected.binaryGroupFlags);
    assert(actual.binaryGroups == expected.binaryGroups);
}

// Test VitcDecoder::decode
void testDecode()
{
    using Vitc = VitcDecoder::Vitc;

    VitcDecoder decoder;

    // We check that the decoder gives the expected output for various inputs.
    // Since some bit assignments differ for 25-frame and 30-frame systems, we
    // must check both standards in some cases.

    cerr << "Testing VitcDecoder::decode\n";
    cerr << "ITU-R BR.780-2 - 6.16.1 - Valid times\n";

    {
        Vitc expected;

        setTime(expected, 0, 0, 0, 0);
        assertSame(decoder.decode({ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, PAL), expected);
        assertSame(decoder.decode({ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, NTSC), expected);
        assertSame(decoder.decode({ 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00 }, PAL_M), expected);

        setTime(expected, 23, 59, 59, 24);
        assertSame(decoder.decode({ 0x04, 0x02, 0x09, 0x05, 0x09, 0x05, 0x03, 0x02 }, PAL), expected);

        setTime(expected, 23, 59, 59, 29);
        assertSame(decoder.decode({ 0x09, 0x02, 0x09, 0x05, 0x09, 0x05, 0x03, 0x02 }, NTSC), expected);
    }

    cerr << "ITU-R BR.780-2 - 5.2 - Invalid BCD digits\n";

    {
        // Only the unit positions have enough bits to go beyond 9
        assert(!decoder.decode({ 0x0A, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL).isValid);
        assert(!decoder.decode({ 0x01, 0x01, 0x0A, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL).isValid);
        assert(!decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x0A, 0x01, 0x01, 0x01 }, PAL).isValid);
        assert(!decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x0A, 0x01 }, PAL).isValid);
    }

    cerr << "ITU-R BR.780-2 - 1.1/1.2/2.1 - Invalid time fields\n";

    {
        assert(!decoder.decode({ 0x05, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL).isValid);
        assert(decoder.decode({ 0x05, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, NTSC).isValid);
        assert(decoder.decode({ 0x05, 0x02, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL_M).isValid);
        assert(!decoder.decode({ 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, NTSC).isValid);
        assert(!decoder.decode({ 0x00, 0x03, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL_M).isValid);
        assert(!decoder.decode({ 0x01, 0x01, 0x00, 0x06, 0x01, 0x01, 0x01, 0x01 }, PAL).isValid);
        assert(!decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x00, 0x06, 0x01, 0x01 }, PAL).isValid);
        assert(!decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x04, 0x02 }, PAL).isValid);
    }

    cerr << "ITU-R BR.780-2 - 6.16.2 - Drop frame\n";

    {
        Vitc expected;
        setTime(expected, 11, 11, 11, 11);
        expected.isDropFrame = true;

        assertSame(decoder.decode({ 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, NTSC), expected);
        assertSame(decoder.decode({ 0x01, 0x05, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL_M), expected);
        // Doesn't exist for PAL
    }

    cerr << "ITU-R BR.780-2 - 6.16.2 - Colour frame\n";

    {
        Vitc expected;
        setTime(expected, 11, 11, 11, 11);
        expected.isColFrame = true;

        assertSame(decoder.decode({ 0x01, 0x09, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, PAL), expected);
        assertSame(decoder.decode({ 0x01, 0x09, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01 }, NTSC), expected);
    }

    cerr << "ITU-R BR.780-2 - 6.16.4 - Field mark\n";

    {
        Vitc expected;
        setTime(expected, 11, 11, 11, 11);
        expected.isFieldMark = true;

        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x09 }, PAL), expected);
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x09, 0x01, 0x01, 0x01, 0x01 }, NTSC), expected);
    }

    cerr << "ITU-R BR.780-2 - 6.16.2 - Binary group flags\n";

    {
        Vitc expected;
        setTime(expected, 11, 11, 11, 11);

        // BGF0
        expected.binaryGroupFlags = 1;
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x09, 0x01, 0x01, 0x01, 0x01 }, PAL), expected);
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x09, 0x01, 0x01 }, NTSC), expected);
        // BGF1
        expected.binaryGroupFlags = 2;
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x05 }, PAL), expected);
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x05 }, NTSC), expected);
        // BGF2
        expected.binaryGroupFlags = 4;
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x09, 0x01, 0x01 }, PAL), expected);
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x01, 0x09 }, NTSC), expected);
        // All of them
        expected.binaryGroupFlags = 7;
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x09, 0x01, 0x09, 0x01, 0x05 }, PAL), expected);
        assertSame(decoder.decode({ 0x01, 0x01, 0x01, 0x01, 0x01, 0x09, 0x01, 0x0D }, NTSC), expected);
    }

    cerr << "ITU-R BR.780-2 - 6.16.3 - Binary groups\n";

    {
        Vitc expected;
        setTime(expected, 11, 11, 11, 11);

        for (int i = 0; i < 8; i++) expected.binaryGroups[i] = i + 2;
        assertSame(decoder.decode({ 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81, 0x91 }, PAL), expected);
        assertSame(decoder.decode({ 0x21, 0x31, 0x41, 0x51, 0x61, 0x71, 0x81, 0x91 }, NTSC), expected);

        for (int i = 0; i < 8; i++) expected.binaryGroups[i] = 0xF;
        assertSame(decoder.decode({ 0xF1, 0xF1, 0xF1, 0xF1, 0xF1, 0xF1, 0xF1, 0xF1 }, PAL), expected);
        assertSame(decoder.decode({ 0xF1, 0xF1, 0xF1, 0xF1, 0xF1, 0xF1, 0xF1, 0xF1 }, NTSC), expected);
    }
}

int main()
{
    testDecode();

    return 0;
}
