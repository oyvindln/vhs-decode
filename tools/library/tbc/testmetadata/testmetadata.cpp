/************************************************************************

    testmetadata.cpp

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

#include <QCoreApplication>
#include <QCommandLineParser>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <sstream>

#include "jsonio.h"
#include "lddecodemetadata.h"

// Run unit tests for the JSON parser
void testJsonReader()
{
    std::cerr << "Testing JsonReader\n";
    std::cerr << "Correct syntax\n";

    {
        std::string s;
        bool b;
        int i;
        double d;

        // This includes all the JSON syntax, with whitespace wherever it is permitted.
        // See: https://www.rfc-editor.org/rfc/rfc8259.html
        // XXX Does not include \u escapes or null - JsonReader doesn't support them yet
        std::istringstream input(
            " \t\n\r {"
            " \"emptyArray\" : [ ] ,"
            " \"oneArray\" : [ 42 ] ,"
            " \"threeArray\" : [ 33 , 45 , 78 ] ,"
            " \"emptyObject\" : { } ,"
            " \"oneObject\" : { \"a\": \"b\" } ,"
            " \"emptyString\" : \"\" ,"
            " \"string\" : \"hello world\" ,"
            " \"escapedString\" : \" \\\\ \\/ \\\" \\b \\f \\n \\r \\t \" ,"
            " \"number\" : 12345678 ,"
            " \"floatNumber\" : -123.456e-78 ,"
            " \"trueBool\" : true ,"
            " \"falseBool\" : false } ");

        // Check we can discard them
        JsonReader discardReader(input);
        discardReader.discard();

        // Check we can parse everything correctly
        input.seekg(0);
        JsonReader reader(input);

        reader.beginObject();

        b = reader.readMember(s);
        assert(b && s == "emptyArray");
        reader.beginArray();
        b = reader.readElement();
        assert(!b);
        reader.endArray();

        b = reader.readMember(s);
        assert(b && s == "oneArray");
        reader.beginArray();
        b = reader.readElement();
        assert(b);
        reader.read(i);
        assert(i == 42);
        b = reader.readElement();
        assert(!b);
        reader.endArray();

        b = reader.readMember(s);
        assert(b && s == "threeArray");
        reader.beginArray();
        for (int expected : { 33, 45, 78 }) {
            b = reader.readElement();
            assert(b);
            reader.read(i);
            assert(i == expected);
        }
        b = reader.readElement();
        assert(!b);
        reader.endArray();

        b = reader.readMember(s);
        assert(b && s == "emptyObject");
        reader.beginObject();
        b = reader.readMember(s);
        assert(!b);
        reader.endObject();

        b = reader.readMember(s);
        assert(b && s == "oneObject");
        reader.beginObject();
        b = reader.readMember(s);
        assert(b && s == "a");
        reader.read(s);
        assert(s == "b");
        b = reader.readMember(s);
        assert(!b);
        reader.endObject();

        b = reader.readMember(s);
        assert(b && s == "emptyString");
        reader.read(s);
        assert(s == "");

        b = reader.readMember(s);
        assert(b && s == "string");
        reader.read(s);
        assert(s == "hello world");

        b = reader.readMember(s);
        assert(b && s == "escapedString");
        reader.read(s);
        assert(s == " \\ / \" \b \f \n \r \t ");

        b = reader.readMember(s);
        assert(b && s == "number");
        reader.read(i);
        assert(i == 12345678);

        b = reader.readMember(s);
        assert(b && s == "floatNumber");
        reader.read(d);
        assert(fabs(-123.456e-78 - d) < 1e-80);

        b = reader.readMember(s);
        assert(b && s == "trueBool");
        reader.read(b);
        assert(b);

        b = reader.readMember(s);
        assert(b && s == "falseBool");
        reader.read(b);
        assert(!b);

        b = reader.readMember(s);
        assert(!b);

        reader.endObject();
    }

    // These tests try to exercise all the places that call throwError in the parser
    const char *bad_json[] = {
        "",
        "mystery",
        "terrible",     // i.e. not true
        "fake",         // i.e. not false
        "{,}",
        "{\"a\":42,}",
        "{\"a\":}",
        "{\"a\"}",
        "[,]",
        "[42,]",
        "[\"a\":42]",
        "{33:45}",
        "{",
        "{\"a\":42",
        "[",
        "[42",
        "\"incomplete",
        "\"\\x\"",
        "-x",
        "1.x",
        "1ex",
        "null",         // XXX should be supported
        "\"\\u0042\"",  // XXX should be supported
        "\"\\uABCD\"",  // XXX should be supported
    };
    for (const char *json : bad_json) {
        std::cerr << "Invalid syntax: " << json << "\n";

        // Check that we get an exception when parsing
        bool got_exception = false;
        try {
            std::istringstream input(json);
            JsonReader reader(input);
            reader.discard();
        } catch (JsonReader::Error &e) {
            got_exception = true;
        }
        assert(got_exception);
    }
}

// Run unit tests for VideoSystem
void testVideoSystem() {
    std::cerr << "Testing VideoSystem\n";

    VideoSystem system;
    bool b;

    b = parseVideoSystemName("PAL", system);
    assert(b);
    assert(system == PAL);
    b = parseVideoSystemName("NTSC", system);
    assert(b);
    assert(system == NTSC);
    b = parseVideoSystemName("PAL-M", system);
    assert(b);
    assert(system == PAL_M);
    b = parseVideoSystemName("", system);
    assert(!b);
    b = parseVideoSystemName("SPURIOUS", system);
    assert(!b);
}

int main(int argc, char *argv[])
{
    // Initialise Qt
    QCoreApplication app(argc, argv);
    QCoreApplication::setApplicationName("testmetadata");
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription("testmetadata - unit tests for ld-decode's metadata library");
    parser.addHelpOption();
    parser.addVersionOption();

    // Test options
    QCommandLineOption statsOption(QStringList() << "s" << "stats",
                                   "parse all fields and show statistics");
    parser.addOption(statsOption);
    QCommandLineOption exitOption(QStringList() << "x" << "exit",
                                  "call exit(0) after parsing, to analyse memory usage");
    parser.addOption(exitOption);

    // Positional argument to specify input video file
    parser.addPositionalArgument("input", "Input JSON file (omit to run unit tests)");

    // Positional argument to specify output video file
    parser.addPositionalArgument("output", "Output JSON file (omit to only read input)");

    // Parse the command line
    parser.process(app);

    // Process the positional args
    QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.count() == 0) {
        // Run unit tests
        testJsonReader();
        testVideoSystem();
        return 0;
    }
    if (positionalArguments.count() > 2) {
        qCritical("You may specify one input file and (optionally) one output file");
        return 1;
    }

    // Read the input file
    LdDecodeMetaData metaData;
    if (!metaData.read(positionalArguments.at(0))) {
        qCritical("Unable to read input file");
        return 1;
    }

    // Show statistics
    if (parser.isSet(statsOption)) {
        // These are not very useful, but they ensure that LdDecodeMetaData has
        // actually fully parsed all of the metadata.

        qint32 numFields = metaData.getNumberOfFields();
        qint32 numMetrics = 0;
        double meanWSNR = 0.0;
        qint32 numDropOuts = 0;

        for (qint32 i = 1; i <= numFields; i++) {
            const LdDecodeMetaData::Field &field = metaData.getField(i);

            if (field.vitsMetrics.inUse) {
                ++numMetrics;
                meanWSNR += field.vitsMetrics.wSNR;
            }

            numDropOuts += field.dropOuts.size();
        }

        if (numMetrics > 0) meanWSNR /= numMetrics;

        std::cout << "fields=" << numFields << " metrics=" << numMetrics << " wSNR=" << meanWSNR << " dropouts=" << numDropOuts << "\n";
    }

    // Force an exit if requested
    if (parser.isSet(exitOption)) {
        // We use the C exit function so that C++ destructors don't get called.
        // We can then use a memory debugger like valgrind to analyse the
        // memory that was "leaked", and see how much the parser allocated.
        exit(0);
    }

    // Write the output file, if given
    if (positionalArguments.count() == 2) {
        if (!metaData.write(positionalArguments.at(1))) {
            qCritical("Unable to write output file");
            return 1;
        }
    }

    return 0;
}
