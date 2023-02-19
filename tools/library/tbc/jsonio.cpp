/************************************************************************

    jsonio.cpp

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

#include "jsonio.h"

#include <cmath>
#include <limits>

// Recognise JSON space characters
static bool isAsciiSpace(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

// Recognise JSON digit characters
static bool isAsciiDigit(char c)
{
    return c >= '0' && c <= '9';
}

JsonReader::JsonReader(std::istream &_input)
    : input(_input), position(0), atStart(true)
{
}

void JsonReader::read(int &value)
{
    // Round to the nearest integer
    double d;
    readNumber(d);
    value = static_cast<int>(std::lround(d));
}

void JsonReader::read(double &value)
{
    readNumber(value);
}

void JsonReader::read(bool &value)
{
    const char *rest;

    char c = spaceGet();
    if (c == 't') {
        rest = "rue";
    } else if (c == 'f') {
        rest = "alse";
    } else {
        throwError("expected true or false");
    }

    while (*rest) {
        if (get() != *rest++) throwError("expected true or false");
    }

    value = (c == 't');
}

void JsonReader::read(std::string &value)
{
    readString(value);
}

void JsonReader::read(QString &value)
{
    readString(buf);
    value = QString::fromUtf8(buf.c_str());
}

void JsonReader::beginObject()
{
    if (spaceGet() != '{') throwError("expected {");

    atStarts.push(atStart);
    atStart = true;
}

bool JsonReader::readMember(std::string &member)
{
    char c = spaceGet();
    if (c == '}') {
        // Found the end - don't consume the }
        unget();
        return false;
    }
    if (atStart) {
        // Start of the first member - don't consume it
        unget();
    } else {
        // Consume the ,
        if (c != ',') throwError("expected , or }");
    }

    readString(member);

    if (spaceGet() != ':') throwError("expected :");

    atStart = false;
    return true;
}

void JsonReader::endObject()
{
    if (spaceGet() != '}') throwError("expected }");

    atStart = atStarts.top();
    atStarts.pop();
}

void JsonReader::beginArray()
{
    if (spaceGet() != '[') throwError("expected [");

    atStarts.push(atStart);
    atStart = true;
}

bool JsonReader::readElement()
{
    char c = spaceGet();
    if (c == ']') {
        // Found the end - don't consume the ]
        unget();
        return false;
    }
    if (atStart) {
        // Start of the first element - don't consume it
        unget();
    } else {
        // Consume the ,
        if (c != ',') throwError("expected , or ]");
    }

    atStart = false;
    return true;
}

void JsonReader::endArray()
{
    if (spaceGet() != ']') throwError("expected ]");

    atStart = atStarts.top();
    atStarts.pop();
}

void JsonReader::discard()
{
    // Peek at the first character to see what type it is
    char c = spaceGet();
    unget();

    if (c == '-' || isAsciiDigit(c)) {
        double dummy;
        readNumber(dummy);
    } else if (c == 't' || c == 'f') {
        bool dummy;
        read(dummy);
    } else if (c == '"') {
        readString(buf);
    } else if (c == '{') {
        beginObject();
        while (readMember(buf)) discard();
        endObject();
    } else if (c == '[') {
        beginArray();
        while (readElement()) discard();
        endArray();
    } else {
        // XXX recognise null
        throwError("unrecognised value");
    }
}

// Get the next input character, returning 0 on EOF or error
char JsonReader::get()
{
    char c;
    input.get(c);
    if (!input.good()) return 0;
    ++position;
    return c;
}

// Get the next input character, discarding spaces before it
char JsonReader::spaceGet()
{
    char c;
    do {
        c = get();
    } while (isAsciiSpace(c));
    return c;
}

// Put back an input character to be read again
void JsonReader::unget()
{
    // We assume the input stream supports unget.
    // (If this becomes a problem in the future, we could simulate it.)
    input.unget();
    --position;
}

// Read a JSON string. The result is unescaped and doesn't include the quotes.
void JsonReader::readString(std::string &value)
{
    char c = spaceGet();
    if (c != '"') throwError("expected string");

    value.clear();

    while (true) {
        c = get();
        switch (c) {
        case 0:
            // End of input
            throwError("end of input in string");
        case '"':
            // End of string
            return;
        case '\\':
            c = get();
            switch (c) {
            case '"':
            case '/':
            case '\\':
                value.push_back(c);
                break;
            case 'b':
                value.push_back('\b');
                break;
            case 'f':
                value.push_back('\f');
                break;
            case 'n':
                value.push_back('\n');
                break;
            case 'r':
                value.push_back('\r');
                break;
            case 't':
                value.push_back('\t');
                break;
            case 'u':
                throwError("\\u escapes not supported");
            default:
                throwError("unrecognised \\ escape");
            }
            break;
        default:
            value.push_back(c);
            break;
        }
    }
}

// Read a JSON number
void JsonReader::readNumber(double &value)
{
    // JSON only has "numbers"; it doesn't distinguish between floating point
    // and integers. This means we have to parse as a double, since a value
    // we're expecting to use as an integer might be written as 1.234e3 or
    // similar.

    // Check that the number matches JSON's number syntax, which is more
    // restrictive than the C/C++ parsers accept.

    // XXX could track if we only saw the int part, and parse straight to int

    buf.clear();

    char c = spaceGet();
    if (c == '-') {
        buf.push_back(c);
        c = get();
    }
    if (!isAsciiDigit(c)) throwError("expected - or digit");
    buf.push_back(c);
    c = get();

    while (isAsciiDigit(c)) {
        buf.push_back(c);
        c = get();
    }

    if (c == '.') {
        buf.push_back(c);
        c = get();
        if (!isAsciiDigit(c)) throwError("expected digit after .");
        buf.push_back(c);
        while (true) {
            c = get();
            if (!isAsciiDigit(c)) break;
            buf.push_back(c);
        }
    }

    if (c == 'e') {
        buf.push_back(c);
        c = get();
        if (c == '-' || c == '+') {
            buf.push_back(c);
            c = get();
        }
        if (!isAsciiDigit(c)) throwError("expected digit after e");
        buf.push_back(c);
        while (true) {
            c = get();
            if (!isAsciiDigit(c)) break;
            buf.push_back(c);
        }
    }

    // XXX In C++17 we could use std::from_chars instead, which is slightly more efficient
    value = std::stod(buf);

    // We've read one character beyond the end of the number (there's no way to
    // tell where the end is otherwise), so we must unget the last char
    unget();
}

JsonWriter::JsonWriter(std::ostream &_output)
    : output(_output), atStart(true)
{
    // Use maximum double precision for floating-point output
    output.precision(std::numeric_limits<double>::digits10 + 1);
}

void JsonWriter::write(int value)
{
    output << value;
}

void JsonWriter::write(double value)
{
    output << value;
}

void JsonWriter::write(bool value)
{
    output << (value ? "true" : "false");
}

void JsonWriter::write(const char *value)
{
    writeString(value);
}

void JsonWriter::write(const QString &value)
{
    writeString(value.toUtf8());
}

void JsonWriter::beginObject()
{
    output.put('{');

    atStarts.push(atStart);
    atStart = true;
}

void JsonWriter::writeMember(const char *member)
{
    if (!atStart) output.put(',');

    writeString(member);
    output << ":";

    atStart = false;
}

void JsonWriter::endObject()
{
    output.put('}');

    atStart = atStarts.top();
    atStarts.pop();
}

void JsonWriter::beginArray()
{
    output.put('[');

    atStarts.push(atStart);
    atStart = true;
}

void JsonWriter::writeElement()
{
    if (!atStart) output.put(',');

    atStart = false;
}

void JsonWriter::endArray()
{
    output.put(']');

    atStart = atStarts.top();
    atStarts.pop();
}

void JsonWriter::writeString(const char *str)
{
    output.put('"');

    while (char c = *str++) {
        switch (c) {
        case '"':
        case '\\':
            output.put('\\');
            output.put(c);
            break;
        case '\b':
            output << "\\b";
            break;
        case '\f':
            output << "\\f";
            break;
        case '\n':
            output << "\\n";
            break;
        case '\r':
            output << "\\r";
            break;
        case '\t':
            output << "\\t";
            break;
        default:
            output.put(c);
            break;
        }
    }

    output.put('"');
}
