/************************************************************************

    dropouts.h

    ld-decode-tools TBC library
    Copyright (C) 2018-2020 Simon Inns

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

#include "dropouts.h"

#include "jsonio.h"

#include <cassert>

DropOuts::DropOuts(const QVector<qint32> &startx, const QVector<qint32> &endx, const QVector<qint32> &fieldLine)
    : m_startx(startx), m_endx(endx), m_fieldLine(fieldLine)
{
}

DropOuts::DropOuts(int reserve_size)
    : m_startx(), m_endx(), m_fieldLine()
{
    reserve(reserve_size);
}

// Assignment '=' operator
DropOuts& DropOuts::operator=(const DropOuts &inDropouts)
{
    // Do not allow assignment if both objects are the same
    if (this != &inDropouts) {
        // Copy the object's data
        m_startx = inDropouts.m_startx;
        m_endx = inDropouts.m_endx;
        m_fieldLine = inDropouts.m_fieldLine;;
    }

    return *this;
}

// Append a dropout
void DropOuts::append(const qint32 startx, const qint32 endx, const qint32 fieldLine)
{
    m_startx.append(startx);
    m_endx.append(endx);
    m_fieldLine.append(fieldLine);
}

void DropOuts::reserve(int size)
{
    m_startx.reserve(size);
    m_endx.reserve(size);
    m_fieldLine.reserve(size);
}

// Resize the size of the DropOuts record
void DropOuts::resize(qint32 size)
{
    m_startx.resize(size);
    m_endx.resize(size);
    m_fieldLine.resize(size);
}

// Clear the DropOuts record
void DropOuts::clear()
{
    m_startx.clear();
    m_endx.clear();
    m_fieldLine.clear();
}

// Method to concatenate dropouts on the same line that are close together
// (to cut down on the amount of generated dropout data when processing noisy/bad sources)

// TODO: Sort the DOs by fieldLine otherwise this won't work correctly unless the
// caller is aware of the restriction (used in ld-diffdod only at the moment)
void DropOuts::concatenate()
{
    qint32 sizeAtStart = m_startx.size();

    // This variable controls the minimum allowed gap between dropouts
    // if the gap between the end of the last dropout and the start of
    // the next is less than minimumGap, the two dropouts will be
    // concatenated together
    qint32 minimumGap = 50;

    // Start from dropout 1 as 0 has no previous dropout
    qint32 i = 1;

    while (i < m_startx.size()) {
        // Is the current dropout on the same frame line as the last?
        if (m_fieldLine[i - 1] == m_fieldLine[i]) {
            if ((m_endx[i - 1] + minimumGap) > (m_startx[i])) {
                // Concatenate
                m_endx[i - 1] = m_endx[i];

                // Remove the current dropout
                m_startx.removeAt(i);
                m_endx.removeAt(i);
                m_fieldLine.removeAt(i);
            }
        }

        // Next dropout
        i++;
    }

    qDebug() << "Concatenated dropouts: was" << sizeAtStart << "now" << m_startx.size() << "dropouts";
}

// Custom debug streaming operator
QDebug operator<<(QDebug dbg, DropOuts &dropOuts)
{
    dbg.nospace() << "Dropout object contains " << dropOuts.size() << " entries:\n";

    for (qint32 i = 0; i < dropOuts.size(); i++) {
        dbg.nospace() << "  [" << i << "] startx = " << dropOuts.startx(i) <<
                         " - endx = " << dropOuts.endx(i) << " - line = " << dropOuts.fieldLine(i) << "\n";
    }

    return dbg.maybeSpace();
}

// Read DropOuts from JSON
void DropOuts::read(JsonReader &reader)
{
    reader.beginObject();

    std::string member;
    while (reader.readMember(member)) {
        if (member == "endx") readArray(reader, m_endx);
        else if (member == "fieldLine") readArray(reader, m_fieldLine);
        else if (member == "startx") readArray(reader, m_startx);
        else reader.discard();
    }

    if (m_endx.size() != m_fieldLine.size() || m_endx.size() != m_startx.size()) {
        reader.throwError("dropout array sizes do not match");
    }

    reader.endObject();
}

// Write DropOuts to JSON
void DropOuts::write(JsonWriter &writer) const
{
    assert(!empty());

    writer.beginObject();

    // Keep members in alphabetical order
    writer.writeMember("endx");
    writeArray(writer, m_endx);
    writer.writeMember("fieldLine");
    writeArray(writer, m_fieldLine);
    writer.writeMember("startx");
    writeArray(writer, m_startx);

    writer.endObject();
}

// Read an array of values from JSON
void DropOuts::readArray(JsonReader &reader, QVector<qint32> &array)
{
    array.clear();

    reader.beginArray();

    while (reader.readElement()) {
        qint32 value;
        reader.read(value);
        array.push_back(value);
    }

    reader.endArray();
}

// Write an array of values to JSON
void DropOuts::writeArray(JsonWriter &writer, const QVector<qint32> &array) const
{
    writer.beginArray();

    for (auto value : array) {
        writer.writeElement();
        writer.write(value);
    }

    writer.endArray();
}
