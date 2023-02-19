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

#ifndef DROPOUTS_H
#define DROPOUTS_H

#include <QDebug>
#include <QtGlobal>
#include <QMetaType>

class JsonReader;
class JsonWriter;

class DropOuts
{
public:
    DropOuts() = default;
    DropOuts(int reserve);
    ~DropOuts() = default;
    DropOuts(const DropOuts &) = default;

    DropOuts(const QVector<qint32> &startx, const QVector<qint32> &endx, const QVector<qint32> &fieldLine);
    DropOuts &operator=(const DropOuts &);

    void append(const qint32 startx, const qint32 endx, const qint32 fieldLine);
    void reserve(int size);
    void resize(qint32 size);
    void clear();
    void concatenate();

    // Return the number of dropouts
    qint32 size() const {
        return m_startx.size();
    }

    // Return true if there are no dropouts
    bool empty() const {
        return m_startx.empty();
    }

    // Get position of a dropout
    qint32 startx(qint32 index) const {
        return m_startx[index];
    }
    qint32 endx(qint32 index) const {
        return m_endx[index];
    }
    qint32 fieldLine(qint32 index) const {
        return m_fieldLine[index];
    }

    void read(JsonReader &reader);
    void write(JsonWriter &writer) const;

private:
    QVector<qint32> m_startx;
    QVector<qint32> m_endx;
    QVector<qint32> m_fieldLine;

    void readArray(JsonReader &reader, QVector<qint32> &array);
    void writeArray(JsonWriter &writer, const QVector<qint32> &array) const;
};

#endif // DROPOUTS_H
