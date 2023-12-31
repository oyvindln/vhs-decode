/************************************************************************

    logging.h

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

#ifndef LOGGING_H
#define LOGGING_H

#include <QCoreApplication>
#include <QDebug>
#include <QFile>
#include <QString>
#include <QCommandLineParser>

#ifdef _WIN32
    #include <io.h>
    #include <fcntl.h>
#endif

// Prototypes
void debugOutputHandler(QtMsgType type, const QMessageLogContext &context, const QString &msg);
void setDebug(bool state);
void setQuiet(bool state);
void setBinaryMode(void);
void openDebugFile(QString filename);
void closeDebugFile(void);
void addStandardDebugOptions(QCommandLineParser &parser);
void processStandardDebugOptions(QCommandLineParser &parser);
bool getDebugState();

#endif // LOGGING_H
