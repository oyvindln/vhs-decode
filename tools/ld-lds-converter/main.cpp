/************************************************************************

    main.cpp

    ld-lds-converter - 10-bit to 16-bit .lds converter for ld-decode
    Copyright (C) 2019-2020 Simon Inns

    This file is part of ld-decode-tools.

    ld-lds-converter is free software: you can redistribute it and/or
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
#include <QDebug>
#include <QtGlobal>
#include <QCommandLineParser>

#include "logging.h"
#include "dataconverter.h"

int main(int argc, char *argv[])
{
    // Install the local debug message handler
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    // Set application name and version
    QCoreApplication::setApplicationName("ld-lds-converter");
    QCoreApplication::setApplicationVersion(QString("Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-lds-converter - 10-bit to 16-bit .lds converter for ld-decode\n"
                "\n"
                "(c)2018-2020 Simon Inns\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
    parser.addHelpOption();
    parser.addVersionOption();

    // Add the standard debug options --debug and --quiet
    addStandardDebugOptions(parser);

    // Option to specify input video file (-i)
    QCommandLineOption sourceVideoFileOption(QStringList() << "i" << "input",
                QCoreApplication::translate("main", "Specify input laserdisc sample file (default is stdin)"),
                QCoreApplication::translate("main", "file"));
    parser.addOption(sourceVideoFileOption);

    // Option to specify output video file (-o)
    QCommandLineOption targetVideoFileOption(QStringList() << "o" << "output",
                QCoreApplication::translate("main", "Specify output laserdisc sample file (default is stdout)"),
                QCoreApplication::translate("main", "file"));
    parser.addOption(targetVideoFileOption);

    // Option to unpack 10-bit data (-u)
    QCommandLineOption showUnpackOption(QStringList() << "u" << "unpack",
                                       QCoreApplication::translate("main", "Unpack 10-bit data into 16-bit (default)"));
    parser.addOption(showUnpackOption);

    // Option to pack 16-bit data (-p)
    QCommandLineOption showPackOption(QStringList() << "p" << "pack",
                                       QCoreApplication::translate("main", "Pack 16-bit data into 10-bit"));
    parser.addOption(showPackOption);

    // Option to unpack 10-bit data with RIFF WAV headers (-r)
    QCommandLineOption showRIFFOption(QStringList() << "r" << "riff",
                                        QCoreApplication::translate("main", "Unpack 10-bit data into 16-bit with RIFF WAV headers (use this ONLY for FlaCCL)"));
    parser.addOption(showRIFFOption);

    // Process the command line arguments given by the user
    parser.process(a);

    // Standard logging options
    processStandardDebugOptions(parser);

    // Get the configured settings from the parser
    bool isUnpacking = parser.isSet(showUnpackOption);
    bool isPacking = parser.isSet(showPackOption);
    bool isRIFF = parser.isSet(showRIFFOption);
    QString inputFileName = parser.value(sourceVideoFileOption);
    QString outputFileName = parser.value(targetVideoFileOption);

    bool modeUnpack = true;
    if (isPacking) modeUnpack = false;

    bool modeRIFF = false;
    if (isRIFF) modeRIFF = true;

    // Check that both pack and unpack are not set
    if (isUnpacking && isPacking) {
        // Quit with error
        qCritical("Specify only --unpack (-u) or --pack (-p) - not both!");
        return -1;
    }

    // Check that RIFF option is not set with pack (only unpack)
    if (isRIFF && ! isUnpacking) {
        // Quit with error
        qCritical("You can only write RIFF headers with --unpack (-u)");
        return -1;
    }

    // Initialise the data conversion object
    DataConverter dataConverter(inputFileName, outputFileName, !modeUnpack, modeRIFF);

    // Process the data conversion
    dataConverter.process();

    // Quit with success
    return 0;
}
