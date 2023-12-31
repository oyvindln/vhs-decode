/************************************************************************

    main.cpp

    ld-chroma-encoder - Composite video encoder
    Copyright (C) 2019-2022 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-encoder is free software: you can redistribute it and/or
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
#include <QFile>
#include <QtGlobal>
#include <QCommandLineParser>
#include <cstdio>

#ifdef _WIN32
	#include <io.h>
	#include <fcntl.h>
#endif

#include "lddecodemetadata.h"
#include "logging.h"

#include "ntscencoder.h"
#include "palencoder.h"

int main(int argc, char *argv[])
{
	#ifdef _WIN32
	_setmode(_fileno(stdout), O_BINARY);
	_setmode(_fileno(stdin), O_BINARY);	
	#endif
    // Install the local debug message handler
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    // Set application name and version
    QCoreApplication::setApplicationName("ld-chroma-encoder");
    QCoreApplication::setApplicationVersion(QString("Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-chroma-encoder - Composite video encoder\n"
                "\n"
                "(c)2019-2022 Adam Sampson\n"
                "(c)2022 Phillip Blucas\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
    parser.addHelpOption();
    parser.addVersionOption();

    // -- General options --

    // Add the standard debug options --debug and --quiet
    addStandardDebugOptions(parser);

    // Option to select video system (-f)
    QCommandLineOption systemOption(QStringList() << "f" << "system",
                                    QCoreApplication::translate("main", "Video system (PAL, NTSC; default PAL)"),
                                    QCoreApplication::translate("main", "system"));
    parser.addOption(systemOption);

    // Option to select the input format (-p)
    QCommandLineOption inputFormatOption(QStringList() << "p" << "input-format",
                                       QCoreApplication::translate("main", "Input format (rgb, yuv; default rgb); RGB48, YUV444P16 formats are supported"),
                                       QCoreApplication::translate("main", "input-format"));
    parser.addOption(inputFormatOption);

    // Option to specify where to start in the field sequence (--field-offset)
    QCommandLineOption fieldOffsetOption(QStringList() << "field-offset",
                                         QCoreApplication::translate("main", "Offset of the first output field within the field sequence (0, 2 for NTSC; 0, 2, 4, 6 for PAL; default: 0)"),
                                         QCoreApplication::translate("main", "offset"));
    parser.addOption(fieldOffsetOption);

    // -- NTSC options --

    // Option to select chroma mode (--chroma-mode)
    QCommandLineOption chromaOption(QStringList() << "chroma-mode",
                                     QCoreApplication::translate("main", "NTSC: Chroma encoder mode to use (wideband-yuv, wideband-yiq, narrowband-q; default: wideband-yuv)"),
                                     QCoreApplication::translate("main", "chroma-mode"));
    parser.addOption(chromaOption);

    // Option to disable 7.5 IRE setup, as in NTSC-J (--no-setup)
    QCommandLineOption setupOption(QStringList() << "no-setup",
                                   QCoreApplication::translate("main", "NTSC: Output NTSC-J, without 7.5 IRE setup"));
    parser.addOption(setupOption);

    // -- PAL options --

    // Option to produce subcarrier-locked output (-c)
    QCommandLineOption scLockedOption(QStringList() << "c" << "sc-locked",
                                      QCoreApplication::translate("main", "PAL: Output samples are subcarrier-locked (default: line-locked)"));
    parser.addOption(scLockedOption);

    // -- Positional arguments --

    // Positional argument to specify input video file
    parser.addPositionalArgument("input", QCoreApplication::translate("main", "Specify input RGB/YCbCr file (- for piped input)"));

    // Positional argument to specify output video file
    parser.addPositionalArgument("output", QCoreApplication::translate("main", "Specify output TBC file"));

    // Positional argument to specify chroma output video file
    parser.addPositionalArgument("chroma", QCoreApplication::translate("main", "Specify chroma output TBC file (optional)"),
                                 "[chroma]");

    // Process the command line options and arguments given by the user
    parser.process(a);

    // Standard logging options
    processStandardDebugOptions(parser);

    VideoSystem system = PAL;
    QString systemName;
    if (parser.isSet(systemOption)) {
        systemName = parser.value(systemOption);
        if (!parseVideoSystemName(systemName.toUpper(), system)
            || (system != NTSC && system != PAL)) {
            // Quit with error
            qCritical("Unsupported color system");
            return -1;
        }
    }

    int fieldOffset = 0;
    if (parser.isSet(fieldOffsetOption)) {
        fieldOffset = parser.value(fieldOffsetOption).toInt();
        if ((fieldOffset % 2) != 0
            || fieldOffset < 0
            || (system == PAL && fieldOffset > 7)
            || (system == NTSC && fieldOffset > 3)) {
            // Quit with error
            qCritical("Field offset must be 0 or 2 for NTSC, or 0, 2, 4 or 6 for PAL");
            return -1;
        }
    }

    bool addSetup = !parser.isSet(setupOption);

    // Select the input format
    bool isComponent = false;
    QString inputFormatName;
    if (parser.isSet(inputFormatOption)) {
        inputFormatName = parser.value(inputFormatOption);
    } else {
        inputFormatName = "rgb";
    }
    if (inputFormatName == "yuv") {
        isComponent = true;
    } else if (inputFormatName == "rgb") {
        isComponent = false;
    } else {
        qCritical() << "Unknown input format" << inputFormatName;
        return -1;
    }

    ChromaMode chromaMode = WIDEBAND_YUV;
    QString chromaName;
    if (parser.isSet(chromaOption)) {
        chromaName = parser.value(chromaOption);
        if (chromaName == "wideband-yiq") {
            chromaMode = WIDEBAND_YIQ;
        } else if (chromaName == "narrowband-q") {
            chromaMode = NARROWBAND_Q;
        } else if (chromaName == "wideband-yuv") {
            chromaMode = WIDEBAND_YUV;
        } else {
            // Quit with error
            qCritical("Unsupported chroma encoder mode");
            return -1;
        }
    }

    const bool scLocked = parser.isSet(scLockedOption);

    // Get the arguments from the parser
    QString inputFileName;
    QString outputFileName;
    QString chromaFileName;
    QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.count() == 2 || positionalArguments.count() == 3) {
        inputFileName = positionalArguments.at(0);
        outputFileName = positionalArguments.at(1);
        if (positionalArguments.count() > 2) {
            chromaFileName = positionalArguments.at(2);
        }
    } else {
        // Quit with error
        qCritical("You must specify the input RGB/YCbCr and output TBC files");
        return -1;
    }

    if (inputFileName == outputFileName) {
        // Quit with error
        qCritical("Input and output files cannot be the same");
        return -1;
    }

    // Open the input file
    QFile inputFile(inputFileName);
    if (inputFileName == "-") {
        if (!inputFile.open(stdin, QFile::ReadOnly)) {
            qCritical("Cannot open stdin");
            return -1;
        }
    } else {
        if (!inputFile.open(QFile::ReadOnly)) {
            qCritical() << "Cannot open input file:" << inputFileName;
            return -1;
        }
    }

    // Open the main output file
    QFile tbcFile(outputFileName);
    if (!tbcFile.open(QFile::WriteOnly)) {
        qCritical() << "Cannot open output file:" << outputFileName;
        return -1;
    }

    // Open the chroma output file, if specified
    QFile chromaFile(chromaFileName);
    if (chromaFileName != "" && !chromaFile.open(QFile::WriteOnly)) {
        qCritical() << "Cannot open chroma output file:" << chromaFileName;
        return -1;
    }

    // Encode the data
    LdDecodeMetaData metaData;
    if (system == NTSC) {
        NTSCEncoder encoder(inputFile, tbcFile, chromaFile, metaData, fieldOffset, isComponent, chromaMode, addSetup);
        if (!encoder.encode()) {
            return -1;
        }
    } else {
        PALEncoder encoder(inputFile, tbcFile, chromaFile, metaData, fieldOffset, isComponent, scLocked);
        if (!encoder.encode()) {
            return -1;
        }
    }

    // Write the metadata
    if (!metaData.write(outputFileName + ".json")) {
        return -1;
    }

    // Quit with success
    return 0;
}
