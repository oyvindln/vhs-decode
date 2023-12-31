/************************************************************************

    main.cpp

    ld-process-vits - Vertical Interval Test Signal processing
    Copyright (C) 2020 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-vits is free software: you can redistribute it and/or
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
#include <QThread>
#include <QFile>
#include <QFileInfo>

#include "logging.h"
#include "lddecodemetadata.h"
#include "sourcevideo.h"
#include "processingpool.h"

int main(int argc, char *argv[])
{
    //set 'binary mode' for stdin and stdout on windows
    setBinaryMode();
    // Install the local debug message handler
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    // Set application name and version
    QCoreApplication::setApplicationName("ld-process-vits");
    QCoreApplication::setApplicationVersion(QString("Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser ---------------------------------------------------------------------------------
    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-process-vits - Vertical Interval Test Signal processing\n"
                "\n"
                "(c)2020 Simon Inns\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
    parser.addHelpOption();
    parser.addVersionOption();

    // Add the standard debug options --debug and --quiet
    addStandardDebugOptions(parser);

    // Option to specify a different JSON input file
    QCommandLineOption inputJsonOption(QStringList() << "input-json",
                                       QCoreApplication::translate("main", "Specify the input JSON file (default input.json)"),
                                       QCoreApplication::translate("main", "filename"));
    parser.addOption(inputJsonOption);

    // Option to specify a different JSON output file
    QCommandLineOption outputJsonOption(QStringList() << "output-json",
                                        QCoreApplication::translate("main", "Specify the output JSON file (default same as input)"),
                                        QCoreApplication::translate("main", "filename"));
    parser.addOption(outputJsonOption);

    // Option to disable JSON back-up (-n)
    QCommandLineOption showNoBackupOption(QStringList() << "n" << "nobackup",
                                       QCoreApplication::translate("main", "Do not create a backup of the input JSON metadata"));
    parser.addOption(showNoBackupOption);

    // Option to select the number of threads (-t)
    QCommandLineOption threadsOption(QStringList() << "t" << "threads",
                                        QCoreApplication::translate("main", "Specify the number of concurrent threads (default is the number of logical CPUs)"),
                                        QCoreApplication::translate("main", "number"));
    parser.addOption(threadsOption);

    // Positional argument to specify input TBC file
    parser.addPositionalArgument("input", QCoreApplication::translate("main", "Specify input TBC file"));

    // Process the command line options and arguments given by the user
    parser.process(a);

    // Standard logging options
    processStandardDebugOptions(parser);

    // Get the options from the parser
    bool noBackup = parser.isSet(showNoBackupOption);

    qint32 maxThreads = QThread::idealThreadCount();
    if (parser.isSet(threadsOption)) {
        maxThreads = parser.value(threadsOption).toInt();

        if (maxThreads < 1) {
            // Quit with error
            qCritical("Specified number of threads must be greater than zero");
            return -1;
        }
    }

    // Get the arguments from the parser
    QString inputFilename;
    QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.count() == 1) {
        inputFilename = positionalArguments.at(0);
    } else {
        // Quit with error
        qCritical("You must specify an input TBC file");
        return -1;
    }

    // Work out the metadata filenames
    QString inputJsonFilename = inputFilename + ".json";
    if (parser.isSet(inputJsonOption)) {
        inputJsonFilename = parser.value(inputJsonOption);
    }
    QString outputJsonFilename = inputJsonFilename;
    if (parser.isSet(outputJsonOption)) {
        outputJsonFilename = parser.value(outputJsonOption);
    }

    // Open the source video metadata
    LdDecodeMetaData metaData;
    qInfo().nospace().noquote() << "Reading JSON metadata from " << inputJsonFilename;
    if (!metaData.read(inputJsonFilename)) {
        qCritical() << "Unable to open TBC JSON metadata file";
        return 1;
    }

    // If we're overwriting the input JSON file, back it up first
    if (inputJsonFilename == outputJsonFilename && !noBackup) {
        qInfo().nospace().noquote() << "Backing up JSON metadata to " << inputJsonFilename << ".vbup";
        if (!QFile::copy(inputJsonFilename, inputJsonFilename + ".vbup")) {
            qCritical() << "Unable to back-up input JSON metadata file - back-up already exists?";
            return 1;
        }
    }

    // Perform the processing
    qInfo() << "Beginning VITS processing...";
    ProcessingPool processingPool(inputFilename, outputJsonFilename, maxThreads, metaData);
    if (!processingPool.process()) return 1;

    // Quit with success
    return 0;
}
