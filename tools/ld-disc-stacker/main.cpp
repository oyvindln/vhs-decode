/************************************************************************

    main.cpp

    ld-disc-stacker - Disc stacking for ld-decode
    Copyright (C) 2020-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-disc-stacker is free software: you can redistribute it and/or
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
#include "stackingpool.h"

int main(int argc, char *argv[])
{
    // Install the local debug message handler
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    // Set application name and version
    QCoreApplication::setApplicationName("ld-disc-stacker");
    QCoreApplication::setApplicationVersion(QString("Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser ---------------------------------------------------------------------------------
    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-disc-stacker - Disc stacking for ld-decode\n"
                "\n"
                "(c)2020-2022 Simon Inns\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
    parser.addHelpOption();
    parser.addVersionOption();

    // Add the standard debug options --debug and --quiet
    addStandardDebugOptions(parser);

    // Option to specify a different JSON input file
    QCommandLineOption inputJsonOption(QStringList() << "input-json",
                                       QCoreApplication::translate("main", "Specify the input JSON file for the first input file (default input.json)"),
                                       QCoreApplication::translate("main", "filename"));
    parser.addOption(inputJsonOption);

    // Option to specify a different JSON output file
    QCommandLineOption outputJsonOption(QStringList() << "output-json",
                                        QCoreApplication::translate("main", "Specify the output JSON file (default output.json)"),
                                        QCoreApplication::translate("main", "filename"));
    parser.addOption(outputJsonOption);

    // Option to reverse the field order (-r)
    QCommandLineOption setReverseOption(QStringList() << "r" << "reverse",
                                       QCoreApplication::translate("main", "Reverse the field order to second/first (default first/second)"));
    parser.addOption(setReverseOption);

    // Option to select the number of threads (-t)
    QCommandLineOption threadsOption(QStringList() << "t" << "threads",
                                        QCoreApplication::translate(
                                         "main", "Specify the number of concurrent threads (default is the number of logical CPUs)"),
                                        QCoreApplication::translate("main", "number"));
    parser.addOption(threadsOption);

    // Option to disable differential dropout detection
    QCommandLineOption noDiffDodOption(QStringList() << "no-diffdod",
                                        QCoreApplication::translate(
                                         "main", "Do not use differential dropout detection on low source pixels"));
    parser.addOption(noDiffDodOption);

    // Option to passthrough dropouts present in every source
    QCommandLineOption passthroughOption(QStringList() << "passthrough",
                                        QCoreApplication::translate(
                                         "main", "Pass-through dropouts present on every source"));
    parser.addOption(passthroughOption);

    // Positional argument to specify input video file
    parser.addPositionalArgument("inputs", QCoreApplication::translate(
                                     "main", "Specify input TBC files (- as first source for piped input)"));

    // Positional argument to specify output video file
    parser.addPositionalArgument("output", QCoreApplication::translate(
                                     "main", "Specify output TBC file (omit or - for piped output)"));

    // Process the command line options and arguments given by the user -----------------------------------------------
    parser.process(a);

    // Standard logging options
    processStandardDebugOptions(parser);

    // Get the options from the parser
    bool reverse = parser.isSet(setReverseOption);
    bool noDiffDod = parser.isSet(noDiffDodOption);
    bool passThrough = parser.isSet(passthroughOption);

    // Get the arguments from the parser
    qint32 maxThreads = QThread::idealThreadCount();
    if (parser.isSet(threadsOption)) {
        maxThreads = parser.value(threadsOption).toInt();

        if (maxThreads < 1) {
            // Quit with error
            qCritical("Specified number of threads must be greater than zero");
            return -1;
        }
    }

    // Require source and target filenames
    QVector<QString> inputFilenames;
    QString outputFilename = "-";
    QStringList positionalArguments = parser.positionalArguments();
    qint32 totalNumberOfInputFiles = positionalArguments.count() - 1;

    // Ensure we don't have more than 32 sources
    if (totalNumberOfInputFiles > 32) {
        qCritical() << "A maximum of 32 input TBC files are supported";
        return -1;
    }

    // Get the input TBC sources
    if (positionalArguments.count() >= 3) {
        // Resize the input filenames vector according to the number of input files supplied
        inputFilenames.resize(totalNumberOfInputFiles);

        for (qint32 i = 0; i < positionalArguments.count() - 1; i++) {
            inputFilenames[i] = positionalArguments.at(i);
        }

        // Warn if only 2 sources are used
        if (positionalArguments.count() == 3) {
            qInfo() << "Only 2 input sources specified - stack will be only based on averaging (3 or more sources are recommended)";
        }
    } else {
        // Quit with error
        qCritical("You must specify at least 2 input and 1 output TBC file");
        return -1;
    }

    // Get the output TBC (should be the last argument of the command line
    outputFilename = positionalArguments.at(positionalArguments.count() - 1);

    // If the first input filename is "-" (piped input) - verify a JSON file has been specified
    if (inputFilenames[0] == "-" && !parser.isSet(inputJsonOption)) {
        // Quit with error
        qCritical("With piped input, you must also specify the input JSON file with --input-json");
        return -1;
    }

    // If the output filename is "-" (piped output) - verify a JSON file has been specified
    if (outputFilename == "-" && !parser.isSet(outputJsonOption)) {
        // Quit with error
        qCritical("With piped output, you must also specify the output JSON file with --output-json");
        return -1;
    }

    // Check that none of the input filenames are used as the output file
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        if (inputFilenames[i] == outputFilename) {
            // Quit with error
            qCritical("Input and output files cannot have the same filenames");
            return -1;
        }
    }

    // Check that none of the input filenames are repeated
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        for (qint32 j = 0; j < totalNumberOfInputFiles; j++) {
            if (i != j) {
                if (inputFilenames[i] == inputFilenames[j]) {
                    // Quit with error
                    qCritical("Each input file should only be specified once - some filenames were repeated");
                    return -1;
                }
            }
        }
    }

    // Check that the output file does not already exist
    if (outputFilename != "-") {
        QFileInfo outputFileInfo(outputFilename);
        if (outputFileInfo.exists()) {
            // Quit with error
            qCritical("Specified output file already exists - will not overwrite");
            return -1;
        }
    }

    // Metadata filename for output TBC
    QString outputJsonFilename = outputFilename + ".json";
    if (parser.isSet(outputJsonOption)) {
        outputJsonFilename = parser.value(outputJsonOption);
    }

    // Prepare for stacking process -----------------------------------------------------------------------------------

    qInfo() << "Starting preparation for disc stacking processes...";
    // Open the source video metadata
    qDebug() << "main(): Opening source video metadata files..";
    QVector<LdDecodeMetaData *> ldDecodeMetaData;
    ldDecodeMetaData.resize(totalNumberOfInputFiles);
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        // Create an object for the source video
        ldDecodeMetaData[i] = new LdDecodeMetaData;
    }

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        // Work out the metadata filename
        QString jsonFilename = inputFilenames[i] + ".json";
        if (parser.isSet(inputJsonOption) && i == 0) jsonFilename = parser.value(inputJsonOption);
        qInfo().nospace().noquote() << "Reading input #" << i << " JSON metadata from " << jsonFilename;

        // Open it
        if (!ldDecodeMetaData[i]->read(jsonFilename)) {
            qCritical() << "Unable to open TBC JSON metadata file - cannot continue";
            return -1;
        }
    }

    // Reverse field order if required
    if (reverse) {
        qInfo() << "Expected field order is reversed to second field/first field";
        for (qint32 i = 0; i < totalNumberOfInputFiles; i++)
            ldDecodeMetaData[i]->setIsFirstFieldFirst(false);
    }

    // Show if DiffDOD is disabled
    if (noDiffDod) {
        qInfo() << "Differential Dropout Detection is disabled";
    }

    // Show if pass-through is enabled
    if (passThrough) {
        qInfo() << "Passing through dropouts present on every input source";
    }

    // Show and open input source TBC files
    qDebug() << "Opening source video files...";
    QVector<SourceVideo *> sourceVideos;
    sourceVideos.resize(totalNumberOfInputFiles);
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        // Create an object for the source video
        sourceVideos[i] = new SourceVideo;
    }

    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) {
        LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData[i]->getVideoParameters();

        qInfo().nospace() << "Opening input #" << i << ": " << videoParameters.fieldWidth << "x" << videoParameters.fieldHeight <<
                    " - input filename is " << inputFilenames[i];

        // Open the source TBC
        if (!sourceVideos[i]->open(inputFilenames[i], videoParameters.fieldWidth * videoParameters.fieldHeight)) {
            // Could not open source video file
            qInfo() << "Unable to open input source" << i;
            qInfo() << "Please verify that the specified source video files exist with the correct file permissions";
            return 1;
        }

        // Verify TBC and JSON input fields match
        if (sourceVideos[i]->getNumberOfAvailableFields() != ldDecodeMetaData[i]->getNumberOfFields()) {
            qInfo() << "Warning: TBC file contains" << sourceVideos[i]->getNumberOfAvailableFields() <<
                       "fields but the JSON indicates" << ldDecodeMetaData[i]->getNumberOfFields() <<
                       "fields - some fields will be ignored";
            qInfo() << "Update your copy of ld-decode and try again, this shouldn't happen unless the JSON metadata has been corrupted";
        }

        // Ensure source video has VBI data
        if (!ldDecodeMetaData[i]->getFieldVbi(1).inUse) {
            qInfo() << "Source video" << i << "does not appear to have valid VBI data in the JSON metadata.";
            qInfo() << "Please try running ld-process-vbi on the source video and then try again";
            return 1;
        }

        // Ensure that the video source standard matches the primary source
        if (ldDecodeMetaData[0]->getVideoParameters().system != videoParameters.system) {
            qInfo() << "All additional input sources must have the same video system as the initial source!";
            qInfo() << "Initial source is" << ldDecodeMetaData[0]->getVideoSystemDescription() << " and current source is" <<
                       ldDecodeMetaData[i]->getVideoSystemDescription();
            return 1;
        }

        if (!videoParameters.isMapped) {
            qInfo() << "Source video" << i << "has not been mapped - run ld-discmap on all source videos and try again";
            qInfo() << "Disc stacking relies on accurate VBI frame numbering to match source frames together";
            return 1;
        }
    }

    // Perform the disc stacking processes ----------------------------------------------------------------------------
    qInfo() << "Initial source checks are ok and sources are loaded";
    qint32 result = 0;
    StackingPool stackingPool(outputFilename, outputJsonFilename, maxThreads,
                                ldDecodeMetaData, sourceVideos, reverse, noDiffDod, passThrough);
    if (!stackingPool.process()) result = 1;

    // Close open source video files
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) sourceVideos[i]->close();

    // Remove metadata objects
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) delete ldDecodeMetaData[i];

    // Remove source video objects
    for (qint32 i = 0; i < totalNumberOfInputFiles; i++) delete sourceVideos[i];

    // Quit
    return result;
}
