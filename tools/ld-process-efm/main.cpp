/************************************************************************

    main.cpp

    ld-process-efm - EFM data decoder
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-efm is free software: you can redistribute it and/or
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

#include "logging.h"
#include "efmprocess.h"

int main(int argc, char *argv[])
{
    // Install the local debug message handler
    setDebug(true);
    qInstallMessageHandler(debugOutputHandler);

    QCoreApplication a(argc, argv);

    // Set application name and version
    QCoreApplication::setApplicationName("ld-process-efm");
    QCoreApplication::setApplicationVersion(QString("Branch: %1 / Commit: %2").arg(APP_BRANCH, APP_COMMIT));
    QCoreApplication::setOrganizationDomain("domesday86.com");

    // Set up the command line parser
    QCommandLineParser parser;
    parser.setApplicationDescription(
                "ld-process-efm - EFM data decoder\n"
                "\n"
                "(c)2019-2022 Simon Inns\n"
                "GPLv3 Open-Source - github: https://github.com/happycube/ld-decode");
    parser.addHelpOption();
    parser.addVersionOption();

    // Add the standard debug options --debug and --quiet
    addStandardDebugOptions(parser);

    // Audio processing options
    QCommandLineOption concealAudioOption(QStringList() << "c" << "conceal",
                                       QCoreApplication::translate("main", "Conceal corrupt audio data (default)"));
    parser.addOption(concealAudioOption);

    QCommandLineOption silenceAudioOption(QStringList() << "s" << "silence",
                                       QCoreApplication::translate("main", "Silence corrupt audio data"));
    parser.addOption(silenceAudioOption);

    QCommandLineOption passThroughAudioOption(QStringList() << "g" << "pass-through",
                                       QCoreApplication::translate("main", "Pass-through corrupt audio data"));
    parser.addOption(passThroughAudioOption);

    // General decoder options
    QCommandLineOption padOption(QStringList() << "p" << "pad",
                                       QCoreApplication::translate("main", "Pad start of audio from 00:00 to match initial disc time"));
    parser.addOption(padOption);

    QCommandLineOption decodeAsDataOption(QStringList() << "b" << "data",
                                       QCoreApplication::translate("main", "Decode F1 frames as data instead of audio"));
    parser.addOption(decodeAsDataOption);

    QCommandLineOption audioIsDtsOption(QStringList() << "D" << "dts",
                                        QCoreApplication::translate("main", "Audio is DTS rather than PCM (allow non-standard F3 syncs)"));
    parser.addOption(audioIsDtsOption);

    QCommandLineOption noTimeStampOption(QStringList() << "t" << "time",
                                       QCoreApplication::translate("main", "Non-standard audio decode (no time-stamp information)"));
    parser.addOption(noTimeStampOption);

    // Detailed debuging options
    QCommandLineOption debug_efmToF3FramesOption(QStringList() << "debug-efmtof3frames",
                                       QCoreApplication::translate("main", "Show EFM To F3 frame decode detailed debug"));
    parser.addOption(debug_efmToF3FramesOption);

    QCommandLineOption debug_syncF3FramesOption(QStringList() << "debug-syncf3frames",
                                       QCoreApplication::translate("main", "Show F3 frame synchronisation detailed debug"));
    parser.addOption(debug_syncF3FramesOption);

    QCommandLineOption debug_f3ToF2FramesOption(QStringList() << "debug-f3tof2frames",
                                       QCoreApplication::translate("main", "Show F3 To F2 frame decode detailed debug"));
    parser.addOption(debug_f3ToF2FramesOption);

    QCommandLineOption debug_f2ToF1FrameOption(QStringList() << "debug-f2tof1frame",
                                       QCoreApplication::translate("main", "Show F2 to F1 frame detailed debug"));
    parser.addOption(debug_f2ToF1FrameOption);

    QCommandLineOption debug_f1ToAudioOption(QStringList() << "debug-f1toaudio",
                                       QCoreApplication::translate("main", "Show F1 to audio detailed debug"));
    parser.addOption(debug_f1ToAudioOption);

    QCommandLineOption debug_f1ToDataOption(QStringList() << "debug-f1todata",
                                       QCoreApplication::translate("main", "Show F1 to data detailed debug"));
    parser.addOption(debug_f1ToDataOption);

    // -- Positional arguments --
    // Positional argument to specify input EFM file
    parser.addPositionalArgument("input", QCoreApplication::translate("main", "Specify input EFM file"));

    // Positional argument to specify output audio file
    parser.addPositionalArgument("output", QCoreApplication::translate("main", "Specify output file"));

    // Process the command line options and arguments given by the user
    parser.process(a);

    // Standard logging options
    processStandardDebugOptions(parser);

    // Get the audio options from the parser.
    // Default to conceal for PCM audio, and passThrough for DTS audio.
    EfmProcess::ErrorTreatment errorTreatment = parser.isSet(audioIsDtsOption) ? EfmProcess::ErrorTreatment::passThrough
                                                                               : EfmProcess::ErrorTreatment::conceal;
    int numTreatments = 0;
    if (parser.isSet(concealAudioOption)) {
        errorTreatment = EfmProcess::ErrorTreatment::conceal;
        numTreatments++;
    }
    if (parser.isSet(silenceAudioOption)) {
        errorTreatment = EfmProcess::ErrorTreatment::silence;
        numTreatments++;
    }
    if (parser.isSet(passThroughAudioOption)) {
        errorTreatment = EfmProcess::ErrorTreatment::passThrough;
        numTreatments++;
    }
    if (numTreatments > 1) {
        qCritical() << "You may only specify one error treatment option (-c, -s or -g)";
        return 1;
    }

    // Get the decoding options from the parser
    bool pad = parser.isSet(padOption);
    bool decodeAsData = parser.isSet(decodeAsDataOption);
    bool audioIsDts = parser.isSet(audioIsDtsOption);
    bool noTimeStamp = parser.isSet(noTimeStampOption);

    // Get the additional debug options from the parser
    bool debug_efmToF3Frames = parser.isSet(debug_efmToF3FramesOption);
    bool debug_syncF3Frames = parser.isSet(debug_syncF3FramesOption);
    bool debug_f3ToF2Frames = parser.isSet(debug_f3ToF2FramesOption);
    bool debug_f2ToF1Frame = parser.isSet(debug_f2ToF1FrameOption);
    bool debug_f1ToAudio = parser.isSet(debug_f1ToAudioOption);
    bool debug_f1ToData = parser.isSet(debug_f1ToDataOption);

    // Get the filename arguments from the parser
    QString inputFilename;
    QString outputFilename;
    QStringList positionalArguments = parser.positionalArguments();
    if (positionalArguments.count() == 2) {
        inputFilename = positionalArguments.at(0);
        outputFilename = positionalArguments.at(1);
    } else {
        qWarning() << "You must specify the input EFM filename and the output filename";
        return 1;
    }

    // Perform the processing
    qInfo() << "Beginning EFM processing of" << inputFilename;
    EfmProcess efmProcess;
    efmProcess.setDebug(debug_efmToF3Frames, debug_syncF3Frames, debug_f3ToF2Frames,
                        debug_f2ToF1Frame, debug_f1ToAudio, debug_f1ToData);
    efmProcess.setDecoderOptions(pad, decodeAsData, audioIsDts, noTimeStamp);
    efmProcess.setAudioErrorTreatment(errorTreatment);

    if (!efmProcess.process(inputFilename, outputFilename)) return 1;

    // Quit with success
    return 0;
}
