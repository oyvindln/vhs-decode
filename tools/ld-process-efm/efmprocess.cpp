/************************************************************************

    efmprocess.cpp

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

#include "efmprocess.h"

#include <vector>

EfmProcess::EfmProcess()
{
    debug_efmToF3Frames = false;
    debug_syncF3Frames = false;
    debug_f3ToF2Frames = false;
    debug_f2ToF1Frame = false;
    debug_f1ToAudio = false;
    debug_f1ToData = false;

    padInitialDiscTime = false;

    decodeAsAudio = true;
    decodeAsData = false;
    noTimeStamp = false;
}

// Set the detailed debug output flags
void EfmProcess::setDebug(bool _debug_efmToF3Frames, bool _debug_syncF3Frames,
                          bool _debug_f3ToF2Frames, bool _debug_f2ToF1Frames,
                          bool _debug_f1ToAudio, bool _debug_f1ToData)
{
    qDebug() << "EfmProcess::setDebug(): Setting detailed debug flags";
    debug_efmToF3Frames = _debug_efmToF3Frames;
    debug_syncF3Frames = _debug_syncF3Frames;
    debug_f3ToF2Frames = _debug_f3ToF2Frames;
    debug_f2ToF1Frame = _debug_f2ToF1Frames;
    debug_f1ToAudio = _debug_f1ToAudio;
    debug_f1ToData = _debug_f1ToData;
}

// Set the audio error treatment type
void EfmProcess::setAudioErrorTreatment(EfmProcess::ErrorTreatment _errorTreatment)
{
    qDebug() << "EfmProcess::setAudioErrorTreatment(): Error treatment =" << _errorTreatment;

    // Set the audio error treatment option
    errorTreatment = _errorTreatment;

    // Set the conceal type option (THIS SHOULD BE REMOVED)
    concealType = F1ToAudio::ConcealType::linear;
}

// Set the decoder options
void EfmProcess::setDecoderOptions(bool _padInitialDiscTime, bool _decodeAsData, bool _audioIsDts, bool _noTimeStamp)
{
    qDebug() << "EfmProcess::setDecoderOptions(): Pad initial disc time is" << _padInitialDiscTime;
    qDebug() << "EfmProcess::setDecoderOptions(): Audio-is-DTS is" << _audioIsDts;
    qDebug() << "EfmProcess::setDecoderOptions(): No time-stamp is" << _noTimeStamp;
    padInitialDiscTime = _padInitialDiscTime;

    // Either output audio or data
    if (_decodeAsData) {
        decodeAsAudio = false;
        decodeAsData = true;
        qDebug() << "EfmProcess::setDecoderOptions(): Decoding F1 frames as data";
    } else {
        decodeAsAudio = true;
        decodeAsData = false;
        qDebug() << "EfmProcess::setDecoderOptions(): Decoding F1 frames as audio";
    }

    audioIsDts = _audioIsDts;
    noTimeStamp = _noTimeStamp;
}

// Output the result of the decode to qInfo
void EfmProcess::reportStatistics() const
{
    efmToF3Frames.reportStatistics();
    syncF3Frames.reportStatistics();
    f3ToF2Frames.reportStatistics();
    f2ToF1Frames.reportStatistics();
    if (decodeAsAudio) f1ToAudio.reportStatistics();
    if (decodeAsData) f1ToData.reportStatistics();
}

// Process the EFM file
bool EfmProcess::process(QString inputFilename, QString outputFilename)
{
    // Start of file handling
    QFile inputFileHandle;
    QFile outputFileHandle;

    // Ensure input file is valid
    if (inputFilename.isEmpty()) {
        qDebug() << "EfmDecoder::process(): Input EFM filename is empty!";
        return false;
    }
    if (inputFilename == outputFilename) {
        qDebug() << "EfmDecoder::process(): Input and output files cannot be the same!";
        return false;
    }

    // Open the input EFM data file
    inputFileHandle.setFileName(inputFilename);
    if (!inputFileHandle.open(QIODevice::ReadOnly)) {
        // Failed to open file
        qDebug() << "EfmDecoder::process(): Could not open EFM input file";
        return false;
    } else qDebug() << "EfmDecoder::process(): Opened EFM input file:" << inputFilename;

    // Open output file
    outputFileHandle.setFileName(outputFilename);
    if (outputFileHandle.exists()) outputFileHandle.remove();
    if (!outputFileHandle.open(QIODevice::WriteOnly)) {
        // Failed to open file
        qFatal("EfmDecoder::process(): Could not open output file - this is fatal!");
        return false;
    } else qDebug() << "EfmDecoder::process(): Opened output file:" << outputFilename;

    qDebug() << "EfmProcess::process(): Starting EFM processing";

    // Clear EFM decoding statistics
    reset();

    // Variables for processing progress reporting
    qint64 initialInputFileSize = inputFileHandle.bytesAvailable();
    qint32 lastPercent = 0;

    // Set input EFM data buffer size in 256K blocks
    qint32 bufferSize = 1024 * 256;

    while(inputFileHandle.bytesAvailable() > 0) {
        // Get a buffer of EFM data
        QByteArray inputEfmBuffer;
        inputEfmBuffer.resize(bufferSize);

        qint64 bytesRead = inputFileHandle.read(inputEfmBuffer.data(), inputEfmBuffer.size());
        if (bytesRead != bufferSize) inputEfmBuffer.resize(static_cast<qint32>(bytesRead));

        // Perform EFM processing
        const std::vector<F3Frame> &initialF3Frames = efmToF3Frames.process(inputEfmBuffer, debug_efmToF3Frames, audioIsDts);
        const std::vector<F3Frame> &syncedF3Frames = syncF3Frames.process(initialF3Frames, debug_syncF3Frames);
        const std::vector<F2Frame> &f2Frames = f3ToF2Frames.process(syncedF3Frames, debug_f3ToF2Frames, noTimeStamp);
        const std::vector<F1Frame> &f1Frames = f2ToF1Frames.process(f2Frames, debug_f2ToF1Frame, noTimeStamp);

        // Process as either audio or data
        if (decodeAsAudio) {
            outputFileHandle.write(f1ToAudio.process(f1Frames, padInitialDiscTime, errorTreatment, concealType, debug_f1ToAudio));
        } else {
            outputFileHandle.write(f1ToData.process(f1Frames, debug_f1ToData));
        }

        // Report progress to user
        double percent = 100 - (100.0 / static_cast<double>(initialInputFileSize)) * static_cast<double>(inputFileHandle.bytesAvailable());
        if (static_cast<qint32>(percent) > lastPercent) {
            qInfo().nospace() << "Processed " << static_cast<qint32>(percent) << "%";
        }
        lastPercent = static_cast<qint32>(percent);
    }

    // Check if audio is available
    if (f1ToAudio.getStatistics().totalSamples > 0) qDebug() << "EfmProcess::process(): Audio is available";
    if (f1ToData.getStatistics().totalSectors > 0) qDebug() << "EfmProcess::process(): Data is available";

    // Close all files
    inputFileHandle.close();
    outputFileHandle.close();

    // Report the final decode statistics to qInfo
    reportStatistics();

    // Processing complete
    qDebug() << "EfmProcess::process(): EFM processing complete";
    return true;
}

// Return statistics about the decoding process
EfmProcess::Statistics EfmProcess::getStatistics()
{
    // Gather statistics
    statistics.f3ToF2Frames = f3ToF2Frames.getStatistics();
    statistics.syncF3Frames = syncF3Frames.getStatistics();
    statistics.efmToF3Frames = efmToF3Frames.getStatistics();
    statistics.f2ToF1Frames = f2ToF1Frames.getStatistics();
    statistics.f1ToAudio = f1ToAudio.getStatistics();
    statistics.f1ToData = f1ToData.getStatistics();

    return statistics;
}

// Method to reset decoding classes
void EfmProcess::reset()
{
    efmToF3Frames.reset();
    syncF3Frames.reset();
    f3ToF2Frames.reset();
    f2ToF1Frames.reset();
    f1ToAudio.reset();
    f1ToData.reset();
}
