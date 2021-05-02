/************************************************************************

    sourceaudio.cpp

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

#include "sourceaudio.h"

SourceAudio::SourceAudio()
{
    audioFileByteLength = 0;
}

SourceAudio::~SourceAudio()
{
    if (audioFileByteLength != 0) inputAudioFile.close();
}

// Open an audio source file
bool SourceAudio::open(QFileInfo inputFileInfo)
{
    // Get the input audio fileinfo from the input TBC fileinfo:
    QFileInfo inputAudioFileInfo(inputFileInfo.absolutePath() + "/" + inputFileInfo.baseName() + ".pcm");

    // Open the audio source data file
    inputAudioFile.setFileName(inputAudioFileInfo.filePath());
    if (!inputAudioFile.open(QIODevice::ReadOnly)) {
        // Failed to open named input file
        qDebug() << "Could not open" << inputAudioFileInfo.filePath() << "as source audio input file";
        qFatal("Could not open PCM audio file!");
        return false;
    }

    // Get the length of the PCM audio file
    audioFileByteLength = inputAudioFileInfo.size();
    if (audioFileByteLength == 0) {
        qDebug() << "Could get file size of" << inputAudioFileInfo.filePath() << "(or file was 0 bytes length)";
        qFatal("Could not get PCM audio file length!");
        return false;
    }

    return true;
}

// Close an audio source file
void SourceAudio::close()
{
    // Close the audio source data file
    inputAudioFile.close();
    audioFileByteLength = 0;
}

// Get audio data for a single field from the audio source file
SourceAudio::Data SourceAudio::getAudioData(qint32 startSample, qint32 numberOfSamples)
{
    // Create a buffer for the sample data
    SourceAudio::Data sampleData;

    // Check that audio data is available
    if (audioFileByteLength == 0) {
        qFatal("getAudioData requested, but no sample data is available (not open?)!");
        return sampleData;
    }

    // Translate the start and number from stereo 16-bit pair samples to bytes (x4)
    qint64 startByte = static_cast<qint64>(startSample) * 4;
    qint64 lengthInBytes = static_cast<qint64>(numberOfSamples) * 4;

    // Range check the request
    if (startByte < 0) {
        qFatal("getAudioData requested, but startSample was less than 0!");
        return sampleData;
    }

    if (lengthInBytes < 1) {
        qFatal("getAudioData requested, but numberOfSamples was less than 1!");
        return sampleData;
    }

    if ((startByte + lengthInBytes) > audioFileByteLength) {
        qFatal("getAudioData requested, but startSample + numberOfSamples was out of bounds!");
        return sampleData;
    }

    // Seek to the correct file position (if not already there)
    if (!inputAudioFile.seek(startByte)) {
        // Seek failed
        qFatal("Could not seek to field position in input audio file!");
        return sampleData;
    }

    // Define a data stream to read 16 bit values from the audio file
    QDataStream audioData(&inputAudioFile);
    audioData.setByteOrder(QDataStream::LittleEndian);
    qint16 x;
    for (qint32 sample = 0; sample < numberOfSamples; sample++) {
        // Left 16 bit sample
        audioData >> x;
        if (audioData.atEnd()) {
            qFatal("getAudioData hit premature end of file!");
            return sampleData;
        }
        sampleData.append(x);

        // Right 16 bit sample
        audioData >> x;
        if (audioData.atEnd()) {
            qFatal("getAudioData hit premature end of file!");
            return sampleData;
        }
        sampleData.append(x);
    }

    return sampleData;
}

