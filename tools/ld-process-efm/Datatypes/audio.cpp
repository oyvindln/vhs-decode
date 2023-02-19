/************************************************************************

    audio.cpp

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

#include "audio.h"

Audio::Audio()
{
    // Generate a silent sample
    for (qint32 i = 0; i < 24; i++) sample.sampleFrame[i] = 0;

    // Populate the sample values from the frame
    createSampleValuesFromFrame();
}

// Method to set the audio sample frame data from a F2 Frame
Audio::Audio(const uchar *_sampleFrame)
{
    for (qint32 i = 0; i < 24; i++) sample.sampleFrame[i] = _sampleFrame[i];

    // Populate the sample values from the frame
    createSampleValuesFromFrame();
}

// Method to get the audio data as a sample frame (of 24 bytes)
const uchar *Audio::getSampleFrame() const
{
    return sample.sampleFrame;
}

// Method to set the signed 16-bit sample values
void Audio::setSampleValues(const Audio::SampleValues &_sampleValues)
{
    sample.sampleValues = _sampleValues;

    // Create the sample frame data from the passed sample values
    createSampleFrameFromValues();
}

// Method to get the signed 16-bit sample values
const Audio::SampleValues &Audio::getSampleValues() const
{
    return sample.sampleValues;
}

// Method to set the signed 16-bit sample values to silence
void Audio::setSampleToSilence()
{
    silenceSample();
}

// Private methods ----------------------------------------------------------------------------------------------------

// Method to create the 16-bit signed sample values from the 24-bit frame
void Audio::createSampleValuesFromFrame()
{
    // Convert frame data to qint16 samples
    qint16* pcmDataBlock = reinterpret_cast<qint16*>(sample.sampleFrame);
    sample.sampleValues.leftSamples[0]  = pcmDataBlock[0];
    sample.sampleValues.rightSamples[0] = pcmDataBlock[1];
    sample.sampleValues.leftSamples[1]  = pcmDataBlock[2];
    sample.sampleValues.rightSamples[1] = pcmDataBlock[3];
    sample.sampleValues.leftSamples[2]  = pcmDataBlock[4];
    sample.sampleValues.rightSamples[2] = pcmDataBlock[5];
    sample.sampleValues.leftSamples[3]  = pcmDataBlock[6];
    sample.sampleValues.rightSamples[3] = pcmDataBlock[7];
    sample.sampleValues.leftSamples[4]  = pcmDataBlock[8];
    sample.sampleValues.rightSamples[4] = pcmDataBlock[9];
    sample.sampleValues.leftSamples[5]  = pcmDataBlock[10];
    sample.sampleValues.rightSamples[5] = pcmDataBlock[11];
}

// Method to create the 24-bit frame from the 16-bit signed sample values
void Audio::createSampleFrameFromValues()
{
    // Create the PCM data from the passed sample values
    uchar *leftSampleFrameData = reinterpret_cast<uchar*>(sample.sampleValues.leftSamples);
    uchar *rightSampleFrameData = reinterpret_cast<uchar*>(sample.sampleValues.rightSamples);

    qint32 pointer = 0;
    for (qint32 i = 0; i < 12; i += 2) {
        sample.sampleFrame[pointer + 0] = leftSampleFrameData[i];
        sample.sampleFrame[pointer + 1] = leftSampleFrameData[i + 1];
        sample.sampleFrame[pointer + 2] = rightSampleFrameData[i];
        sample.sampleFrame[pointer + 3] = rightSampleFrameData[i + 1];
        pointer += 4;
    }
}

// Method to set a sample to silence
void Audio::silenceSample()
{
    // Silence sample values
    for (qint32 i = 0; i < 6; i++) {
        sample.sampleValues.leftSamples[i] = 0;
        sample.sampleValues.rightSamples[i] = 0;
    }

    // Silence sample frame
    for (qint32 i = 0; i < 24; i++) sample.sampleFrame[i] = 0;
}









