/************************************************************************

    audio.h

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

#ifndef AUDIO_H
#define AUDIO_H

#include <QCoreApplication>
#include <QDebug>

class Audio
{
public:
    struct SampleValues {
        qint16 leftSamples[6];
        qint16 rightSamples[6];
    };

    struct Sample {
        uchar sampleFrame[24];
        SampleValues sampleValues;
    };

    Audio();
    Audio(const uchar *_sampleFrame);

    const uchar *getSampleFrame() const;
    void setSampleValues(const Audio::SampleValues &_sampleValues);
    const SampleValues &getSampleValues() const;
    void setSampleToSilence();

private:
    Sample sample;

    void createSampleValuesFromFrame();
    void createSampleFrameFromValues();
    void silenceSample();
};

#endif // AUDIO_H
