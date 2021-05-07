/************************************************************************

    transformpal.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019-2021 Adam Sampson

    Reusing code from pyctools-pal, which is:
    Copyright (C) 2014 Jim Easterbrook

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
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

#include "transformpal.h"

#include <cassert>
#include <cmath>

TransformPal::TransformPal(qint32 _xComplex, qint32 _yComplex, qint32 _zComplex)
    : xComplex(_xComplex), yComplex(_yComplex), zComplex(_zComplex), configurationSet(false)
{
}

TransformPal::~TransformPal()
{
}

void TransformPal::updateConfiguration(const LdDecodeMetaData::VideoParameters &_videoParameters,
                                       TransformPal::TransformMode _mode, double threshold,
                                       const QVector<double> &_thresholds)
{
    videoParameters = _videoParameters;
    mode = _mode;

    // Resize thresholds to match the number of FFT bins we will consider in
    // applyFilter. The x loop there doesn't need to look at every bin.
    const qint32 thresholdsSize = ((xComplex / 4) + 1) * yComplex * zComplex;

    if (_thresholds.size() == 0) {
        // Use the same (squared) threshold value for all bins
        thresholds.fill(threshold * threshold, thresholdsSize);
    } else {
        // Square the provided thresholds
        assert(_thresholds.size() == thresholdsSize);
        thresholds.resize(thresholdsSize);
        for (int i = 0; i < thresholdsSize; i++) {
            thresholds[i] = _thresholds[i] * _thresholds[i];
        }
    }

    configurationSet = true;
}

void TransformPal::overlayFFT(qint32 positionX, qint32 positionY,
                              const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                              QVector<ComponentFrame> &componentFrames)
{
    // Visualise the first field for each frame
    for (int fieldIndex = startIndex, outputIndex = 0; fieldIndex < endIndex; fieldIndex += 2, outputIndex++) {
        overlayFFTFrame(positionX, positionY, inputFields, fieldIndex, componentFrames[outputIndex]);
    }
}

// Overlay the input and output FFT arrays, in either 2D or 3D
void TransformPal::overlayFFTArrays(const fftw_complex *fftIn, const fftw_complex *fftOut,
                                    FrameCanvas &canvas)
{
    // Colours
    const auto green = canvas.rgb(0, 0xFFFF, 0);

    // How many pixels to draw for each bin
    const qint32 xScale = 2;
    const qint32 yScale = 2;

    // Each block shows the absolute value of the real component of an FFT bin using a log scale.
    // Work out a scaling factor to make all values visible.
    double maxValue = 0;
    for (qint32 i = 0; i < xComplex * yComplex * zComplex; i++) {
        maxValue = qMax(maxValue, fabs(fftIn[i][0]));
        maxValue = qMax(maxValue, fabs(fftOut[i][0]));
    }
    const double valueScale = 65535.0 / log2(maxValue);

    // Draw each 2D plane of the array
    for (qint32 z = 0; z < zComplex; z++) {
        for (qint32 column = 0; column < 2; column++) {
            const fftw_complex *fftData = column == 0 ? fftIn : fftOut;

            // Work out where this 2D array starts
            const qint32 yStart = canvas.top() + (z * ((yScale * yComplex) + 1));
            const qint32 xStart = canvas.right() - ((2 - column) * ((xScale * xComplex) + 1)) - 1;

            // Outline the array
            canvas.drawRectangle(xStart, yStart, (xScale * xComplex) + 2, (yScale * yComplex) + 2, green);

            // Draw the bins in the array
            for (qint32 y = 0; y < yComplex; y++) {
                for (qint32 x = 0; x < xComplex; x++) {
                    const double value = fabs(fftData[(((z * yComplex) + y) * xComplex) + x][0]);
                    const double shade = value <= 0 ? 0 : log2(value) * valueScale;
                    const quint16 shade16 = static_cast<quint16>(qBound(0.0, shade, 65535.0));
                    canvas.fillRectangle(xStart + (x * xScale) + 1, yStart + (y * yScale) + 1, xScale, yScale, canvas.grey(shade16));
                }
            }
        }
    }
}
