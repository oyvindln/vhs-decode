/************************************************************************

    transformpal2d.h

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

#ifndef TRANSFORMPAL2D_H
#define TRANSFORMPAL2D_H

#include <QVector>
#include <fftw3.h>

#include "componentframe.h"
#include "outputwriter.h"
#include "sourcefield.h"
#include "transformpal.h"

class TransformPal2D : public TransformPal {
public:
    TransformPal2D();
    virtual ~TransformPal2D();

    // Return the expected size of the thresholds array.
    static qint32 getThresholdsSize();

    void filterFields(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                      QVector<const double *> &outputFields) override;

protected:
    void filterField(const SourceField& inputField, qint32 outputIndex);
    void forwardFFTTile(qint32 tileX, qint32 tileY, qint32 startY, qint32 endY, const SourceField &inputField);
    void inverseFFTTile(qint32 tileX, qint32 tileY, qint32 startY, qint32 endY, qint32 outputIndex);
    template <TransformMode MODE>
    void applyFilter();
    void overlayFFTFrame(qint32 positionX, qint32 positionY,
                         const QVector<SourceField> &inputFields, qint32 fieldIndex,
                         ComponentFrame &componentFrame) override;

    // FFT input and output sizes.
    // The input field is divided into tiles of XTILE x YTILE, with adjacent
    // tiles overlapping by HALFXTILE/HALFYTILE.
    static constexpr qint32 YTILE = 16;
    static constexpr qint32 HALFYTILE = YTILE / 2;
    static constexpr qint32 XTILE = 32;
    static constexpr qint32 HALFXTILE = XTILE / 2;

    // Each tile is converted to the frequency domain using forwardPlan, which
    // gives a complex result of size XCOMPLEX x YCOMPLEX (roughly half the
    // size of the input, because the input data was real, i.e. contained no
    // negative frequencies).
    static constexpr qint32 YCOMPLEX = YTILE;
    static constexpr qint32 XCOMPLEX = (XTILE / 2) + 1;

    // Window function applied before the FFT
    double windowFunction[YTILE][XTILE];

    // FFT input/output buffers
    double *fftReal;
    fftw_complex *fftComplexIn;
    fftw_complex *fftComplexOut;

    // FFT plans
    fftw_plan forwardPlan, inversePlan;

    // The combined result of all the FFT processing for each input field.
    // Inverse-FFT results are accumulated into these buffers.
    QVector<QVector<double>> chromaBuf;
};

#endif
