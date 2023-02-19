/************************************************************************

    transformpal3d.cpp

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

#include "transformpal3d.h"

#include <QtMath>
#include <cassert>
#include <cmath>
#include <cstring>

#include "framecanvas.h"

/*!
    \class TransformPal3D

    3D Transform PAL filter, based on Jim Easterbrook's implementation in
    pyctools-pal. Given a composite signal, this extracts a chroma signal from
    it using frequency-domain processing.

    For a description of the algorithm with examples, see the Transform PAL web
    site (http://www.jim-easterbrook.me.uk/pal/).
 */

// Compute one value of the window function, applied to the data blocks before
// the FFT to reduce edge effects. This is a symmetrical raised-cosine
// function, which means that the overlapping inverse-FFT blocks can be summed
// directly without needing an inverse window function.
static double computeWindow(qint32 element, qint32 limit)
{
    return 0.5 - (0.5 * cos((2 * M_PI * (element + 0.5)) / limit));
}

TransformPal3D::TransformPal3D()
    : TransformPal(XCOMPLEX, YCOMPLEX, ZCOMPLEX)
{
    // Compute the window function.
    for (qint32 z = 0; z < ZTILE; z++) {
        const double windowZ = computeWindow(z, ZTILE);
        for (qint32 y = 0; y < YTILE; y++) {
            const double windowY = computeWindow(y, YTILE);
            for (qint32 x = 0; x < XTILE; x++) {
                const double windowX = computeWindow(x, XTILE);
                windowFunction[z][y][x] = windowZ * windowY * windowX;
            }
        }
    }

    // Allocate buffers for FFTW. These must be allocated using FFTW's own
    // functions so they're properly aligned for SIMD operations.
    fftReal = fftw_alloc_real(ZTILE * YTILE * XTILE);
    fftComplexIn = fftw_alloc_complex(ZCOMPLEX * YCOMPLEX * XCOMPLEX);
    fftComplexOut = fftw_alloc_complex(ZCOMPLEX * YCOMPLEX * XCOMPLEX);

    // Plan FFTW operations
    forwardPlan = fftw_plan_dft_r2c_3d(ZTILE, YTILE, XTILE, fftReal, fftComplexIn, FFTW_MEASURE);
    inversePlan = fftw_plan_dft_c2r_3d(ZTILE, YTILE, XTILE, fftComplexOut, fftReal, FFTW_MEASURE);
}

TransformPal3D::~TransformPal3D()
{
    // Free FFTW plans and buffers
    fftw_destroy_plan(forwardPlan);
    fftw_destroy_plan(inversePlan);
    fftw_free(fftReal);
    fftw_free(fftComplexIn);
    fftw_free(fftComplexOut);
}

qint32 TransformPal3D::getThresholdsSize()
{
    // On the X axis, include only the bins we actually use in applyFilter
    return ZCOMPLEX * YCOMPLEX * ((XCOMPLEX / 4) + 1);
}

qint32 TransformPal3D::getLookBehind()
{
    // We overlap at most half a tile (in frames) into the past...
    return (HALFZTILE + 1) / 2;
}

qint32 TransformPal3D::getLookAhead()
{
    // ... and at most a tile minus one bin into the future.
    return (ZTILE - 1 + 1) / 2;
}

void TransformPal3D::filterFields(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                                  QVector<const double *> &outputFields)
{
    assert(configurationSet);

    // Check we have a valid vector of input fields, and a matching output vector
    assert((inputFields.size() % 2) == 0);
    for (qint32 i = 0; i < inputFields.size(); i++) {
        assert(!inputFields[i].data.empty());
    }
    assert(outputFields.size() == (endIndex - startIndex));

    // Check that we've been given enough surrounding fields to compute FFTs
    // that overlap the fields we're actually interested in by half a tile
    assert(startIndex >= HALFZTILE);
    assert((inputFields.size() - endIndex) >= HALFZTILE);

    // Allocate and clear output buffers
    chromaBuf.resize(endIndex - startIndex);
    for (qint32 i = 0; i < chromaBuf.size(); i++) {
        chromaBuf[i].resize(videoParameters.fieldWidth * videoParameters.fieldHeight);
        chromaBuf[i].fill(0.0);

        outputFields[i] = chromaBuf[i].data();
    }

    // Iterate through the overlapping tile positions, covering the active area.
    // (See TransformPal3D member variable documentation for how the tiling works;
    // if you change the Z tiling here, also review getLookBehind/getLookAhead above.)
    for (qint32 tileZ = startIndex - HALFZTILE; tileZ < endIndex; tileZ += HALFZTILE) {
        for (qint32 tileY = videoParameters.firstActiveFrameLine - HALFYTILE; tileY < videoParameters.lastActiveFrameLine; tileY += HALFYTILE) {
            for (qint32 tileX = videoParameters.activeVideoStart - HALFXTILE; tileX < videoParameters.activeVideoEnd; tileX += HALFXTILE) {
                // Compute the forward FFT
                forwardFFTTile(tileX, tileY, tileZ, inputFields);

                // Apply the frequency-domain filter
                applyFilter();

                // Compute the inverse FFT
                inverseFFTTile(tileX, tileY, tileZ, startIndex, endIndex);
            }
        }
    }
}

// Apply the forward FFT to an input tile, populating fftComplexIn
void TransformPal3D::forwardFFTTile(qint32 tileX, qint32 tileY, qint32 tileZ, const QVector<SourceField> &inputFields)
{
    // Work out which lines of this tile are within the active region
    const qint32 startY = qMax(videoParameters.firstActiveFrameLine - tileY, 0);
    const qint32 endY = qMin(videoParameters.lastActiveFrameLine - tileY, YTILE);

    // Copy the input signal into fftReal, applying the window function
    for (qint32 z = 0; z < ZTILE; z++) {
        const qint32 fieldIndex = tileZ + z;
        const quint16 *inputPtr = inputFields[fieldIndex].data.data();

        for (qint32 y = 0; y < YTILE; y++) {
            // If this frame line is not available in the field
            // we're reading from (either because it's above/below
            // the active region, or because it's in the other
            // field), fill it with black instead.
            if (y < startY || y >= endY || ((tileY + y) % 2) != (fieldIndex % 2)) {
                for (qint32 x = 0; x < XTILE; x++) {
                    fftReal[(((z * YTILE) + y) * XTILE) + x] = videoParameters.black16bIre * windowFunction[z][y][x];
                }
                continue;
            }

            const qint32 fieldLine = (tileY + y) / 2;
            const quint16 *b = inputPtr + (fieldLine * videoParameters.fieldWidth);
            for (qint32 x = 0; x < XTILE; x++) {
                fftReal[(((z * YTILE) + y) * XTILE) + x] = b[tileX + x] * windowFunction[z][y][x];
            }
        }
    }

    // Convert time domain in fftReal to frequency domain in fftComplexIn
    fftw_execute(forwardPlan);
}

// Apply the inverse FFT to fftComplexOut, overlaying the result into chromaBuf
void TransformPal3D::inverseFFTTile(qint32 tileX, qint32 tileY, qint32 tileZ, qint32 startIndex, qint32 endIndex)
{
    // Work out what portion of this tile is inside the active area
    const qint32 startX = qMax(videoParameters.activeVideoStart - tileX, 0);
    const qint32 endX = qMin(videoParameters.activeVideoEnd - tileX, XTILE);
    const qint32 startY = qMax(videoParameters.firstActiveFrameLine - tileY, 0);
    const qint32 endY = qMin(videoParameters.lastActiveFrameLine - tileY, YTILE);
    const qint32 startZ = qMax(startIndex - tileZ, 0);
    const qint32 endZ = qMin(endIndex - tileZ, ZTILE);

    // Convert frequency domain in fftComplexOut back to time domain in fftReal
    fftw_execute(inversePlan);

    // Overlay the result, normalising the FFTW output, into the chroma buffers
    for (qint32 z = startZ; z < endZ; z++) {
        const qint32 outputIndex = tileZ + z - startIndex;
        double *outputPtr = chromaBuf[outputIndex].data();

        for (qint32 y = startY; y < endY; y++) {
            // If this frame line is not part of this field, ignore it.
            if (((tileY + y) % 2) != (outputIndex % 2)) {
                continue;
            }

            const qint32 outputLine = (tileY + y) / 2;
            double *b = outputPtr + (outputLine * videoParameters.fieldWidth);
            for (qint32 x = startX; x < endX; x++) {
                b[tileX + x] += fftReal[(((z * YTILE) + y) * XTILE) + x] / (ZTILE * YTILE * XTILE);
            }
        }
    }
}

// Return the absolute value squared of an fftw_complex
static inline double fftwAbsSq(const fftw_complex &value)
{
    return (value[0] * value[0]) + (value[1] * value[1]);
}

// Apply the frequency-domain filter.
void TransformPal3D::applyFilter()
{
    // Get pointer to squared threshold values
    const double *thresholdsPtr = thresholds.data();

    // Clear fftComplexOut. We discard values by default; the filter only
    // copies values that look like chroma.
    for (qint32 i = 0; i < ZCOMPLEX * YCOMPLEX * XCOMPLEX; i++) {
        fftComplexOut[i][0] = 0.0;
        fftComplexOut[i][1] = 0.0;
    }

    // This is a direct translation of transform_filter from pyctools-pal, with
    // an extra loop added to extend it to 3D. The main simplification is that
    // we don't need to worry about conjugates, because FFTW only returns half
    // the result in the first place.
    //
    // The general idea is that a real modulated chroma signal will be
    // symmetrical around the U carrier, which is at fSC Hz, 72 c/aph, 18.75 Hz
    // -- and because we're sampling at 4fSC, this is handily equivalent to
    // being symmetrical around the V carrier owing to wraparound. We look at
    // every bin that might be a chroma signal, and only keep it if it's
    // sufficiently symmetrical with its reflection.
    //
    // The Z axis covers 0 to 50 Hz;      18.75 Hz is 3/8 * ZTILE.
    // The Y axis covers 0 to 576 c/aph;  72 c/aph is 1/8 * YTILE.
    // The X axis covers 0 to 4fSC Hz;    fSC HZ   is 1/4 * XTILE.

    for (qint32 z = 0; z < ZTILE; z++) {
        // Reflect around 18.75 Hz temporally.
        // XXX Why ZTILE / 4? It should be (6 * ZTILE) / 8...
        const qint32 z_ref = ((ZTILE / 4) + ZTILE - z) % ZTILE;

        for (qint32 y = 0; y < YTILE; y++) {
            // Reflect around 72 c/aph vertically.
            const qint32 y_ref = ((YTILE / 4) + YTILE - y) % YTILE;

            // Input data for this line and its reflection
            const fftw_complex *bi = fftComplexIn + (((z * YCOMPLEX) + y) * XCOMPLEX);
            const fftw_complex *bi_ref = fftComplexIn + (((z_ref * YCOMPLEX) + y_ref) * XCOMPLEX);

            // Output data for this line and its reflection
            fftw_complex *bo = fftComplexOut + (((z * YCOMPLEX) + y) * XCOMPLEX);
            fftw_complex *bo_ref = fftComplexOut + (((z_ref * YCOMPLEX) + y_ref) * XCOMPLEX);

            // We only need to look at horizontal frequencies that might be chroma (0.5fSC to 1.5fSC).
            for (qint32 x = XTILE / 8; x <= XTILE / 4; x++) {
                // Reflect around fSC horizontally
                const qint32 x_ref = (XTILE / 2) - x;

                // Get the threshold for this bin
                const double threshold_sq = *thresholdsPtr++;

                const fftw_complex &in_val = bi[x];
                const fftw_complex &ref_val = bi_ref[x_ref];

                if (x == x_ref && y == y_ref && z == z_ref) {
                    // This bin is its own reflection (i.e. it's a carrier). Keep it!
                    bo[x][0] = in_val[0];
                    bo[x][1] = in_val[1];
                    continue;
                }

                // Get the squares of the magnitudes (to minimise the number of sqrts)
                const double m_in_sq = fftwAbsSq(in_val);
                const double m_ref_sq = fftwAbsSq(ref_val);

                // Compare the magnitudes of the two values, and discard
                // both if they are more different than the threshold for
                // this bin.
                if (m_in_sq < m_ref_sq * threshold_sq || m_ref_sq < m_in_sq * threshold_sq) {
                    // Probably not a chroma signal; throw it away.
                } else {
                    // They're similar. Keep it!
                    bo[x][0] = in_val[0];
                    bo[x][1] = in_val[1];
                    bo_ref[x_ref][0] = ref_val[0];
                    bo_ref[x_ref][1] = ref_val[1];
                }
            }
        }
    }

    assert(thresholdsPtr == thresholds.data() + thresholds.size());
}

void TransformPal3D::overlayFFTFrame(qint32 positionX, qint32 positionY,
                                     const QVector<SourceField> &inputFields, qint32 fieldIndex,
                                     ComponentFrame &componentFrame)
{
    // Do nothing if the tile isn't within the frame
    if (positionX < 0 || positionX + XTILE > videoParameters.fieldWidth
        || positionY < 0 || positionY + YTILE > (2 * videoParameters.fieldHeight) + 1) {
        return;
    }

    // Compute the forward FFT
    forwardFFTTile(positionX, positionY, fieldIndex, inputFields);

    // Apply the frequency-domain filter
    applyFilter();

    // Create a canvas
    FrameCanvas canvas(componentFrame, videoParameters);

    // Outline the selected tile
    const auto green = canvas.rgb(0, 0xFFFF, 0);
    canvas.drawRectangle(positionX - 1, positionY - 1, XTILE + 1, YTILE + 1, green);

    // Draw the arrays
    overlayFFTArrays(fftComplexIn, fftComplexOut, canvas);
}
