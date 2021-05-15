/************************************************************************

    palcolour.cpp

    Performs 2D subcarrier filtering to process stand-alone fields of
    a video signal

    Copyright (C) 2018  William Andrew Steer
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2019 Adam Sampson

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

// PALcolour original copyright notice:
// Copyright (C) 2018  William Andrew Steer
// Contact the author at palcolour@techmind.org

#include "palcolour.h"

#include "transformpal2d.h"
#include "transformpal3d.h"

#include "firfilter.h"

#include "deemp.h"

#include <array>
#include <cassert>
#include <cmath>

/*!
    \class PalColour

    PALcolour, originally written by William Andrew Steer, is a line-locked PAL
    decoder using 2D FIR filters.

    For a good overview of line-locked PAL decoding techniques, see
    BBC Research Department Report 1986/02 (https://www.bbc.co.uk/rd/publications/rdreport_1986_02),
    "Colour encoding and decoding techniques for line-locked sampled PAL and
    NTSC television signals" by C.K.P. Clarke. PALcolour uses the architecture
    shown in Figure 23(c), except that it has three separate baseband filters,
    one each for Y, U and V, with different characteristics. Rather than
    tracking the colour subcarrier using a PLL, PALcolour detects the phase of
    the subcarrier at the colourburst, and rotates the U/V output to
    compensate when decoding.

    BBC Research Department Report 1988/11 (https://www.bbc.co.uk/rd/publications/rdreport_1988_11),
    "PAL decoding: Multi-dimensional filter design for chrominance-luminance
    separation", also by C.K.P. Clarke, describes the design concerns behind
    these filters. As PALcolour is a software implementation, it can use larger
    filters with more complex coefficients than the report describes.
 */

// Definitions of static constexpr data members, for compatibility with
// pre-C++17 compilers
constexpr qint32 PalColour::MAX_WIDTH;
constexpr qint32 PalColour::FILTER_SIZE;

PalColour::PalColour()
    : configurationSet(false)
{
}

qint32 PalColour::Configuration::getThresholdsSize() const
{
    if (chromaFilter == transform2DFilter) {
        return TransformPal2D::getThresholdsSize();
    } else if (chromaFilter == transform3DFilter) {
        return TransformPal3D::getThresholdsSize();
    } else {
        return 0;
    }
}

qint32 PalColour::Configuration::getLookBehind() const
{
    if (chromaFilter == transform3DFilter) {
        return TransformPal3D::getLookBehind();
    } else {
        return 0;
    }
}

qint32 PalColour::Configuration::getLookAhead() const
{
    if (chromaFilter == transform3DFilter) {
        return TransformPal3D::getLookAhead();
    } else {
        return 0;
    }
}

// Return the current configuration
const PalColour::Configuration &PalColour::getConfiguration() const {
    return configuration;
}

void PalColour::updateConfiguration(const LdDecodeMetaData::VideoParameters &_videoParameters,
                                    const Configuration &_configuration)
{
    // Copy the configuration parameters
    videoParameters = _videoParameters;
    configuration = _configuration;

    // Build the look-up tables
    buildLookUpTables();

    if (configuration.chromaFilter == transform2DFilter || configuration.chromaFilter == transform3DFilter) {
        // Create the Transform PAL filter
        if (configuration.chromaFilter == transform2DFilter) {
            transformPal.reset(new TransformPal2D);
        } else {
            transformPal.reset(new TransformPal3D);
        }

        // Configure the filter
        transformPal->updateConfiguration(videoParameters, configuration.transformMode, configuration.transformThreshold,
                                          configuration.transformThresholds);
    }

    configurationSet = true;
}

// Rebuild the lookup tables based on the configuration
void PalColour::buildLookUpTables()
{
    // Generate the reference carrier: quadrature samples of a sine wave at the
    // subcarrier frequency. We'll use this for two purposes below:
    // - product-detecting the line samples, to give us quadrature samples of
    //   the chroma information centred on 0 Hz
    // - working out what the phase of the subcarrier is on each line,
    //   so we can rotate the chroma samples to put U/V on the right axes
    if(videoParameters.fieldHeight != 263) {
        for (qint32 i = 0; i < videoParameters.fieldWidth; i++) {
            const double rad = 2 * M_PI * i * videoParameters.fsc / videoParameters.sampleRate;
            sine[i] = sin(rad);
            cosine[i] = cos(rad);
        }
    }
    else {
        // HACK - For whatever reason Pal-M ends up with the vectors swapped and out of phase
        // swapping the cos and sin references seem to work around that.
        // TODO: Find a proper solution to this.
        for (qint32 i = 0; i < videoParameters.fieldWidth; i++) {
            const double rad = 2 * M_PI * i * videoParameters.fsc / videoParameters.sampleRate;
            sine[i] = cos(rad);
            cosine[i] = sin(rad);
        }
    }

    // Create filter profiles for colour filtering.
    //
    // One can argue over merits of different filters, but I stick with simple
    // raised cosine unless there's compelling reason to do otherwise.
    // PAL-I colour bandwidth should be around 1.1 or 1.2 MHz:
    // acc to Rec.470, +1066 or -1300kHz span of colour sidebands!
    // The width of the filter window should scale with the sample rate.
    //
    // chromaBandwidthHz values between 1.1MHz and 1.3MHz can be tried. Some
    // specific values in that range may work best at minimising residual dot
    // pattern at given sample rates due to the discrete nature of the filters.
    // It'd be good to find ways to optimise this more rigourously.
    //
    // Note in principle you could have different bandwidths for extracting the
    // luma and chroma, according to aesthetic tradeoffs. Not really very
    // justifyable though. Keeping the Y and C bandwidth the same (or at least
    // similar enough for the filters to be the same size) allows them to be
    // computed together later.
    //
    // The 0.93 is a bit empirical for the 4Fsc sampled LaserDisc scans.
    const double chromaBandwidthHz = 1100000.0 / 0.93;

    // Compute filter widths based on chroma bandwidth.
    // FILTER_SIZE must be wide enough to hold both filters (and ideally no
    // wider, else we're doing more computation than we need to).
    // XXX where does the 0.5* come from?
    const double ca = 0.5 * videoParameters.sampleRate / chromaBandwidthHz;
    const double ya = 0.5 * videoParameters.sampleRate / chromaBandwidthHz;
    assert(FILTER_SIZE >= static_cast<qint32>(ca));
    assert(FILTER_SIZE >= static_cast<qint32>(ya));

    // Note that we choose to make the y-filter *much* less selective in the
    // vertical direction: this is to prevent castellation on horizontal colour
    // boundaries.
    //
    // We may wish to broaden vertical bandwidth *slightly* so as to better
    // pass one- or two-line colour bars - underlines/graphics etc.

    double cdiv = 0, ydiv = 0;
    for (qint32 f = 0; f <= FILTER_SIZE; f++) {
        // 0-2-4-6 sequence here because we're only processing one field.
        const double fc   = qMin(ca, static_cast<double>(f));
        const double ff   = qMin(ca, sqrt(f * f + 2 * 2));
        const double fff  = qMin(ca, sqrt(f * f + 4 * 4));
        const double ffff = qMin(ca, sqrt(f * f + 6 * 6));

        // We will sum the zero-th horizontal tap twice later (when b == 0 in
        // the filter loop), so halve the coefficient to compensate
        const qint32 d = (f == 0) ? 2 : 1;

        // For U/V.
        // 0, 2, 1, 3 are vertical taps 0, +/- 1, +/- 2, +/- 3 (see filter loop below).
        cfilt[f][0] = (1 + cos(M_PI * fc   / ca)) / d;
        cfilt[f][2] = (1 + cos(M_PI * ff   / ca)) / d;
        cfilt[f][1] = (1 + cos(M_PI * fff  / ca)) / d;
        cfilt[f][3] = (1 + cos(M_PI * ffff / ca)) / d;

        // Each horizontal coefficient is applied to 2 columns (when b == 0,
        // it's the same column twice).
        // The zero-th vertical coefficient is applied to 1 line, and the
        // others are applied to pairs of lines.
        cdiv += 2 * (1 * cfilt[f][0] + 2 * cfilt[f][2] + 2 * cfilt[f][1] + 2 * cfilt[f][3]);

        const double fy   = qMin(ya, static_cast<double>(f));
        const double fffy = qMin(ya, sqrt(f * f + 4 * 4));

        // For Y, only use lines n, n+/-2: the others cancel!!!
        //  *have tried* using lines +/-1 & 3 --- can be made to work, but
        //  introduces *phase-sensitivity* to the filter -> leaks too much
        //  subcarrier if *any* phase-shifts!
        // note omission of yfilt taps 1 and 3 for PAL
        //
        // Tap 2 is only used for PAL; 0.2 factor makes it much less sensitive
        // to adjacent lines and reduces castellations and residual dot
        // patterning.
        //
        // 0, 1 are vertical taps 0, +/- 2 (see filter loop below).
        yfilt[f][0] =       (1 + cos(M_PI * fy   / ya)) / d;
        yfilt[f][1] = 0.2 * (1 + cos(M_PI * fffy / ya)) / d;

        ydiv += 2 * (1 * yfilt[f][0] + 2 * 0 + 2 * yfilt[f][1] + 2 * 0);
    }

    // Normalise the filter coefficients.
    for (qint32 f = 0; f <= FILTER_SIZE; f++) {
        for (qint32 i = 0; i < 4; i++) {
            cfilt[f][i] /= cdiv;
        }
        for (qint32 i = 0; i < 2; i++) {
            yfilt[f][i] /= ydiv;
        }
    }
}

void PalColour::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                             QVector<ComponentFrame> &componentFrames)
{
    assert(configurationSet);
    assert((componentFrames.size() * 2) == (endIndex - startIndex));

    QVector<const double *> chromaData(endIndex - startIndex);
    if (configuration.chromaFilter != palColourFilter) {
        // Use Transform PAL filter to extract chroma
        transformPal->filterFields(inputFields, startIndex, endIndex, chromaData);
    }

    for (qint32 i = startIndex, j = 0, k = 0; i < endIndex; i += 2, j += 2, k++) {
        // Initialise and clear the component frame
        componentFrames[k].init(videoParameters);

        decodeField(inputFields[i], chromaData[j], componentFrames[k]);
        decodeField(inputFields[i + 1], chromaData[j + 1], componentFrames[k]);
    }

    if (configuration.showFFTs && configuration.chromaFilter != palColourFilter) {
        // Overlay the FFT visualisation
        transformPal->overlayFFT(configuration.showPositionX, configuration.showPositionY,
                                 inputFields, startIndex, endIndex, componentFrames);
    }
}

// Decode one field into componentFrame
void PalColour::decodeField(const SourceField &inputField, const double *chromaData, ComponentFrame &componentFrame)
{
    // Pointer to the composite signal data
    const quint16 *compPtr = inputField.data.data();

    const qint32 firstLine = inputField.getFirstActiveLine(videoParameters);
    const qint32 lastLine = inputField.getLastActiveLine(videoParameters);
    for (qint32 fieldLine = firstLine; fieldLine < lastLine; fieldLine++) {
        LineInfo line(fieldLine);

        // Detect the colourburst from the composite signal
        detectBurst(line, compPtr);

        // Rotate and scale line.bp/line.bq to apply gain and phase adjustment
        const double oldBp = line.bp, oldBq = line.bq;
        const double theta = (configuration.chromaPhase * M_PI) / 180;
        line.bp = (oldBp * cos(theta) - oldBq * sin(theta)) * configuration.chromaGain;
        line.bq = (oldBp * sin(theta) + oldBq * cos(theta)) * configuration.chromaGain;

        if (configuration.chromaFilter == palColourFilter) {
            // Decode chroma and luma from the composite signal
            decodeLine<quint16, false>(inputField, compPtr, line, componentFrame);
        } else {
            // Decode chroma and luma from the Transform PAL output
            decodeLine<double, true>(inputField, chromaData, line, componentFrame);
        }
    }
}

PalColour::LineInfo::LineInfo(qint32 _number)
    : number(_number)
{
}

// Detect the colourburst on a line.
// Stores the burst details into line.
void PalColour::detectBurst(LineInfo &line, const quint16 *inputData)
{
    // Dummy black line, used when the filter needs to look outside the field.
    static constexpr quint16 blackLine[MAX_WIDTH] = {0};

    // Get pointers to the surrounding lines of input data.
    // If a line we need is outside the field, use blackLine instead.
    // (Unlike below, we don't need to stay in the active area, since we're
    // only looking at the colourburst.)
    const quint16 *in0, *in1, *in2, *in3, *in4;
    in0 =                                                                 inputData +  (line.number      * videoParameters.fieldWidth);
    in1 = (line.number - 1) <  0                           ? blackLine : (inputData + ((line.number - 1) * videoParameters.fieldWidth));
    in2 = (line.number + 1) >= videoParameters.fieldHeight ? blackLine : (inputData + ((line.number + 1) * videoParameters.fieldWidth));
    in3 = (line.number - 2) <  0                           ? blackLine : (inputData + ((line.number - 2) * videoParameters.fieldWidth));
    in4 = (line.number + 2) >= videoParameters.fieldHeight ? blackLine : (inputData + ((line.number + 2) * videoParameters.fieldWidth));

    // Find absolute burst phase relative to the reference carrier by
    // product detection.
    //
    // To avoid hue-shifts on alternate lines, the phase is determined by
    // averaging the phase on the current-line with the average of two
    // other lines, one above and one below the current line.
    //
    // For PAL we use the next-but-one line above and below (in the field),
    // which will have the same V-switch phase as the current-line (and 180
    // degree change of phase), and we also analyse the average (bpo/bqo
    // 'old') of the line immediately above and below, which have the
    // opposite V-switch phase (and a 90 degree subcarrier phase shift).
    double bp = 0, bq = 0, bpo = 0, bqo = 0;
    for (qint32 i = videoParameters.colourBurstStart; i < videoParameters.colourBurstEnd; i++) {
        bp += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * sine[i];
        bq += ((in0[i] - ((in3[i] + in4[i]) / 2.0)) / 2.0) * cosine[i];
        bpo += ((in2[i] - in1[i]) / 2.0) * sine[i];
        bqo += ((in2[i] - in1[i]) / 2.0) * cosine[i];
    }

    // Normalise the sums above
    const qint32 colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;
    bp /= colourBurstLength;
    bq /= colourBurstLength;
    bpo /= colourBurstLength;
    bqo /= colourBurstLength;

    // Detect the V-switch state on this line.
    //
    // I forget exactly why this works, but it's essentially comparing the
    // vector magnitude /difference/ between the phases of the burst on the
    // present line and previous line to the magnitude of the burst. This
    // may effectively be a dot-product operation...
    line.Vsw = -1;
    if ((((bp - bpo) * (bp - bpo) + (bq - bqo) * (bq - bqo)) < (bp * bp + bq * bq) * 2)) {
        line.Vsw = 1;
    }

    // Average the burst phase to get -U (reference) phase out -- burst
    // phase is (-U +/-V). bp and bq will be of the order of 1000.
    line.bp = (bp - bqo) / 2;
    line.bq = (bq + bpo) / 2;

    // Normalise the magnitude of the bp/bq vector to 1.
    // Kill colour if burst too weak.
    // XXX magic number 130000 !!! check!
    const double burstNorm = qMax(sqrt(line.bp * line.bp + line.bq * line.bq), 130000.0 / 128);
    line.bp /= burstNorm;
    line.bq /= burstNorm;
}

// Perform analog-style noise coring.
void PalColour::doYNR(double *Yline)
{
    // nr_y is the coring level
    const double irescale = (videoParameters.white16bIre - videoParameters.black16bIre) / 100;
    double nr_y = configuration.yNRLevel * irescale;

    // High-pass filter for Y
    auto yFilter(f_nrpal);

    // Filter delay (since it's a symmetric FIR filter)
    const qint32 delay = c_nrpal_b.size() / 2;

    // High-pass result
    double hpY[videoParameters.activeVideoEnd + delay];

    // Feed zeros into the filter outside the active area
    for (qint32 h = videoParameters.activeVideoStart - delay; h < videoParameters.activeVideoStart; h++) {
        yFilter.feed(0.0);
    }
    for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
        hpY[h] = yFilter.feed(Yline[h]);
    }
    for (qint32 h = videoParameters.activeVideoEnd; h < videoParameters.activeVideoEnd + delay; h++) {
        hpY[h] = yFilter.feed(0.0);
    }

    for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
        // Offset to cover the filter delay
        double a = hpY[h + delay];

        // Clip the filter strength
        if (fabs(a) > nr_y) {
            a = (a > 0) ? nr_y : -nr_y;
        }

        Yline[h] -= a;
    }
}

// Decode one line into componentFrame.
// chromaData (templated, so it can be any numeric type) is the input to
// the chroma demodulator; this may be the composite signal from
// inputField, or it may be pre-filtered down to chroma.
template <typename ChromaSample, bool PREFILTERED_CHROMA>
void PalColour::decodeLine(const SourceField &inputField, const ChromaSample *chromaData, const LineInfo &line,
                           ComponentFrame &componentFrame)
{
    // Dummy black line, used when the filter needs to look outside the active region.
    static constexpr ChromaSample blackLine[MAX_WIDTH] = {0};

    // Get pointers to the surrounding lines of input data.
    // If a line we need is outside the active area, use blackLine instead.
    const qint32 firstLine = inputField.getFirstActiveLine(videoParameters);
    const qint32 lastLine = inputField.getLastActiveLine(videoParameters);
    const ChromaSample *in0, *in1, *in2, *in3, *in4, *in5, *in6;
    in0 =                                               chromaData +  (line.number      * videoParameters.fieldWidth);
    in1 = (line.number - 1) <  firstLine ? blackLine : (chromaData + ((line.number - 1) * videoParameters.fieldWidth));
    in2 = (line.number + 1) >= lastLine  ? blackLine : (chromaData + ((line.number + 1) * videoParameters.fieldWidth));
    in3 = (line.number - 2) <  firstLine ? blackLine : (chromaData + ((line.number - 2) * videoParameters.fieldWidth));
    in4 = (line.number + 2) >= lastLine  ? blackLine : (chromaData + ((line.number + 2) * videoParameters.fieldWidth));
    in5 = (line.number - 2) <  firstLine ? blackLine : (chromaData + ((line.number - 3) * videoParameters.fieldWidth));
    in6 = (line.number + 3) >= lastLine  ? blackLine : (chromaData + ((line.number + 3) * videoParameters.fieldWidth));

    double pu[MAX_WIDTH], qu[MAX_WIDTH], pv[MAX_WIDTH], qv[MAX_WIDTH], py[MAX_WIDTH], qy[MAX_WIDTH];
    if (PREFILTERED_CHROMA && configuration.simplePAL) {
        // Use Simple PAL 1D filter.
        // (Only for Transform PAL mode, since we don't have a 1D notch filter.)

        // LPF equivalent to the BBC Transform PAL implementation's UV postfilter.
        // Generated by: sps.remez(17, [0.0, 2.15e6, 4.6e6, rate/2], [1.0, 0.0], [1.0, 1.0], fs=rate)
        static constexpr std::array<double, 17> uvFilterCoeffs {
            -0.00199265, 0.01226292, 0.01767698, -0.01034077, -0.05538487, -0.03793064,
            0.09913768, 0.29007115, 0.38112572, 0.29007115, 0.09913768, -0.03793064,
            -0.05538487, -0.01034077, 0.01767698, 0.01226292, -0.00199265
        };
        static constexpr auto uvFilter = makeFIRFilter(uvFilterCoeffs);

        const qint32 overlap = uvFilterCoeffs.size() / 2;
        const qint32 startPos = videoParameters.activeVideoStart - overlap;
        const qint32 endPos = videoParameters.activeVideoEnd + overlap + 1;

        // Multiply the composite input signal by the reference carrier, giving
        // quadrature samples where the colour subcarrier is now at 0 Hz
        double m[MAX_WIDTH], n[MAX_WIDTH];
        for (qint32 i = startPos; i < endPos; i++) {
            m[i] = in0[i] * sine[i];
            n[i] = in0[i] * cosine[i];
        }

        // Apply the filter to U, and copy the result to V
        uvFilter.apply(&m[startPos], &pu[startPos], endPos - startPos);
        uvFilter.apply(&n[startPos], &qu[startPos], endPos - startPos);
        for (qint32 i = videoParameters.activeVideoStart; i < videoParameters.activeVideoEnd; i++) {
            pv[i] = pu[i];
            qv[i] = qu[i];
        }
    } else {
        // Use PALcolour's 2D filter

        // Multiply the composite input signal by the reference carrier, giving
        // quadrature samples where the colour subcarrier is now at 0 Hz.
        // There will be a considerable amount of energy at higher frequencies
        // resulting from the luma information and aliases of the signal, so
        // we need to low-pass filter it before extracting the colour
        // components.
        //
        // After filtering -- i.e. removing all the terms with sin(i) and sin^2(i)
        // from the product -- we'll be left with just the chroma signal, at half
        // its original amplitude. Phase errors will cancel between lines with
        // opposite Vsw sense, giving correct phase (hue) but lower amplitude
        // (saturation).
        //
        // As the 2D filters are vertically symmetrical, we can pre-compute the
        // sums of pairs of lines above and below line.number to save some work
        // in the inner loop below.
        //
        // Vertical taps 1 and 2 are swapped in the array to save one addition
        // in the filter loop, as U and V use the same sign for taps 0 and 2.
        double m[4][MAX_WIDTH], n[4][MAX_WIDTH];
        for (qint32 i = videoParameters.activeVideoStart - FILTER_SIZE; i < videoParameters.activeVideoEnd + FILTER_SIZE + 1; i++) {
            m[0][i] =  in0[i] * sine[i];
            m[2][i] =  in1[i] * sine[i] - in2[i] * sine[i];
            m[1][i] = -in3[i] * sine[i] - in4[i] * sine[i];
            m[3][i] = -in5[i] * sine[i] + in6[i] * sine[i];

            n[0][i] =  in0[i] * cosine[i];
            n[2][i] =  in1[i] * cosine[i] - in2[i] * cosine[i];
            n[1][i] = -in3[i] * cosine[i] - in4[i] * cosine[i];
            n[3][i] = -in5[i] * cosine[i] + in6[i] * cosine[i];
        }

        // p & q should be sine/cosine components' amplitudes
        // NB: Multiline averaging/filtering assumes perfect
        //     inter-line phase registration...

        for (qint32 i = videoParameters.activeVideoStart; i < videoParameters.activeVideoEnd; i++) {
            double PU = 0, QU = 0, PV = 0, QV = 0, PY = 0, QY = 0;

            // Carry out 2D filtering. P and Q are the two arbitrary SINE & COS
            // phases components. U filters for U, V for V, and Y for Y.
            //
            // U and V are the same for lines n ([0]), n+/-2 ([1]), but
            // differ in sign for n+/-1 ([2]), n+/-3 ([3]) owing to the
            // forward/backward axis slant.

            for (qint32 b = 0; b <= FILTER_SIZE; b++) {
                const qint32 l = i - b;
                const qint32 r = i + b;

                PY += (m[0][r] + m[0][l]) * yfilt[b][0] + (m[1][r] + m[1][l]) * yfilt[b][1];
                QY += (n[0][r] + n[0][l]) * yfilt[b][0] + (n[1][r] + n[1][l]) * yfilt[b][1];

                PU += (m[0][r] + m[0][l]) * cfilt[b][0] + (m[1][r] + m[1][l]) * cfilt[b][1]
                        + (n[2][r] + n[2][l]) * cfilt[b][2] + (n[3][r] + n[3][l]) * cfilt[b][3];
                QU += (n[0][r] + n[0][l]) * cfilt[b][0] + (n[1][r] + n[1][l]) * cfilt[b][1]
                        - (m[2][r] + m[2][l]) * cfilt[b][2] - (m[3][r] + m[3][l]) * cfilt[b][3];
                PV += (m[0][r] + m[0][l]) * cfilt[b][0] + (m[1][r] + m[1][l]) * cfilt[b][1]
                        - (n[2][r] + n[2][l]) * cfilt[b][2] - (n[3][r] + n[3][l]) * cfilt[b][3];
                QV += (n[0][r] + n[0][l]) * cfilt[b][0] + (n[1][r] + n[1][l]) * cfilt[b][1]
                        + (m[2][r] + m[2][l]) * cfilt[b][2] + (m[3][r] + m[3][l]) * cfilt[b][3];
            }

            pu[i] = PU;
            qu[i] = QU;
            pv[i] = PV;
            qv[i] = QV;
            py[i] = PY;
            qy[i] = QY;
        }
    }

    // Pointer to composite signal data
    const quint16 *comp = inputField.data.data() + (line.number * videoParameters.fieldWidth);

    // Pointers to component output
    const qint32 lineNumber = (line.number * 2) + inputField.getOffset();
    double *outY = componentFrame.y(lineNumber);
    double *outU = componentFrame.u(lineNumber);
    double *outV = componentFrame.v(lineNumber);

    for (qint32 i = videoParameters.activeVideoStart; i < videoParameters.activeVideoEnd; i++) {
        // Compute luma by...
        if (PREFILTERED_CHROMA) {
            // ... subtracting pre-filtered chroma from the composite input
            outY[i] = comp[i] - in0[i];
        } else {
            // ... resynthesising the chroma signal that the Y filter
            // extracted (at half amplitude), and subtracting it from the
            // composite input
            outY[i] = comp[i] - ((py[i] * sine[i] + qy[i] * cosine[i]) * 2.0);
        }

        // Rotate the p&q components (at the arbitrary sine/cosine
        // reference phase) backwards by the burst phase (relative to the
        // reference phase), in order to recover U and V. The Vswitch is
        // applied to flip the V-phase on alternate lines for PAL.
        // The result is doubled because the filter extracts the chroma signal
        // at half amplitude.
        outU[i] =            -(pu[i] * line.bp + qu[i] * line.bq) * 2.0;
        outV[i] = line.Vsw * -(qv[i] * line.bp - pv[i] * line.bq) * 2.0;
    }

    if (configuration.yNRLevel > 0.0) {
        doYNR(outY);
    }
}
