/************************************************************************

    comb.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018 Chad Page
    Copyright (C) 2018-2019 Simon Inns
    Copyright (C) 2020 Adam Sampson

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

#include "comb.h"

#include "deemp.h"

#include <QScopedPointer>

// Definitions of static constexpr data members, for compatibility with
// pre-C++17 compilers
constexpr qint32 Comb::MAX_WIDTH;
constexpr qint32 Comb::MAX_HEIGHT;

// Indexes for the candidates considered in 3D adaptive mode
enum CandidateIndex : qint32 {
    CAND_LEFT,
    CAND_RIGHT,
    CAND_UP,
    CAND_DOWN,
    CAND_PREV_FIELD,
    CAND_NEXT_FIELD,
    CAND_PREV_FRAME,
    CAND_NEXT_FRAME,
    NUM_CANDIDATES
};

// Map colours for the candidates
static constexpr quint32 CANDIDATE_SHADES[] = {
    0xFF8080, // CAND_LEFT - red
    0xFF8080, // CAND_RIGHT - red
    0xFFFF80, // CAND_UP - yellow
    0xFFFF80, // CAND_DOWN - yellow
    0x80FF80, // CAND_PREV_FIELD - green
    0x80FF80, // CAND_NEXT_FIELD - green
    0x8080FF, // CAND_PREV_FRAME - blue
    0xFF80FF, // CAND_NEXT_FRAME - purple
};

static constexpr std::array<double, 4> sin4fsc_data = {1, 0, -1, 0};

constexpr double sin4fsc(const std::size_t i) {
    return sin4fsc_data[i % 4];
}

constexpr double cos4fsc(const std::size_t i) {
    // cos(i) is just sin(i + pi/2) and we are at 4 fsc.
    return sin4fsc(i + 1);
}

// Public methods -----------------------------------------------------------------------------------------------------

Comb::Comb()
    : configurationSet(false)
{
}

qint32 Comb::Configuration::getLookBehind() const {
    if (dimensions == 3) {
        // In 3D mode, we need to see the previous frame
        return 1;
    }

    return 0;
}

qint32 Comb::Configuration::getLookAhead() const {
    if (dimensions == 3) {
        // ... and also the next frame
        return 1;
    }

    return 0;
}

// Return the current configuration
const Comb::Configuration &Comb::getConfiguration() const {
    return configuration;
}

// Set the comb filter configuration parameters
void Comb::updateConfiguration(const LdDecodeMetaData::VideoParameters &_videoParameters, const Comb::Configuration &_configuration)
{
    // Copy the configuration parameters
    videoParameters = _videoParameters;
    configuration = _configuration;

    // Range check the frame dimensions
    if (videoParameters.fieldWidth > MAX_WIDTH) qCritical() << "Comb::Comb(): Frame width exceeds allowed maximum!";
    if (((videoParameters.fieldHeight * 2) - 1) > MAX_HEIGHT) qCritical() << "Comb::Comb(): Frame height exceeds allowed maximum!";

    // Range check the video start
    if (videoParameters.activeVideoStart < 16) qCritical() << "Comb::Comb(): activeVideoStart must be > 16!";

    if (videoParameters.sampleRate / videoParameters.fsc != 4)
    {
        // Decoder assumes 4fsc sample rate at the moment.
        qCritical() << "Data is not in 4fsc sample rate, color decoding will not work properly!";
    }

    configurationSet = true;
}

void Comb::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                        QVector<OutputFrame> &outputFrames)
{
    assert(configurationSet);
    assert((outputFrames.size() * 2) == (endIndex - startIndex));

    // Buffers for the next, current and previous frame.
    // Because we only need three of these, we allocate them upfront then
    // rotate the pointers below.
    QScopedPointer<FrameBuffer> nextFrameBuffer, currentFrameBuffer, previousFrameBuffer;
    nextFrameBuffer.reset(new FrameBuffer(videoParameters, configuration));
    currentFrameBuffer.reset(new FrameBuffer(videoParameters, configuration));
    previousFrameBuffer.reset(new FrameBuffer(videoParameters, configuration));

    // Decode each pair of fields into a frame.
    // To support 3D operation, where we need to see three input frames at a time,
    // each iteration of the loop loads and 1D/2D-filters frame N + 1, then
    // 3D-filters and outputs frame N.
    const qint32 preStartIndex = (configuration.dimensions == 3) ? startIndex - 4 : startIndex - 2;
    for (qint32 fieldIndex = preStartIndex; fieldIndex < endIndex; fieldIndex += 2) {
        const qint32 frameIndex = (fieldIndex - startIndex) / 2;

        // Rotate the buffers
        {
            QScopedPointer<FrameBuffer> recycle(previousFrameBuffer.take());
            previousFrameBuffer.reset(currentFrameBuffer.take());
            currentFrameBuffer.reset(nextFrameBuffer.take());
            nextFrameBuffer.reset(recycle.take());
        }

        // If there's another input field, bring it into nextFrameBuffer
        if (fieldIndex + 3 < inputFields.size()) {
            // Load fields into the buffer
            nextFrameBuffer->loadFields(inputFields[fieldIndex + 2], inputFields[fieldIndex + 3]);

            // Extract chroma using 1D filter
            nextFrameBuffer->split1D();

            // Extract chroma using 2D filter
            nextFrameBuffer->split2D();
        }

        if (fieldIndex < startIndex) {
            // This is a look-behind frame; no further decoding needed.
            continue;
        }

        if (configuration.dimensions == 3) {
            // Extract chroma using 3D filter
            currentFrameBuffer->split3D(*previousFrameBuffer, *nextFrameBuffer);
        }

        // Demodulate chroma giving I/Q
        if (configuration.phaseCompensation) {
            currentFrameBuffer->splitIQlocked();
            currentFrameBuffer->filterIQFull();
        } else {
            currentFrameBuffer->splitIQ();
            // Extract Y from baseband and I/Q
            currentFrameBuffer->adjustY();
            // Post-filter I/Q
            if (configuration.colorlpf) currentFrameBuffer->filterIQ();
        }

        // Apply noise reduction
        currentFrameBuffer->doYNR();
        currentFrameBuffer->doCNR();

        // Convert the YIQ result to RGB or YCbCr
        outputFrames[frameIndex] = configuration.outputYCbCr ? currentFrameBuffer->yiqToYUVFrame() :
                                                               currentFrameBuffer->yiqToRGBFrame();

        // Overlay the map if required
        if (configuration.dimensions == 3 && configuration.showMap) {
            currentFrameBuffer->overlayMap(*previousFrameBuffer, *nextFrameBuffer, outputFrames[frameIndex]);
        }
    }
}

// Private methods ----------------------------------------------------------------------------------------------------

Comb::FrameBuffer::FrameBuffer(const LdDecodeMetaData::VideoParameters &videoParameters_,
                               const Configuration &configuration_)
    : videoParameters(videoParameters_), configuration(configuration_)
{
    // Set the frame height
    frameHeight = ((videoParameters.fieldHeight * 2) - 1);

    // Set the IRE scale
    irescale = (videoParameters.white16bIre - videoParameters.black16bIre) / 100;
}

/*
 * The color burst frequency is 227.5 cycles per line, so it flips 180 degrees for each line.
 *
 * The color burst *signal* is at 180 degrees, which is a greenish yellow.
 *
 * When SCH phase is 0 (properly aligned) the color burst is in phase with the leading edge of the HSYNC pulse.
 *
 * Per RS-170 note 6, Fields 1 and 4 have positive/rising burst phase at that point on even (1-based!) lines.
 * The color burst signal should begin exactly 19 cycles later.
 *
 * getLinePhase returns true if the color burst is rising at the leading edge.
 */

inline qint32 Comb::FrameBuffer::getFieldID(qint32 lineNumber) const
{
    bool isFirstField = ((lineNumber % 2) == 0);

    return isFirstField ? firstFieldPhaseID : secondFieldPhaseID;
}

// NOTE:  lineNumber is presumed to be starting at 1.  (This lines up with how splitIQ calls it)
inline bool Comb::FrameBuffer::getLinePhase(qint32 lineNumber) const
{
    qint32 fieldID = getFieldID(lineNumber);
    bool isPositivePhaseOnEvenLines = (fieldID == 1) || (fieldID == 4);

    int fieldLine = (lineNumber / 2);
    bool isEvenLine = (fieldLine % 2) == 0;

    return isEvenLine ? isPositivePhaseOnEvenLines : !isPositivePhaseOnEvenLines;
}

// Interlace two source fields into the framebuffer.
void Comb::FrameBuffer::loadFields(const SourceField &firstField, const SourceField &secondField)
{
    // Interlace the input fields and place in the frame buffer
    qint32 fieldLine = 0;
    rawbuffer.clear();
    for (qint32 frameLine = 0; frameLine < frameHeight; frameLine += 2) {
        rawbuffer.append(firstField.data.mid(fieldLine * videoParameters.fieldWidth, videoParameters.fieldWidth));
        rawbuffer.append(secondField.data.mid(fieldLine * videoParameters.fieldWidth, videoParameters.fieldWidth));
        fieldLine++;
    }

    // Set the phase IDs for the frame
    firstFieldPhaseID = firstField.field.fieldPhaseID;
    secondFieldPhaseID = secondField.field.fieldPhaseID;
}

// Extract chroma into clpbuffer[0] using a 1D bandpass filter.
//
// The filter is [-0.25, 0, 0.5, 0, -0.25], a gentle bandpass centred on fSC.
// So the output will contain all of the chroma signal, but also whatever luma
// components ended up in the same frequency range.
//
// This also acts as an alias removal pre-filter for the quadrature detector in
// splitIQ, so we use its result for split2D rather than the raw signal.
void Comb::FrameBuffer::split1D()
{
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line's data
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double tc1 = (line[h] - ((line[h - 2] + line[h + 2]) / 2.0)) / 2.0;

            // Record the 1D C value
            clpbuffer[0].pixel[lineNumber][h] = tc1;
        }
    }
}

// Extract chroma into clpbuffer[1] using a 2D 3-line adaptive filter.
//
// Because the phase of the chroma signal changes by 180 degrees from line to
// line, subtracting two adjacent lines that contain the same information will
// give you just the chroma signal. But real images don't necessarily contain
// the same information on every line.
//
// The "3-line adaptive" part means that we look at both surrounding lines to
// estimate how similar they are to this one. We can then compute the 2D chroma
// value as a blend of the two differences, weighted by similarity.
void Comb::FrameBuffer::split2D()
{
    // Dummy black line
    static constexpr double blackLine[MAX_WIDTH] = {0};

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get pointers to the surrounding lines of 1D chroma.
        // If a line we need is outside the active area, use blackLine instead.
        const double *previousLine = blackLine;
        if (lineNumber - 2 >= videoParameters.firstActiveFrameLine) {
            previousLine = clpbuffer[0].pixel[lineNumber - 2];
        }
        const double *currentLine = clpbuffer[0].pixel[lineNumber];
        const double *nextLine = blackLine;
        if (lineNumber + 2 < videoParameters.lastActiveFrameLine) {
            nextLine = clpbuffer[0].pixel[lineNumber + 2];
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double kp, kn;

            // Summing the differences of the *absolute* values of the 1D chroma samples
            // will give us a low value if the two lines are nearly in phase (strong Y)
            // or nearly 180 degrees out of phase (strong C) -- i.e. the two cases where
            // the 2D filter is probably usable. Also give a small bonus if
            // there's a large signal (we think).
            kp  = fabs(fabs(currentLine[h]) - fabs(previousLine[h]));
            kp += fabs(fabs(currentLine[h - 1]) - fabs(previousLine[h - 1]));
            kp -= (fabs(currentLine[h]) + fabs(previousLine[h - 1])) * .10;
            kn  = fabs(fabs(currentLine[h]) - fabs(nextLine[h]));
            kn += fabs(fabs(currentLine[h - 1]) - fabs(nextLine[h - 1]));
            kn -= (fabs(currentLine[h]) + fabs(nextLine[h - 1])) * .10;

            // Map the difference into a weighting 0-1.
            // 1 means in phase or unknown; 0 means out of phase (more than kRange difference).
            const double kRange = 45 * irescale;
            kp = qBound(0.0, 1 - (kp / kRange), 1.0);
            kn = qBound(0.0, 1 - (kn / kRange), 1.0);

            double sc = 1.0;

            if ((kn > 0) || (kp > 0)) {
                // At least one of the next/previous lines has a good phase relationship.

                // If one of them is much better than the other, only use that one
                if (kn > (3 * kp)) kp = 0;
                else if (kp > (3 * kn)) kn = 0;

                sc = (2.0 / (kn + kp));
                if (sc < 1.0) sc = 1.0;
            } else {
                // Neither line has a good phase relationship.

                // But are they similar to each other? If so, we can use both of them!
                if ((fabs(fabs(previousLine[h]) - fabs(nextLine[h])) - fabs((nextLine[h] + previousLine[h]) * .2)) <= 0) {
                    kn = kp = 1;
                }

                // Else kn = kp = 0, so we won't extract any chroma for this sample.
                // (Some NTSC decoders fall back to the 1D chroma in this situation.)
            }

            // Compute the weighted sum of differences, giving the 2D chroma value
            double tc1;
            tc1  = ((currentLine[h] - previousLine[h]) * kp * sc);
            tc1 += ((currentLine[h] - nextLine[h]) * kn * sc);
            tc1 /= 4;

            clpbuffer[1].pixel[lineNumber][h] = tc1;
        }
    }
}

// Extract chroma into clpbuffer[2] using an adaptive 3D filter.
//
// For each sample, this builds a list of candidates from other positions that
// should have a 180 degree phase relationship to the current sample, and look
// like they have similar luma/chroma content. It then picks the most similar
// candidate.
void Comb::FrameBuffer::split3D(const FrameBuffer &previousFrame, const FrameBuffer &nextFrame)
{
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            // Select the best candidate
            qint32 bestIndex;
            double bestSample;
            getBestCandidate(lineNumber, h, previousFrame, nextFrame, bestIndex, bestSample);

            if (bestIndex < CAND_PREV_FIELD) {
                // A 1D or 2D candidate was best.
                // Use split2D's output, to save duplicating the line-blending heuristics here.
                clpbuffer[2].pixel[lineNumber][h] = clpbuffer[1].pixel[lineNumber][h];
            } else {
                // Compute a 3D result.
                // This sample is Y + C; the candidate is (ideally) Y - C. So compute C as ((Y + C) - (Y - C)) / 2.
                clpbuffer[2].pixel[lineNumber][h] = (clpbuffer[0].pixel[lineNumber][h] - bestSample) / 2;
            }
        }
    }
}

// Evaluate all candidates for 3D decoding for a given position, and return the best one
void Comb::FrameBuffer::getBestCandidate(qint32 lineNumber, qint32 h,
                                         const FrameBuffer &previousFrame, const FrameBuffer &nextFrame,
                                         qint32 &bestIndex, double &bestSample) const
{
    Candidate candidates[8];

    // Bias the comparison so that we prefer 3D results, then 2D, then 1D
    static constexpr double LINE_BONUS = -2.0;
    static constexpr double FIELD_BONUS = LINE_BONUS - 2.0;
    static constexpr double FRAME_BONUS = FIELD_BONUS - 2.0;

    // 1D: Same line, 2 samples left and right
    candidates[CAND_LEFT]  = getCandidate(lineNumber, h, *this, lineNumber, h - 2, 0);
    candidates[CAND_RIGHT] = getCandidate(lineNumber, h, *this, lineNumber, h + 2, 0);

    // 2D: Same field, 1 line up and down
    candidates[CAND_UP]   = getCandidate(lineNumber, h, *this, lineNumber - 2, h, LINE_BONUS);
    candidates[CAND_DOWN] = getCandidate(lineNumber, h, *this, lineNumber + 2, h, LINE_BONUS);

    // Immediately adjacent lines in previous/next field
    if (getLinePhase(lineNumber) == getLinePhase(lineNumber - 1)) {
        candidates[CAND_PREV_FIELD] = getCandidate(lineNumber, h, previousFrame, lineNumber - 1, h, FIELD_BONUS);
        candidates[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, *this, lineNumber + 1, h, FIELD_BONUS);
    } else {
        candidates[CAND_PREV_FIELD] = getCandidate(lineNumber, h, *this, lineNumber - 1, h, FIELD_BONUS);
        candidates[CAND_NEXT_FIELD] = getCandidate(lineNumber, h, nextFrame, lineNumber + 1, h, FIELD_BONUS);
    }

    // Previous/next frame, same position
    candidates[CAND_PREV_FRAME] = getCandidate(lineNumber, h, previousFrame, lineNumber, h, FRAME_BONUS);
    candidates[CAND_NEXT_FRAME] = getCandidate(lineNumber, h, nextFrame, lineNumber, h, FRAME_BONUS);

    if (configuration.adaptive) {
        // Find the candidate with the lowest penalty
        bestIndex = 0;
        for (qint32 i = 1; i < NUM_CANDIDATES; i++) {
            if (candidates[i].penalty < candidates[bestIndex].penalty) bestIndex = i;
        }
    } else {
        // Adaptive mode is disabled - do 3D against the previous frame
        bestIndex = CAND_PREV_FRAME;
    }

    bestSample = candidates[bestIndex].sample;
}

// Evaluate a candidate for 3D decoding
Comb::FrameBuffer::Candidate Comb::FrameBuffer::getCandidate(qint32 refLineNumber, qint32 refH,
                                                             const FrameBuffer &frameBuffer, qint32 lineNumber, qint32 h,
                                                             double adjustPenalty) const
{
    Candidate result;
    result.sample = frameBuffer.clpbuffer[0].pixel[lineNumber][h];

    // If the candidate is outside the active region (vertically), it's not viable
    if (lineNumber < videoParameters.firstActiveFrameLine || lineNumber >= videoParameters.lastActiveFrameLine) {
        result.penalty = 1000.0;
        return result;
    }

    // The target sample should have 180 degrees phase difference from the reference.
    // If it doesn't (e.g. because it's a blank frame or the player skipped), it's not viable.
    const qint32 wantPhase = (2 + (getLinePhase(refLineNumber) ? 2 : 0) + refH) % 4;
    const qint32 havePhase = ((frameBuffer.getLinePhase(lineNumber) ? 2 : 0) + h) % 4;
    if (wantPhase != havePhase) {
        result.penalty = 1000.0;
        return result;
    }

    // Pointers to the baseband data
    const quint16 *refLine = rawbuffer.data() + (refLineNumber * videoParameters.fieldWidth);
    const quint16 *candidateLine = frameBuffer.rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

    // Penalty based on mean luma difference in IRE over surrounding three samples
    double yPenalty = 0.0;
    for (qint32 offset = -1; offset < 2; offset++) {
        const double refC = clpbuffer[1].pixel[refLineNumber][refH + offset];
        const double refY = refLine[refH + offset] - refC;

        const double candidateC = frameBuffer.clpbuffer[1].pixel[lineNumber][h + offset];
        const double candidateY = candidateLine[h + offset] - candidateC;

        yPenalty += fabs(refY - candidateY);
    }
    yPenalty = yPenalty / 3 / irescale;

    // Penalty based on mean I/Q difference in IRE over surrounding three samples
    double iqPenalty = 0.0;
    for (qint32 offset = -1; offset < 2; offset++) {
        // The reference and candidate are 180 degrees out of phase here, so negate one
        const double refC = clpbuffer[1].pixel[refLineNumber][refH + offset];
        const double candidateC = -frameBuffer.clpbuffer[1].pixel[lineNumber][h + offset];

        // I and Q samples alternate, so weight the two channels equally
        static constexpr double weights[] = {0.5, 1.0, 0.5};
        iqPenalty += fabs(refC - candidateC) * weights[offset + 1];
    }
    // Weaken this relative to luma, to avoid spurious colour in the 2D result from showing through
    iqPenalty = (iqPenalty / 2 / irescale) * 0.28;

    result.penalty = yPenalty + iqPenalty + adjustPenalty;
    return result;
}

namespace {
    // Information about a line we're decoding.
    struct BurstInfo {
        double bsin, bcos;
    };

    BurstInfo detectBurst(const quint16* lineData,
                          const LdDecodeMetaData::VideoParameters& videoParameters)
    {
        double bsin = 0, bcos = 0;

        // Find absolute burst phase relative to the reference carrier by
        // product detection.
        // For now we just use the burst on the current line, but we could possibly do some averaging with
        // neighbouring lines later if needed.
        for (qint32 i = videoParameters.colourBurstStart; i < videoParameters.colourBurstEnd; i++) {
            bsin += lineData[i] * sin4fsc(i);
            bcos += lineData[i] * cos4fsc(i);
        }

        // Normalise the sums above
        const qint32 colourBurstLength = videoParameters.colourBurstEnd - videoParameters.colourBurstStart;
        bsin /= colourBurstLength;
        bcos /= colourBurstLength;

        const double burstNorm = qMax(sqrt(bsin * bsin + bcos * bcos), 130000.0 / 128);

        //qDebug() << "burst norm " << burstNorm;

        bsin /= burstNorm;
        bcos /= burstNorm;

        const BurstInfo info{bsin, bcos};
        return info;
    }
}

// Split I and Q, taking burst phase into account.
void Comb::FrameBuffer::splitIQlocked()
{
    // Clear the target frame YIQ buffer
    for (qint32 lineNumber = 0; lineNumber < MAX_HEIGHT; lineNumber++) {
        for (qint32 h = 0; h < MAX_WIDTH; h++) {
            yiqBuffer[lineNumber][h] = YIQ();
        }
    }

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line's data
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
        // Calculate burst phase
        const auto info = detectBurst(line, videoParameters);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            const auto val = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];

            // Demodulate the sine and cosine components.
            const auto lsin = val * sin4fsc(h) * 2;
            const auto lcos = val * cos4fsc(h) * 2;
            // Rotate the demodulated vector by the burst phase.
            const auto ti = (lsin * info.bcos - lcos * info.bsin);
            const auto tq = (lsin * info.bsin + lcos * info.bcos);
            // Rotate back 33 degrees and invert Q to get the correct I/Q vector.
            yiqBuffer[lineNumber][h].i = (ti * 0.83867056794 - tq * -0.54463903501);
            yiqBuffer[lineNumber][h].q = -(ti * -0.54463903501 + tq * 0.83867056794);
            // Subtract the split chroma part from the luma signal.
            yiqBuffer[lineNumber][h].y = line[h] - val;
        }
    }
}

// Spilt the I and Q
void Comb::FrameBuffer::splitIQ()
{
    // Clear the target frame YIQ buffer
    for (qint32 lineNumber = 0; lineNumber < MAX_HEIGHT; lineNumber++) {
        for (qint32 h = 0; h < MAX_WIDTH; h++) {
            yiqBuffer[lineNumber][h] = YIQ();
        }
    }

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line's data
        const quint16 *line = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);
        bool linePhase = getLinePhase(lineNumber);

        double si = 0, sq = 0;
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            qint32 phase = h % 4;

            double cavg = clpbuffer[configuration.dimensions - 1].pixel[lineNumber][h];

            if (linePhase) cavg = -cavg;

            switch (phase) {
                case 0: sq = cavg; break;
                case 1: si = -cavg; break;
                case 2: sq = -cavg; break;
                case 3: si = cavg; break;
                default: break;
            }

            yiqBuffer[lineNumber][h].y = line[h];
            yiqBuffer[lineNumber][h].i = si;
            yiqBuffer[lineNumber][h].q = sq;
        }
    }
}

// Filter the IQ from the input YIQ buffer
void Comb::FrameBuffer::filterIQ()
{
    auto iFilter(f_colorlpi);
    auto qFilter(configuration.colorlpf_hq ? f_colorlpi : f_colorlpq);

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        iFilter.clear();
        qFilter.clear();

        qint32 qoffset = configuration.colorlpf_hq ? f_colorlpi_offset : f_colorlpq_offset;

        double filti = 0, filtq = 0;

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            qint32 phase = h % 4;

            switch (phase) {
                case 0: filti = iFilter.feed(yiqBuffer[lineNumber][h].i); break;
                case 1: filtq = qFilter.feed(yiqBuffer[lineNumber][h].q); break;
                case 2: filti = iFilter.feed(yiqBuffer[lineNumber][h].i); break;
                case 3: filtq = qFilter.feed(yiqBuffer[lineNumber][h].q); break;
                default: break;
            }

            yiqBuffer[lineNumber][h - qoffset].i = filti;
            yiqBuffer[lineNumber][h - qoffset].q = filtq;

        }
    }
}

// Filter the full set of I and Q values from the input buffer.
void Comb::FrameBuffer::filterIQFull()
{
    auto iFilter(f_colorlpi);
    auto qFilter(configuration.colorlpf_hq ? f_colorlpi : f_colorlpq);

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        iFilter.clear();
        qFilter.clear();

        qint32 qoffset = configuration.colorlpf_hq ? f_colorlpi_offset : f_colorlpq_offset;

        double filti = 0, filtq = 0;

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            filti = iFilter.feed(yiqBuffer[lineNumber][h].i);
            filtq = qFilter.feed(yiqBuffer[lineNumber][h].q);

            yiqBuffer[lineNumber][h - qoffset].i = filti;
            yiqBuffer[lineNumber][h - qoffset].q = filtq;
        }
    }
}

// Remove the colour data from the baseband (Y)
void Comb::FrameBuffer::adjustY()
{
    // remove color data from baseband (Y)
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        bool linePhase = getLinePhase(lineNumber);

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double comp = 0;
            qint32 phase = h % 4;

            YIQ y = yiqBuffer[lineNumber][h];

            switch (phase) {
                case 0: comp = -y.q; break;
                case 1: comp = y.i; break;
                case 2: comp = y.q; break;
                case 3: comp = -y.i; break;
                default: break;
            }

            if (!linePhase) comp = -comp;
            y.y -= comp;

            yiqBuffer[lineNumber][h] = y;
        }
    }
}

/*
 * This applies an FIR coring filter to both I and Q color channels.  It's a simple (crude?) NR technique used
 * by LD players, but effective especially on the Y/luma channel.
 *
 * A coring filter removes high frequency components (.4mhz chroma, 2.8mhz luma) of a signal up to a certain point,
 * which removes small high frequency noise.
 */

void Comb::FrameBuffer::doCNR()
{
    if (configuration.cNRLevel == 0) return;

    // High-pass filters for I/Q
    auto iFilter(f_nrc);
    auto qFilter(f_nrc);

    // nr_c is the coring level
    double nr_c = configuration.cNRLevel * irescale;

    QVector<YIQ> hplinef;
    hplinef.resize(videoParameters.fieldWidth + 32);

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Filters not cleared from previous line

        for (qint32 h = videoParameters.activeVideoStart; h <= videoParameters.activeVideoEnd; h++) {
            hplinef[h].i = iFilter.feed(yiqBuffer[lineNumber][h].i);
            hplinef[h].q = qFilter.feed(yiqBuffer[lineNumber][h].q);
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            // Offset by 12 to cover the filter delay
            double ai = hplinef[h + 12].i;
            double aq = hplinef[h + 12].q;

            if (fabs(ai) > nr_c) {
                ai = (ai > 0) ? nr_c : -nr_c;
            }

            if (fabs(aq) > nr_c) {
                aq = (aq > 0) ? nr_c : -nr_c;
            }

            yiqBuffer[lineNumber][h].i -= ai;
            yiqBuffer[lineNumber][h].q -= aq;
        }
    }
}

void Comb::FrameBuffer::doYNR()
{
    if (configuration.yNRLevel == 0) return;

    // High-pass filter for Y
    auto yFilter(f_nr);

    // nr_y is the coring level
    double nr_y = configuration.yNRLevel * irescale;

    QVector<YIQ> hplinef;
    hplinef.resize(videoParameters.fieldWidth + 32);

    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Filter not cleared from previous line

        for (qint32 h = videoParameters.activeVideoStart; h <= videoParameters.activeVideoEnd; h++) {
            hplinef[h].y = yFilter.feed(yiqBuffer[lineNumber][h].y);
        }

        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            double a = hplinef[h + 12].y;

            if (fabs(a) > nr_y) {
                a = (a > 0) ? nr_y : -nr_y;
            }

            yiqBuffer[lineNumber][h].y -= a;
        }
    }
}

// Convert buffer from YIQ to RGB and store as packed RGB48
OutputFrame Comb::FrameBuffer::yiqToRGBFrame()
{
    OutputFrame outputFrame;
    outputFrame.RGB.resize(videoParameters.fieldWidth * frameHeight * 3); // for RGB 16-16-16

    // Initialise the output frame
    outputFrame.RGB.fill(0);

    // Initialise YIQ to RGB converter
    RGB rgb(videoParameters.white16bIre, videoParameters.black16bIre, configuration.whitePoint75, configuration.chromaGain);

    // Perform YIQ to RGB conversion
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line
        quint16 *linePointer = outputFrame.RGB.data() + (videoParameters.fieldWidth * 3 * lineNumber);

        // Offset the output by the activeVideoStart to keep the output frame
        // in the same x position as the input video frame
        qint32 o = (videoParameters.activeVideoStart * 3);

        // Fill the output line with the RGB values
        rgb.convertLine(&yiqBuffer[lineNumber][videoParameters.activeVideoStart],
                        &yiqBuffer[lineNumber][videoParameters.activeVideoEnd],
                        &linePointer[o]);
    }

    return outputFrame;
}

// Convert buffer from YIQ to RGB
void Comb::FrameBuffer::overlayMap(const FrameBuffer &previousFrame, const FrameBuffer &nextFrame, OutputFrame &rgbFrame)
{
    qDebug() << "Comb::FrameBuffer::overlayMap(): Overlaying map onto RGB output";

    // Overlay the map on the output RGB
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to the line
        quint16 *linePointer = rgbFrame.RGB.data() + (videoParameters.fieldWidth * 3 * lineNumber);

        const quint16 *lineData = rawbuffer.data() + (lineNumber * videoParameters.fieldWidth);

        // Fill the output frame with the RGB values
        for (qint32 h = videoParameters.activeVideoStart; h < videoParameters.activeVideoEnd; h++) {
            // Select the best candidate
            qint32 bestIndex;
            double bestSample;
            getBestCandidate(lineNumber, h, previousFrame, nextFrame, bestIndex, bestSample);

            // Take the 2D luma, and colour according to the candidate index
            const double luma = (lineData[h] - clpbuffer[1].pixel[lineNumber][h]) / 65535.0;
            const quint32 shade = CANDIDATE_SHADES[bestIndex];
            qint32 red = luma * (((shade >> 16) & 0xff) << 8);
            qint32 green = luma * (((shade >> 8) & 0xff) << 8);
            qint32 blue = luma * ((shade & 0xff) << 8);

            if (red > 65535) red = 65535;
            if (green > 65535) green = 65535;
            if (blue > 65535) blue = 65535;

            linePointer[(h * 3)] = static_cast<quint16>(red);
            linePointer[(h * 3) + 1] = static_cast<quint16>(green);
            linePointer[(h * 3) + 2] = static_cast<quint16>(blue);
        }
    }
}

// Convert buffer from YIQ to YCbCr and store as planer YUV444P16
OutputFrame Comb::FrameBuffer::yiqToYUVFrame()
{
    OutputFrame outputFrame;

    outputFrame.Y.resize(videoParameters.fieldWidth * frameHeight);
    outputFrame.Y.fill(16 * 256);

    if (configuration.chromaGain > 0) {
        outputFrame.Cb.resize(videoParameters.fieldWidth * frameHeight);
        outputFrame.Cr.resize(videoParameters.fieldWidth * frameHeight);
        outputFrame.Cb.fill(128 * 256);
        outputFrame.Cr.fill(128 * 256);
    } else {
        outputFrame.Cb.clear();
        outputFrame.Cr.clear();
    }

    // Initialise YIQ to YCbCr converter
    YCbCr ycbcr(videoParameters.white16bIre, videoParameters.black16bIre, configuration.whitePoint75, configuration.chromaGain);

    // Perform YIQ to YCbCr conversion
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        // Get a pointer to each plane
        quint16 *linePointerY = outputFrame.Y.data() + (videoParameters.fieldWidth * lineNumber);
        quint16 *linePointerCb = outputFrame.Cb.data() + (videoParameters.fieldWidth * lineNumber);
        quint16 *linePointerCr = outputFrame.Cr.data() + (videoParameters.fieldWidth * lineNumber);

        // Offset the output by the activeVideoStart to keep the output frame
        // in the same x position as the input video frame
        qint32 o = videoParameters.activeVideoStart;

        // Fill the output line with YCbCr values
        ycbcr.convertLine(&yiqBuffer[lineNumber][videoParameters.activeVideoStart],
                        &yiqBuffer[lineNumber][videoParameters.activeVideoEnd],
                        &linePointerY[o], &linePointerCb[o], &linePointerCr[o]);
    }

    return outputFrame;
}
