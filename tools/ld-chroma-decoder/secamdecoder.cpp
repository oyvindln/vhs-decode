/************************************************************************

    secamdecoder.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2019-2021 Adam Sampson

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

#include "secamdecoder.h"

#include <limits>

#include "comb.h"
#include "decoderpool.h"
#include "palcolour.h"

namespace {
constexpr std::array<double, 101> dbFilterCoeffs {
    -0.01252161,  0.00060423,  0.01351862,  0.00115885, -0.01381441,
           -0.00297438,  0.01334775,  0.00459947, -0.01213053, -0.00578426,
            0.01025243,  0.00629539, -0.00787869, -0.00593937,  0.00524066,
            0.00458385, -0.00261984, -0.00217457,  0.00032629, -0.00125363,
            0.00132669,  0.00557304, -0.00204964, -0.01056728,  0.00160346,
            0.01594324,  0.00017704, -0.02135035, -0.00336464,  0.02640549,
            0.0079281 , -0.03072182, -0.01372872,  0.03393912,  0.02052545,
           -0.03575315, -0.02798863,  0.03594153,  0.03572128, -0.03438389,
           -0.04328641,  0.03107453,  0.05023822, -0.02612632, -0.05615475,
            0.01976542,  0.06066929, -0.01231711, -0.06349797,  0.00418368,
            0.06446133,  0.00418368, -0.06349797, -0.01231711,  0.06066929,
            0.01976542, -0.05615475, -0.02612632,  0.05023822,  0.03107453,
           -0.04328641, -0.03438389,  0.03572128,  0.03594153, -0.02798863,
           -0.03575315,  0.02052545,  0.03393912, -0.01372872, -0.03072182,
            0.0079281 ,  0.02640549, -0.00336464, -0.02135035,  0.00017704,
            0.01594324,  0.00160346, -0.01056728, -0.00204964,  0.00557304,
            0.00132669, -0.00125363,  0.00032629, -0.00217457, -0.00261984,
            0.00458385,  0.00524066, -0.00593937, -0.00787869,  0.00629539,
            0.01025243, -0.00578426, -0.01213053,  0.00459947,  0.01334775,
           -0.00297438, -0.01381441,  0.00115885,  0.01351862,  0.00060423,
           -0.01252161
};



constexpr std::array<double, 101> drFilterCoeffs {
    0.01111974, -0.00598478, -0.01205286,  0.00606233,  0.01257141,
           -0.00590094, -0.01260938,  0.00550563,  0.01211118, -0.00489082,
           -0.01103406,  0.00407976,  0.00935015, -0.00310368, -0.00704816,
            0.00200055,  0.00413447, -0.0008136 , -0.00063381, -0.00041039,
           -0.00341079,  0.00162299,  0.00793852, -0.00277603, -0.01287201,
            0.00382354,  0.01811906, -0.00472361, -0.02357492,  0.00544009,
            0.02912505, -0.00594406, -0.03464823,  0.00621499,  0.04001994,
           -0.00624162, -0.04511596,  0.00602241,  0.04981599, -0.0055656 ,
           -0.0540072 ,  0.00488896,  0.0575876 , -0.00401903, -0.06046914,
            0.00299008,  0.06258038, -0.00184276, -0.06386865,  0.00062247,
            0.06430169,  0.00062247, -0.06386865, -0.00184276,  0.06258038,
            0.00299008, -0.06046914, -0.00401903,  0.0575876 ,  0.00488896,
           -0.0540072 , -0.0055656 ,  0.04981599,  0.00602241, -0.04511596,
           -0.00624162,  0.04001994,  0.00621499, -0.03464823, -0.00594406,
            0.02912505,  0.00544009, -0.02357492, -0.00472361,  0.01811906,
            0.00382354, -0.01287201, -0.00277603,  0.00793852,  0.00162299,
           -0.00341079, -0.00041039, -0.00063381, -0.0008136 ,  0.00413447,
            0.00200055, -0.00704816, -0.00310368,  0.00935015,  0.00407976,
           -0.01103406, -0.00489082,  0.01211118,  0.00550563, -0.01260938,
           -0.00590094,  0.01257141,  0.00606233, -0.01205286, -0.00598478,
            0.01111974
};
}

SecamDecoder::SecamDecoder()
{
}

bool SecamDecoder::configure(const LdDecodeMetaData::VideoParameters &videoParameters) {

    config.videoParameters = videoParameters;

    return true;
}

QThread *SecamDecoder::makeThread(QAtomicInt& abort, DecoderPool& decoderPool) {
    return new SecamThread(abort, decoderPool, config);
}

SecamThread::SecamThread(QAtomicInt& _abort, DecoderPool& _decoderPool,
                       const SecamDecoder::Configuration &_config, QObject *parent)
    : DecoderThread(_abort, _decoderPool, parent), config(_config), dbFilter(makeFIRFilter(dbFilterCoeffs)), drFilter(makeFIRFilter(drFilterCoeffs))
{
    auto size = _config.videoParameters.fieldHeight * _config.videoParameters.fieldWidth;
    fob_buffer = QVector(size, 0.0);
    for_buffer = QVector(size, 0.0);
}

void SecamThread::decodeFrames(const QVector<SourceField> &inputFields, qint32 startIndex, qint32 endIndex,
                              QVector<ComponentFrame> &componentFrames)
{
    for (qint32 fieldIndex = startIndex, frameIndex = 0; fieldIndex < endIndex; fieldIndex += 2, frameIndex++) {
        decodeFrame(inputFields[fieldIndex], inputFields[fieldIndex + 1], componentFrames[frameIndex]);
    }
}

void SecamThread::decodeFrame(const SourceField &firstField, const SourceField &secondField, ComponentFrame &componentFrame)
{
    const LdDecodeMetaData::VideoParameters &videoParameters = config.videoParameters;

    bool ignoreUV = false;//decoderPool.getOutputWriter().getPixelFormat() == OutputWriter::PixelFormat::GRAY16;

    // Initialise and clear the component frame
    // Ignore UV if we're doing Grayscale output.
    // TODO: Fix so we don't need U/V vectors for RGB and YUV output either.
    componentFrame.init(videoParameters, ignoreUV);

    //dbFilter.apply(firstField.data, fob_buffer);
    drFilter.apply(firstField.data, for_buffer);

    auto shift = std::numeric_limits<quint16>::max() / 2;

//    qDebug() << "f" << for_buffer;
//    qDebug() << "shift" << shift;
//    qDebug() << "field data len" << firstField.data.size();
//    qDebug() << "buf len" << for_buffer.size();

    // Interlace the active lines of the two input fields to produce a component frame
    for (qint32 y = videoParameters.firstActiveFrameLine; y < videoParameters.lastActiveFrameLine; y+=2) {
        //const SourceVideo::Data &inputFieldData = (y % 2) == 0 ? firstField.data : secondField.data;
        const quint16 *inputLine = firstField.data.data() + ((y / 2) * videoParameters.fieldWidth);
        const auto *fLine = for_buffer.data() + ((y / 2) * videoParameters.fieldWidth);

        // Copy the whole composite signal to Y (leaving U and V blank)
        double *outY = componentFrame.y(y);
        for (qint32 x = videoParameters.activeVideoStart; x < videoParameters.activeVideoEnd; x++) {
            outY[x] = inputLine[x];
        }
    }

    drFilter.apply(secondField.data, for_buffer);

    // Interlace the active lines of the two input fields to produce a component frame
    for (qint32 y = videoParameters.firstActiveFrameLine + 1; y < videoParameters.lastActiveFrameLine; y+=2) {
        //const SourceVideo::Data &inputFieldData = (y % 2) == 0 ? firstField.data : secondField.data;
        const quint16 *inputLine = secondField.data.data() + ((y / 2) * videoParameters.fieldWidth);
        const auto *fLine = for_buffer.data() + ((y / 2) * videoParameters.fieldWidth);

        // Copy the whole composite signal to Y (leaving U and V blank)
        double *outY = componentFrame.y(y);
        for (qint32 x = videoParameters.activeVideoStart; x < videoParameters.activeVideoEnd; x++) {
            outY[x] = inputLine[x];
        }
    }
}
