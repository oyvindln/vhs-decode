/************************************************************************

    outputwriter.cpp

    ld-chroma-decoder - Colourisation filter for ld-decode
    Copyright (C) 2018-2021 Chad Page
    Copyright (C) 2020-2021 Adam Sampson
    Copyright (C) 2021 Phillip Blucas

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

#include "outputwriter.h"

#include "componentframe.h"

// Limits, zero points and scaling factors (from 0-1) for Y'CbCr colour representations
// [Poynton ch25 p305] [BT.601-7 sec 2.5.3]
static constexpr double Y_MIN   = 1.0    * 256.0;
static constexpr double Y_ZERO  = 16.0   * 256.0;
static constexpr double Y_SCALE = 219.0  * 256.0;
static constexpr double Y_MAX   = 254.75 * 256.0;
static constexpr double C_MIN   = 1.0    * 256.0;
static constexpr double C_ZERO  = 128.0  * 256.0;
static constexpr double C_SCALE = 112.0  * 256.0;
static constexpr double C_MAX   = 254.75 * 256.0;

// ITU-R BT.601-7
// [Poynton eq 25.1 p303 and eq 25.5 p307]
static constexpr double ONE_MINUS_Kb = 1.0 - 0.114;
static constexpr double ONE_MINUS_Kr = 1.0 - 0.299;

// kB = sqrt(209556997.0 / 96146491.0) / 3.0
// kR = sqrt(221990474.0 / 288439473.0)
// [Poynton eq 28.1 p336]
static constexpr double kB = 0.49211104112248356308804691718185;
static constexpr double kR = 0.87728321993817866838972487283129;

void OutputWriter::updateConfiguration(LdDecodeMetaData::VideoParameters &_videoParameters,
                                       const OutputWriter::Configuration &_config)
{
    config = _config;
    videoParameters = _videoParameters;
    topPadLines = 0;
    bottomPadLines = 0;

    activeWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
    activeHeight = videoParameters.lastActiveFrameLine - videoParameters.firstActiveFrameLine;
    outputHeight = activeHeight;

    if (config.usePadding) {
        // Both width and height should be divisible by 8, as video codecs expect this.
        // Expand horizontal active region so the width is divisible by 8.
        while (true) {
            activeWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
            if ((activeWidth % 8) == 0) {
                break;
            }

            // Add pixels to the right and left sides in turn, to keep the active area centred
            if ((activeWidth % 2) == 0) {
                videoParameters.activeVideoEnd++;
            } else {
                videoParameters.activeVideoStart--;
            }
        }

        // Insert empty padding lines so the height is divisible by 8
        while (true) {
            outputHeight = topPadLines + activeHeight + bottomPadLines;
            if ((outputHeight % 8) == 0) {
                break;
            }

            // Add lines to the bottom and top in turn, to keep the active area centred
            if ((outputHeight % 2) == 0) {
                bottomPadLines++;
            } else {
                topPadLines++;
            }
        }

        // Update the caller's copy, now we've adjusted the active area
        _videoParameters = videoParameters;
    }
}

const char *OutputWriter::getPixelName() const
{
    switch (config.pixelFormat) {
    case RGB48:
        return "RGB48";
    case YUV444P16:
        return "YUV444P16";
    case GRAY16:
        return "GRAY16";
    default:
        return "unknown";
    }
}

void OutputWriter::printOutputInfo() const
{
    // Show output information to the user
    const qint32 frameHeight = (videoParameters.fieldHeight * 2) - 1;
    qInfo() << "Input video of" << videoParameters.fieldWidth << "x" << frameHeight
            << "will be colourised and trimmed to" << activeWidth << "x" << outputHeight
            << getPixelName() << "frames";
}

QByteArray OutputWriter::getStreamHeader() const
{
    // Only yuv4mpeg output needs a header
    if (!config.outputY4m) {
        return QByteArray();
    }

    QString header;
    QTextStream str(&header);

    str << "YUV4MPEG2";

    // Frame size
    str << " W" << activeWidth;
    str << " H" << outputHeight;

    // Frame rate
    if (videoParameters.isSourcePal) {
        str << " F25:1";
    } else {
        str << " F30000:1001";
    }

    // Field order
    str << " It";

    // Pixel aspect ratio
    // XXX Can this be computed, in case the width has been adjusted?
    if (videoParameters.isSourcePal) {
        if (videoParameters.isWidescreen) {
            str << " A512:461"; // (16 / 9) * (576 / 922)
        } else {
            str << " A384:461"; // (4 / 3) * (576 / 922)
        }
    } else {
        if (videoParameters.isWidescreen) {
            str << " A194:171"; // (16 / 9) * (485 / 760)
        } else {
            str << " A97:114";  // (4 / 3) * (485 / 760)
        }
    }

    // Pixel format
    switch (config.pixelFormat) {
    case YUV444P16:
        str << " C444p16 XCOLORRANGE=LIMITED";
        break;
    case GRAY16:
        str << " Cmono16 XCOLORRANGE=LIMITED";
        break;
    default:
        qFatal("pixel format not supported in yuv4mpeg header");
        break;
    }

    str << "\n";
    return header.toUtf8();
}

QByteArray OutputWriter::getFrameHeader() const
{
    // Only yuv4mpeg output needs a header
    if (!config.outputY4m) {
        return QByteArray();
    }

    return QStringLiteral("FRAME\n").toUtf8();
}

void OutputWriter::convert(const ComponentFrame &componentFrame, OutputFrame &outputFrame) const
{
    // Work out the number of output values, and resize the vector accordingly
    qint32 totalSize = activeWidth * outputHeight;
    switch (config.pixelFormat) {
    case RGB48:
    case YUV444P16:
        totalSize *= 3;
        break;
    case GRAY16:
        break;
    }
    outputFrame.resize(totalSize);

    // Clear padding
    clearPadLines(0, topPadLines, outputFrame);
    clearPadLines(outputHeight - bottomPadLines, bottomPadLines, outputFrame);

    // Convert active lines
    for (qint32 y = 0; y < activeHeight; y++) {
        convertLine(y, componentFrame, outputFrame);
    }
}

void OutputWriter::clearPadLines(qint32 firstLine, qint32 numLines, OutputFrame &outputFrame) const
{
    switch (config.pixelFormat) {
        case RGB48: {
            // Fill with RGB black
            quint16 *out = outputFrame.data() + (activeWidth * firstLine * 3);

            for (qint32 i = 0; i < numLines * activeWidth * 3; i++) {
                out[i] = 0;
            }

            break;
        }
        case YUV444P16: {
            // Fill Y with black, no chroma
            quint16 *outY  = outputFrame.data() + (activeWidth * firstLine);
            quint16 *outCB = outY + (activeWidth * outputHeight);
            quint16 *outCR = outCB + (activeWidth * outputHeight);

            for (qint32 i = 0; i < numLines * activeWidth; i++) {
                outY[i]  = static_cast<quint16>(Y_ZERO);
                outCB[i] = static_cast<quint16>(C_ZERO);
                outCR[i] = static_cast<quint16>(C_ZERO);
            }

            break;
        }
        case GRAY16: {
            // Fill with black
            quint16 *out = outputFrame.data() + (activeWidth * firstLine);

            for (qint32 i = 0; i < numLines * activeWidth; i++) {
                out[i] = static_cast<quint16>(Y_ZERO);
            }

            break;
        }
    }
}

void OutputWriter::convertLine(qint32 lineNumber, const ComponentFrame &componentFrame, OutputFrame &outputFrame) const
{
    // Get pointers to the component data for the active region
    const qint32 inputLine = videoParameters.firstActiveFrameLine + lineNumber;
    const double *inY = componentFrame.y(inputLine) + videoParameters.activeVideoStart;
    // Not used if output is GRAY16
    const double *inU = (config.pixelFormat != GRAY16) ?
                            componentFrame.u(inputLine) + videoParameters.activeVideoStart : nullptr;
    const double *inV = (config.pixelFormat != GRAY16) ?
                            componentFrame.v(inputLine) + videoParameters.activeVideoStart : nullptr;

    const qint32 outputLine = topPadLines + lineNumber;

    const double yOffset = videoParameters.black16bIre;
    double yRange = videoParameters.white16bIre - videoParameters.black16bIre;
    const double uvRange = yRange;

    switch (config.pixelFormat) {
        case RGB48: {
            // Convert Y'UV to full-range R'G'B' [Poynton eq 28.6 p337]
            quint16 *out = outputFrame.data() + (activeWidth * outputLine * 3);

            const double yScale = 65535.0 / yRange;
            const double uvScale = 65535.0 / uvRange;

            for (qint32 x = 0; x < activeWidth; x++) {
                // Scale Y'UV to 0-65535
                const double rY = qBound(0.0, (inY[x] - yOffset) * yScale, 65535.0);
                const double rU = inU[x] * uvScale;
                const double rV = inV[x] * uvScale;

                // Convert Y'UV to R'G'B'
                const qint32 pos = x * 3;
                out[pos]     = static_cast<quint16>(qBound(0.0, rY                    + (1.139883 * rV),  65535.0));
                out[pos + 1] = static_cast<quint16>(qBound(0.0, rY + (-0.394642 * rU) + (-0.580622 * rV), 65535.0));
                out[pos + 2] = static_cast<quint16>(qBound(0.0, rY + (2.032062 * rU),                     65535.0));
            }

            break;
        }
        case YUV444P16: {
            // Convert Y'UV to Y'CbCr [Poynton eq 25.5 p307]
            quint16 *outY  = outputFrame.data() + (activeWidth * outputLine);
            quint16 *outCB = outY + (activeWidth * outputHeight);
            quint16 *outCR = outCB + (activeWidth * outputHeight);

            const double yScale = Y_SCALE / yRange;
            const double cbScale = (C_SCALE / (ONE_MINUS_Kb * kB)) / uvRange;
            const double crScale = (C_SCALE / (ONE_MINUS_Kr * kR)) / uvRange;

            for (qint32 x = 0; x < activeWidth; x++) {
                outY[x]  = static_cast<quint16>(qBound(Y_MIN, ((inY[x] - yOffset) * yScale)  + Y_ZERO, Y_MAX));
                outCB[x] = static_cast<quint16>(qBound(C_MIN, (inU[x]             * cbScale) + C_ZERO, C_MAX));
                outCR[x] = static_cast<quint16>(qBound(C_MIN, (inV[x]             * crScale) + C_ZERO, C_MAX));
            }

            break;
        }
        case GRAY16: {
            // Throw away UV and just convert Y' to the same scale as Y'CbCr
            quint16 *out = outputFrame.data() + (activeWidth * outputLine);

            const double yScale = Y_SCALE / yRange;

            for (qint32 x = 0; x < activeWidth; x++) {
                out[x] = static_cast<quint16>(qBound(Y_MIN, ((inY[x] - yOffset) * yScale) + Y_ZERO, Y_MAX));
            }

            break;
        }
    }
}
