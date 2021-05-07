/************************************************************************

    tbcsource.cpp

    ld-analyse - TBC output analysis
    Copyright (C) 2018-2021 Simon Inns
    Copyright (C) 2021 Adam Sampson

    This file is part of ld-decode-tools.

    ld-analyse is free software: you can redistribute it and/or
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

#include "tbcsource.h"

#include "sourcefield.h"

TbcSource::TbcSource(QObject *parent) : QObject(parent)
{
    // Default frame image options
    chromaOn = false;
    dropoutsOn = false;
    reverseFoOn = false;
    sourceReady = false;
    frameCacheFrameNumber = -1;

    // Configure the chroma decoder
    palConfiguration = palColour.getConfiguration();
    palConfiguration.chromaFilter = PalColour::transform2DFilter;
    ntscConfiguration = ntscColour.getConfiguration();
    outputConfiguration.pixelFormat = OutputWriter::PixelFormat::RGB48;
    outputConfiguration.usePadding = false;
    decoderConfigurationChanged = false;
}

// Public methods -----------------------------------------------------------------------------------------------------

// Method to load a TBC source file
void TbcSource::loadSource(QString sourceFilename)
{
    // Default frame options
    chromaOn = false;
    dropoutsOn = false;
    reverseFoOn = false;
    sourceReady = false;
    frameCacheFrameNumber = -1;

    // Set the current file name
    QFileInfo inFileInfo(sourceFilename);
    currentSourceFilename = inFileInfo.fileName();
    qDebug() << "TbcSource::startBackgroundLoad(): Opening TBC source file:" << currentSourceFilename;

    // Set up and fire-off background loading thread
    qDebug() << "TbcSource::loadSource(): Setting up background loader thread";
    connect(&watcher, SIGNAL(finished()), this, SLOT(finishBackgroundLoad()));
    future = QtConcurrent::run(this, &TbcSource::startBackgroundLoad, sourceFilename);
    watcher.setFuture(future);
}

// Method to unload a TBC source file
void TbcSource::unloadSource()
{
    sourceVideo.close();
    sourceReady = false;
}

// Method returns true is a TBC source is loaded
bool TbcSource::getIsSourceLoaded()
{
    return sourceReady;
}

// Method returns the filename of the current TBC source
QString TbcSource::getCurrentSourceFilename()
{
    if (!sourceReady) return QString();

    return currentSourceFilename;
}

// Method to set the highlight dropouts mode (true = dropouts highlighted)
void TbcSource::setHighlightDropouts(bool _state)
{
    frameCacheFrameNumber = -1;
    dropoutsOn = _state;
}

// Method to set the chroma decoder mode (true = on)
void TbcSource::setChromaDecoder(bool _state)
{
    frameCacheFrameNumber = -1;
    chromaOn = _state;
}

// Method to set the field order (true = reversed, false = normal)
void TbcSource::setFieldOrder(bool _state)
{
    frameCacheFrameNumber = -1;
    reverseFoOn = _state;

    if (reverseFoOn) ldDecodeMetaData.setIsFirstFieldFirst(false);
    else ldDecodeMetaData.setIsFirstFieldFirst(true);
}

// Method to get the state of the highlight dropouts mode
bool TbcSource::getHighlightDropouts()
{
    return dropoutsOn;
}

// Method to get the state of the chroma decoder mode
bool TbcSource::getChromaDecoder()
{
    return chromaOn;
}

// Method to get the field order
bool TbcSource::getFieldOrder()
{
    return reverseFoOn;
}

// Method to get a QImage from a frame number
QImage TbcSource::getFrameImage(qint32 frameNumber)
{
    if (!sourceReady) return QImage();

    // Check cached QImage
    if (frameCacheFrameNumber == frameNumber && !decoderConfigurationChanged) return frameCache;
    else {
        frameCacheFrameNumber = frameNumber;
        decoderConfigurationChanged = false;
    }

    // Get the required field numbers
    qint32 firstFieldNumber = ldDecodeMetaData.getFirstFieldNumber(frameNumber);
    qint32 secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(frameNumber);

    // Make sure we have a valid response from the frame determination
    if (firstFieldNumber == -1 || secondFieldNumber == -1) {
        qCritical() << "Could not determine field numbers!";

        // Jump back one frame
        if (frameNumber != 1) {
            frameNumber--;

            firstFieldNumber = ldDecodeMetaData.getFirstFieldNumber(frameNumber);
            secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(frameNumber);
        }
        qDebug() << "TbcSource::getFrameImage(): Jumping back one frame due to error";
    }

    // Get a QImage for the frame
    QImage frameImage = generateQImage(frameNumber);

    // Get the field metadata
    LdDecodeMetaData::Field firstField = ldDecodeMetaData.getField(firstFieldNumber);
    LdDecodeMetaData::Field secondField = ldDecodeMetaData.getField(secondFieldNumber);

    // Highlight dropouts
    if (dropoutsOn) {
        // Create a painter object
        QPainter imagePainter;
        imagePainter.begin(&frameImage);

        // Draw the drop out data for the first field
        imagePainter.setPen(Qt::red);
        for (qint32 dropOutIndex = 0; dropOutIndex < firstField.dropOuts.size(); dropOutIndex++) {
            qint32 startx = firstField.dropOuts.startx(dropOutIndex);
            qint32 endx = firstField.dropOuts.endx(dropOutIndex);
            qint32 fieldLine = firstField.dropOuts.fieldLine(dropOutIndex);

            imagePainter.drawLine(startx, ((fieldLine - 1) * 2), endx, ((fieldLine - 1) * 2));
        }

        // Draw the drop out data for the second field
        imagePainter.setPen(Qt::blue);
        for (qint32 dropOutIndex = 0; dropOutIndex < secondField.dropOuts.size(); dropOutIndex++) {
            qint32 startx = secondField.dropOuts.startx(dropOutIndex);
            qint32 endx = secondField.dropOuts.endx(dropOutIndex);
            qint32 fieldLine = secondField.dropOuts.fieldLine(dropOutIndex);

            imagePainter.drawLine(startx, ((fieldLine - 1) * 2) + 1, endx, ((fieldLine - 1) * 2) + 1);
        }

        // End the painter object
        imagePainter.end();
    }

    frameCache = frameImage;
    return frameImage;
}

// Method to get the number of available frames
qint32 TbcSource::getNumberOfFrames()
{
    if (!sourceReady) return 0;
    return ldDecodeMetaData.getNumberOfFrames();
}

// Method to get the number of available fields
qint32 TbcSource::getNumberOfFields()
{
    if (!sourceReady) return 0;
    return ldDecodeMetaData.getNumberOfFields();
}

// Method returns true if the TBC source is PAL (false for NTSC)
bool TbcSource::getIsSourcePal()
{
    if (!sourceReady) return false;
    return ldDecodeMetaData.getVideoParameters().isSourcePal;
}

// Method to get the frame height in scanlines
qint32 TbcSource::getFrameHeight()
{
    if (!sourceReady) return 0;

    // Get the metadata for the fields
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Calculate the frame height
    return (videoParameters.fieldHeight * 2) - 1;
}

// Method to get the frame width in dots
qint32 TbcSource::getFrameWidth()
{
    if (!sourceReady) return 0;

    // Get the metadata for the fields
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Return the frame width
    return (videoParameters.fieldWidth);
}

// Get black SNR data for graphing
QVector<qreal> TbcSource::getBlackSnrGraphData()
{
    return blackSnrGraphData;
}

// Get white SNR data for graphing
QVector<qreal> TbcSource::getWhiteSnrGraphData()
{
    return whiteSnrGraphData;
}

// Get dropout data for graphing
QVector<qreal> TbcSource::getDropOutGraphData()
{
    return dropoutGraphData;
}

// Method to get the size of the graphing data
qint32 TbcSource::getGraphDataSize()
{
    // All data vectors are the same size, just return the size on one
    return dropoutGraphData.size();
}

// Method returns true if frame contains dropouts
bool TbcSource::getIsDropoutPresent(qint32 frameNumber)
{
    if (!sourceReady) return false;

    bool dropOutsPresent = false;

    // Determine the first and second fields for the frame number
    qint32 firstFieldNumber = ldDecodeMetaData.getFirstFieldNumber(frameNumber);
    qint32 secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(frameNumber);

    if (ldDecodeMetaData.getFieldDropOuts(firstFieldNumber).size() > 0) dropOutsPresent = true;
    if (ldDecodeMetaData.getFieldDropOuts(secondFieldNumber).size() > 0) dropOutsPresent = true;

    return dropOutsPresent;
}

// Get scan line data from a frame
TbcSource::ScanLineData TbcSource::getScanLineData(qint32 frameNumber, qint32 scanLine)
{
    if (!sourceReady) return ScanLineData();

    // Determine the first and second fields for the frame number
    qint32 firstFieldNumber = ldDecodeMetaData.getFirstFieldNumber(frameNumber);
    qint32 secondFieldNumber = ldDecodeMetaData.getSecondFieldNumber(frameNumber);

    ScanLineData scanLineData;
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Convert the scan line into field and field line
    bool isFieldTop = true;
    qint32 fieldLine = 0;

    if (scanLine % 2 == 0) isFieldTop = false;
    else isFieldTop = true;

    if (isFieldTop) {
        fieldLine = (scanLine / 2) + 1;
    } else {
        fieldLine = (scanLine / 2);
    }

    // Set the video parameters
    scanLineData.blackIre = videoParameters.black16bIre;
    scanLineData.whiteIre = videoParameters.white16bIre;
    scanLineData.colourBurstStart = videoParameters.colourBurstStart;
    scanLineData.colourBurstEnd = videoParameters.colourBurstEnd;
    scanLineData.activeVideoStart = videoParameters.activeVideoStart;
    scanLineData.activeVideoEnd = videoParameters.activeVideoEnd;
    scanLineData.isSourcePal = videoParameters.isSourcePal;

    // Get the field video and dropout data
    SourceVideo::Data fieldData;
    DropOuts dropouts;
    if (isFieldTop) {
        fieldData = sourceVideo.getVideoField(firstFieldNumber);
        dropouts = ldDecodeMetaData.getFieldDropOuts(firstFieldNumber);
    } else {
        fieldData = sourceVideo.getVideoField(secondFieldNumber);
        dropouts = ldDecodeMetaData.getFieldDropOuts(secondFieldNumber);
    }

    scanLineData.data.resize(videoParameters.fieldWidth);
    scanLineData.isDropout.resize(videoParameters.fieldWidth);
    for (qint32 xPosition = 0; xPosition < videoParameters.fieldWidth; xPosition++) {
        // Get the 16-bit YC value for the current pixel (frame data is numbered 0-624 or 0-524)
        scanLineData.data[xPosition] = fieldData[((fieldLine - 1) * videoParameters.fieldWidth) + xPosition];

        scanLineData.isDropout[xPosition] = false;
        for (qint32 doCount = 0; doCount < dropouts.size(); doCount++) {
            if (dropouts.fieldLine(doCount) == fieldLine) {
                if (xPosition >= dropouts.startx(doCount) && xPosition <= dropouts.endx(doCount)) scanLineData.isDropout[xPosition] = true;
            }
        }
    }

    return scanLineData;
}

// Method to return the decoded VBI data for a frame
VbiDecoder::Vbi TbcSource::getFrameVbi(qint32 frameNumber)
{
    if (!sourceReady) return VbiDecoder::Vbi();

    // Get the field VBI data
    LdDecodeMetaData::Vbi firstField = ldDecodeMetaData.getFieldVbi(ldDecodeMetaData.getFirstFieldNumber(frameNumber));
    LdDecodeMetaData::Vbi secondField = ldDecodeMetaData.getFieldVbi(ldDecodeMetaData.getSecondFieldNumber(frameNumber));

    return vbiDecoder.decodeFrame(firstField.vbiData[0], firstField.vbiData[1], firstField.vbiData[2],
            secondField.vbiData[0], secondField.vbiData[1], secondField.vbiData[2]);
}

// Method returns true if the VBI is valid for the specified frame number
bool TbcSource::getIsFrameVbiValid(qint32 frameNumber)
{
    if (!sourceReady) return false;

    // Get the field VBI data
    LdDecodeMetaData::Vbi firstField = ldDecodeMetaData.getFieldVbi(ldDecodeMetaData.getFirstFieldNumber(frameNumber));
    LdDecodeMetaData::Vbi secondField = ldDecodeMetaData.getFieldVbi(ldDecodeMetaData.getSecondFieldNumber(frameNumber));

    if (firstField.vbiData[0] == -1 || firstField.vbiData[1] == -1 || firstField.vbiData[2] == -1) return false;
    if (secondField.vbiData[0] == -1 || secondField.vbiData[1] == -1 || secondField.vbiData[2] == -1) return false;

    return true;
}

// Method to get the field number of the first field of the specified frame
qint32 TbcSource::getFirstFieldNumber(qint32 frameNumber)
{
    if (!sourceReady) return 0;

    return ldDecodeMetaData.getFirstFieldNumber(frameNumber);
}

// Method to get the field number of the second field of the specified frame
qint32 TbcSource::getSecondFieldNumber(qint32 frameNumber)
{
    if (!sourceReady) return 0;

    return ldDecodeMetaData.getSecondFieldNumber(frameNumber);
}

qint32 TbcSource::getCcData0(qint32 frameNumber)
{
    if (!sourceReady) return false;

    // Get the field metadata
    LdDecodeMetaData::Field firstField = ldDecodeMetaData.getField(ldDecodeMetaData.getFirstFieldNumber(frameNumber));
    LdDecodeMetaData::Field secondField = ldDecodeMetaData.getField(ldDecodeMetaData.getSecondFieldNumber(frameNumber));

    if (firstField.ntsc.ccData0 != -1) return firstField.ntsc.ccData0;
    return secondField.ntsc.ccData0;
}

qint32 TbcSource::getCcData1(qint32 frameNumber)
{
    if (!sourceReady) return false;

    // Get the field metadata
    LdDecodeMetaData::Field firstField = ldDecodeMetaData.getField(ldDecodeMetaData.getFirstFieldNumber(frameNumber));
    LdDecodeMetaData::Field secondField = ldDecodeMetaData.getField(ldDecodeMetaData.getSecondFieldNumber(frameNumber));

    if (firstField.ntsc.ccData1 != -1) return firstField.ntsc.ccData1;
    return secondField.ntsc.ccData1;
}

void TbcSource::setChromaConfiguration(const PalColour::Configuration &_palConfiguration,
                                       const Comb::Configuration &_ntscConfiguration,
                                       const OutputWriter::Configuration &_outputConfiguration)
{
    palConfiguration = _palConfiguration;
    ntscConfiguration = _ntscConfiguration;
    outputConfiguration = _outputConfiguration;

    // Configure the chroma decoder
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();
    if (videoParameters.isSourcePal) {
        palColour.updateConfiguration(videoParameters, palConfiguration);
    } else {
        ntscColour.updateConfiguration(videoParameters, ntscConfiguration);
    }

    // Configure the OutputWriter.
    // Because we have padding disabled, this won't change the VideoParameters.
    outputWriter.updateConfiguration(videoParameters, outputConfiguration);

    decoderConfigurationChanged = true;
}

const PalColour::Configuration &TbcSource::getPalConfiguration()
{
    return palConfiguration;
}

const Comb::Configuration &TbcSource::getNtscConfiguration()
{
    return ntscConfiguration;
}

const OutputWriter::Configuration &TbcSource::getOutputConfiguration()
{
    return outputConfiguration;
}

// Return the frame number of the start of the next chapter
qint32 TbcSource::startOfNextChapter(qint32 currentFrameNumber)
{
    // Do we have a chapter map?
    if (chapterMap.size() == 0) return getNumberOfFrames();

    qint32 mapLocation = -1;
    for (qint32 i = 0; i < chapterMap.size(); i++) {
        if (chapterMap[i] > currentFrameNumber) {
            mapLocation = i;
            break;
        }
    }

    // Found?
    if (mapLocation != -1) {
        return chapterMap[mapLocation];
    }

    return getNumberOfFrames();
}

// Return the frame number of the start of the current chapter
qint32 TbcSource::startOfChapter(qint32 currentFrameNumber)
{
    // Do we have a chapter map?
    if (chapterMap.size() == 0) return 1;

    qint32 mapLocation = -1;
    for (qint32 i = chapterMap.size() - 1; i >= 0; i--) {
        if (chapterMap[i] < currentFrameNumber) {
            mapLocation = i;
            break;
        }
    }

    // Found?
    if (mapLocation != -1) {
        return chapterMap[mapLocation];
    }

    return 1;
}


// Private methods ----------------------------------------------------------------------------------------------------

// Method to create a QImage for a source video frame
QImage TbcSource::generateQImage(qint32 frameNumber)
{
    // Get the metadata for the video parameters
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Calculate the frame height
    qint32 frameHeight = (videoParameters.fieldHeight * 2) - 1;

    // Show debug information
    if (chromaOn) {
        qDebug().nospace() << "TbcSource::generateQImage(): Generating a chroma image from frame " << frameNumber <<
                    " (" << videoParameters.fieldWidth << "x" << frameHeight << ")";
    } else {
        qDebug().nospace() << "TbcSource::generateQImage(): Generating a source image from frame " << frameNumber <<
                    " (" << videoParameters.fieldWidth << "x" << frameHeight << ")";
    }

    // Create a QImage
    QImage frameImage = QImage(videoParameters.fieldWidth, frameHeight, QImage::Format_RGB888);

    // Work out how many frames ahead/behind we need to fetch
    qint32 lookBehind, lookAhead;
    if (!chromaOn) {
        // Not decoding chroma -- so none
        lookBehind = 0;
        lookAhead = 0;
    } else if (videoParameters.isSourcePal) {
        lookBehind = palConfiguration.getLookBehind();
        lookAhead = palConfiguration.getLookAhead();
    } else {
        lookBehind = ntscConfiguration.getLookBehind();
        lookAhead = ntscConfiguration.getLookAhead();
    }

    // Fetch the input fields and metadata
    QVector<SourceField> inputFields;
    qint32 startIndex, endIndex;
    SourceField::loadFields(sourceVideo, ldDecodeMetaData,
                            frameNumber, 1, lookBehind, lookAhead,
                            inputFields, startIndex, endIndex);

    if (chromaOn) {
        // Chroma decode the current frame and display

        // Decode the current frame to components
        QVector<ComponentFrame> componentFrames(1);
        if (videoParameters.isSourcePal) {
            // PAL source
            palColour.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
        } else {
            // NTSC source
            ntscColour.decodeFrames(inputFields, startIndex, endIndex, componentFrames);
        }

        // Convert component video to RGB
        OutputFrame outputFrame;
        outputWriter.convert(componentFrames[0], outputFrame);

        // Get a pointer to the RGB data
        const quint16 *rgbPointer = outputFrame.data();

        // Fill the QImage with black
        frameImage.fill(Qt::black);

        // Copy the RGB16-16-16 data into the RGB888 QImage
        const qint32 activeHeight = videoParameters.lastActiveFrameLine - videoParameters.firstActiveFrameLine;
        const qint32 activeWidth = videoParameters.activeVideoEnd - videoParameters.activeVideoStart;
        for (qint32 y = 0; y < activeHeight; y++) {
            const quint16 *inputLine = rgbPointer + (y * activeWidth * 3);
            uchar *outputLine = frameImage.scanLine(y + videoParameters.firstActiveFrameLine)
                                + (videoParameters.activeVideoStart * 3);

            // Take just the MSB of the RGB input data
            for (qint32 i = 0; i < activeWidth * 3; i++) {
                *outputLine++ = static_cast<uchar>((*inputLine++) / 256);
            }
        }
    } else {
        // Display the current frame as source data

        // Get pointers to the 16-bit greyscale data
        const quint16 *firstFieldPointer = inputFields[startIndex].data.data();
        const quint16 *secondFieldPointer = inputFields[startIndex + 1].data.data();

        // Copy the raw 16-bit grayscale data into the RGB888 QImage
        for (qint32 y = 0; y < frameHeight; y++) {
            for (qint32 x = 0; x < videoParameters.fieldWidth; x++) {
                // Take just the MSB of the input data
                qint32 pixelOffset = (videoParameters.fieldWidth * (y / 2)) + x;
                uchar pixelValue;
                if (y % 2) {
                    pixelValue = static_cast<uchar>(secondFieldPointer[pixelOffset] / 256);
                } else {
                    pixelValue = static_cast<uchar>(firstFieldPointer[pixelOffset] / 256);
                }

                qint32 xpp = x * 3;
                *(frameImage.scanLine(y) + xpp + 0) = static_cast<uchar>(pixelValue); // R
                *(frameImage.scanLine(y) + xpp + 1) = static_cast<uchar>(pixelValue); // G
                *(frameImage.scanLine(y) + xpp + 2) = static_cast<uchar>(pixelValue); // B
            }
        }
    }

    return frameImage;
}

// Generate the data points for the Drop-out and SNR analysis graphs
// We do these both at the same time to reduce calls to the metadata
void TbcSource::generateData()
{
    dropoutGraphData.clear();
    blackSnrGraphData.clear();
    whiteSnrGraphData.clear();

    dropoutGraphData.resize(ldDecodeMetaData.getNumberOfFrames());
    blackSnrGraphData.resize(ldDecodeMetaData.getNumberOfFrames());
    whiteSnrGraphData.resize(ldDecodeMetaData.getNumberOfFrames());

    for (qint32 frameNumber = 0; frameNumber < ldDecodeMetaData.getNumberOfFrames(); frameNumber++) {
        qreal doLength = 0;
        qreal blackSnrTotal = 0;
        qreal whiteSnrTotal = 0;

        // SNR data may be missing in some fields, so we count the points to prevent
        // the frame average from being thrown-off by missing data
        qreal blackSnrPoints = 0;
        qreal whiteSnrPoints = 0;

        LdDecodeMetaData::Field firstField = ldDecodeMetaData.getField(ldDecodeMetaData.getFirstFieldNumber(frameNumber + 1));
        LdDecodeMetaData::Field secondField = ldDecodeMetaData.getField(ldDecodeMetaData.getSecondFieldNumber(frameNumber + 1));

        // Get the first field DOs
        if (firstField.dropOuts.size() > 0) {
            // Calculate the total length of the dropouts
            for (qint32 i = 0; i < firstField.dropOuts.size(); i++) {
                doLength += firstField.dropOuts.endx(i) - firstField.dropOuts.startx(i);
            }
        }

        // Get the second field DOs
        if (secondField.dropOuts.size() > 0) {
            // Calculate the total length of the dropouts
            for (qint32 i = 0; i < secondField.dropOuts.size(); i++) {
                doLength += secondField.dropOuts.endx(i) - secondField.dropOuts.startx(i);
            }
        }

        // Get the first field SNRs
        if (firstField.vitsMetrics.inUse) {
            if (firstField.vitsMetrics.bPSNR > 0) {
                blackSnrTotal += firstField.vitsMetrics.bPSNR;
                blackSnrPoints++;
            }
            if (firstField.vitsMetrics.wSNR > 0) {
                whiteSnrTotal += firstField.vitsMetrics.wSNR;
                whiteSnrPoints++;
            }
        }

        // Get the second field SNRs
        if (secondField.vitsMetrics.inUse) {
            if (secondField.vitsMetrics.bPSNR > 0) {
                blackSnrTotal += secondField.vitsMetrics.bPSNR;
                blackSnrPoints++;
            }
            if (secondField.vitsMetrics.wSNR > 0) {
                whiteSnrTotal += secondField.vitsMetrics.wSNR;
                whiteSnrPoints++;
            }
        }

        // Add the result to the vectors
        dropoutGraphData[frameNumber] = doLength;
        blackSnrGraphData[frameNumber] = blackSnrTotal / blackSnrPoints; // Calc average for frame
        whiteSnrGraphData[frameNumber] = whiteSnrTotal / whiteSnrPoints; // Calc average for frame
    }
}

void TbcSource::startBackgroundLoad(QString sourceFilename)
{
    // Open the TBC metadata file
    qDebug() << "TbcSource::startBackgroundLoad(): Processing JSON metadata...";
    emit busyLoading("Processing JSON metadata...");

    QString jsonFileName = sourceFilename + ".json";

    const bool chroma_tbc = sourceFilename.endsWith("_chroma.tbc");

    // If we are trying to open a _chroma tbc from vhs-decode.
    // Try to look for the json for the luma part if the chroma doesn't have it's own.
    if (!QFileInfo::exists(jsonFileName) && chroma_tbc) {
        jsonFileName.chop(16);
        jsonFileName += ".tbc.json";
    }

    if (!ldDecodeMetaData.read(jsonFileName)) {
        // Open failed
        qWarning() << "Open TBC JSON metadata failed for filename" << sourceFilename;
        currentSourceFilename.clear();

        // Show an error to the user
        lastLoadError = "Could not open TBC JSON metadata file for the TBC input file!";
    } else {
        // Get the video parameters from the metadata
        LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

        // Open the new source video
        qDebug() << "TbcSource::startBackgroundLoad(): Loading TBC file...";
        emit busyLoading("Loading TBC file...");
        if (!sourceVideo.open(sourceFilename, videoParameters.fieldWidth * videoParameters.fieldHeight)) {
            // Open failed
            qWarning() << "Open TBC file failed for filename" << sourceFilename;
            currentSourceFilename.clear();

            // Show an error to the user
            lastLoadError = "Could not open TBC data file!";
        } else {
            // Both the video and metadata files are now open
            sourceReady = true;
            currentSourceFilename = sourceFilename;
        }
    }

    // Get the video parameters
    LdDecodeMetaData::VideoParameters videoParameters = ldDecodeMetaData.getVideoParameters();

    // Configure the chroma decoder
    if (videoParameters.isSourcePal) {
        palColour.updateConfiguration(videoParameters, palConfiguration);
    } else {
        // Enable this option by default if we are loading a vhs-decode chroma only tbc file.
        if(chroma_tbc) {
            ntscConfiguration.phaseCompensation = true;
        }
        ntscColour.updateConfiguration(videoParameters, ntscConfiguration);
    }

    // Generate the graph data for the source
    emit busyLoading("Generating graph data...");
    generateData();

    // Generate a chapter map (used by the chapter skip
    // forwards and backwards buttons)
    emit busyLoading("Generating VBI chapter map...");
    qint32 lastChapter = -1;
    qint32 giveUpCounter = 0;
    chapterMap.clear();
    for (qint32 i = 1; i <= getNumberOfFrames(); i++) {
        qint32 currentChapter = getFrameVbi(i).chNo;
        if (currentChapter != -1) {
            if (currentChapter != lastChapter) {
                lastChapter = currentChapter;
                chapterMap.append(i);
            } else giveUpCounter++;
        }

        if (i == 100 && giveUpCounter < 50) {
            qDebug() << "Not seeing valid chapter numbers, giving up chapter mapping";
            break;
        }
    }
}

void TbcSource::finishBackgroundLoad()
{
    // Send a finished loading message to the main window
    emit finishedLoading();
}
