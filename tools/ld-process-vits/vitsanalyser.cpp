/************************************************************************

    vitsanalyser.cpp

    ld-process-vits - Vertical Interval Test Signal processing
    Copyright (C) 2020 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-vits is free software: you can redistribute it and/or
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

#include "vitsanalyser.h"
#include "processingpool.h"

VitsAnalyser::VitsAnalyser(QAtomicInt& _abort, ProcessingPool& _processingPool, QObject *parent)
    : QThread(parent), abort(_abort), processingPool(_processingPool)
{

}

// Thread main processing method
void VitsAnalyser::run()
{
    qint32 fieldNumber;

    // Input data buffers
    SourceVideo::Data sourceFieldData;
    LdDecodeMetaData::Field fieldMetadata;

    while(!abort) {
        // Get the next field to process from the input file
        if (!processingPool.getInputField(fieldNumber, sourceFieldData, fieldMetadata, videoParameters)) {
            // No more input fields -- exit
            break;
        }

        // Show an update to the user (for every 1000th field)
        if (fieldNumber % 1000 == 0) {
            qInfo() << "Processing field" << fieldNumber;
        }

        // Get multiple possible black and white measurement points based on video format, etc.
        QVector<QVector<double>> wlSlice;
        QVector<QVector<double>> blSlice;

        if (videoParameters.system == PAL) {
            // 625 lines (taken from ld-decode core.py)
            wlSlice.append(getFieldLineSlice(sourceFieldData, 19, 12, 8));
            blSlice.append(getFieldLineSlice(sourceFieldData, 22, 12, 50));
        } else {
            // 525 lines (taken from ld-decode core.py)
            wlSlice.append(getFieldLineSlice(sourceFieldData, 20, 14, 12));
            wlSlice.append(getFieldLineSlice(sourceFieldData, 20, 52, 8));
            wlSlice.append(getFieldLineSlice(sourceFieldData, 13, 13, 15));
            blSlice.append(getFieldLineSlice(sourceFieldData, 1, 10, 20));
        }

        // Only pick the white slice if it has a mean value between 90 and 110 IRE
        qint32 wlSliceToUse = -1;
        for (qint32 i = 0; i < wlSlice.size(); i++) {
            double wlMean = calcMean(wlSlice[i]);
            if (wlMean >= 90 && wlMean <= 110) {
                wlSliceToUse = i;
                break;
            }
        }

        // Always use the first black slice (there is only ever one to choose from)
        // Doing it this way in case more sources are added in the future
        qint32 blSliceToUse = 0;

        // Only calculate the wSNR if we have a valid slice
        double wSNR = 0;
        if (wlSliceToUse != -1) wSNR = calculateSnr(wlSlice[wlSliceToUse], true);

        // Only calculate the bPSNR if we have a valid slice
        double bPSNR = 0;
        if (blSliceToUse != -1) bPSNR = calculateSnr(blSlice[blSliceToUse], true);

        // Update the metadata for the field
        double old_wSNR = fieldMetadata.vitsMetrics.wSNR;
        double old_bPSNR = fieldMetadata.vitsMetrics.bPSNR;
        fieldMetadata.vitsMetrics.wSNR = roundDouble(wSNR, 1);
        fieldMetadata.vitsMetrics.bPSNR = roundDouble(bPSNR, 1);

        // Show the result as debug
        qDebug().nospace() << "Field #" << fieldNumber << " has wSNR of " << fieldMetadata.vitsMetrics.wSNR << " (" << old_wSNR << ")"
                 << " and bPSNR of " << fieldMetadata.vitsMetrics.bPSNR << " (" << old_bPSNR << ")";

        // Write the result to the output metadata
        if (!processingPool.setOutputField(fieldNumber, fieldMetadata)) {
            abort = true;
            break;
        }
    }
}

// Get a specific slice of a field line and return all the values
QVector<double> VitsAnalyser::getFieldLineSlice(const SourceVideo::Data &sourceField, qint32 fieldLine, qint32 startUs, qint32 lengthUs)
{
    QVector<double> returnData;
    fieldLine--; // Adjust for field offset

    // Range-check the field line
    if (fieldLine < 0 || fieldLine >= videoParameters.fieldHeight) {
        qWarning() << "Cannot generate field-line data, line number is out of bounds! Scan line =" << fieldLine;
        return returnData;
    }

    // Calculate the number of samples per uS for the field
    double samplesPerUs = 0;
    if (videoParameters.system == PAL) samplesPerUs = static_cast<double>(videoParameters.fieldWidth) / 64.0;
    else samplesPerUs = static_cast<double>(videoParameters.fieldWidth) / 63.5;

    // Get the start and end sample positions
    double startSampleDouble = startUs * samplesPerUs;
    double lengthSampleDouble = lengthUs * samplesPerUs;

    qint32 startPointer = (fieldLine * videoParameters.fieldWidth) + static_cast<qint32>(startSampleDouble);
    qint32 length = static_cast<qint32>(lengthSampleDouble);

    // Convert data points to floating-point IRE values
    returnData.resize(length);
    for (qint32 i = startPointer; i < startPointer + length; i++) {
        returnData[i - startPointer] =  (static_cast<double>(sourceField[i]) - static_cast<double>(videoParameters.black16bIre)) /
                ((static_cast<double>(videoParameters.white16bIre) - static_cast<double>(videoParameters.black16bIre)) / 100.0);
    }

    return returnData;
}

// Calculate the SNR or Percentage SNR
double VitsAnalyser::calculateSnr(QVector<double> &data, bool usePsnr)
{
    double signal = 0;
    if (usePsnr) signal = 100.0; else signal = calcMean(data); // Compute the arithmetic mean
    double noise = calcStd(data); // Compute the standard deviation

    return 20.0 * log10(signal / noise);
}

// The arithmetic mean is the sum of the elements divided by the number of elements.
double VitsAnalyser::calcMean(QVector<double> &data)
{
    double result = 0;

    for (qint32 i = 0; i < data.size(); i++) {
        result += data[i];
    }

    return result / static_cast<double>(data.size());
}

// The standard deviation is the square root of the average of the squared deviations from the mean
double VitsAnalyser::calcStd(QVector<double> &data)
{
    double sum = 0.0;
    double mean = 0.0;
    double standardDeviation = 0.0;

    for(qint32 i = 0; i < data.size(); ++i)
        sum += data[i];

    mean = sum / static_cast<double>(data.size());

    for(qint32 i = 0; i < data.size(); ++i)
        standardDeviation += pow(data[i] - mean, 2.0);

    return sqrt(standardDeviation / static_cast<double>(data.size()));
}

// Round a double to x decimal places
double VitsAnalyser::roundDouble(double in, qint32 decimalPlaces)
{
    const double multiplier = pow(10.0, decimalPlaces);
    return ceil(in * multiplier) / multiplier;
}


