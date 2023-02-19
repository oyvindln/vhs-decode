/************************************************************************

    vbilinedecoder.cpp

    ld-process-vbi - VBI and IEC NTSC specific processor for ld-decode
    Copyright (C) 2018-2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-vbi is free software: you can redistribute it and/or
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

#include "vbilinedecoder.h"
#include "biphasecode.h"
#include "closedcaption.h"
#include "decoderpool.h"
#include "fmcode.h"
#include "vitccode.h"
#include "whiteflag.h"

VbiLineDecoder::VbiLineDecoder(QAtomicInt& _abort, DecoderPool& _decoderPool, QObject *parent)
    : QThread(parent), abort(_abort), decoderPool(_decoderPool)
{

}

// Thread main processing method
void VbiLineDecoder::run()
{
    qint32 fieldNumber;

    // Input data buffers
    SourceVideo::Data sourceFieldData;
    LdDecodeMetaData::Field fieldMetadata;
    LdDecodeMetaData::VideoParameters videoParameters;

    while (!abort) {
        // Get the next field to process from the input file
        if (!decoderPool.getInputField(fieldNumber, sourceFieldData, fieldMetadata, videoParameters)) {
            // No more input fields -- exit
            break;
        }

        // Show progress (for every 1000th field)
        if (fieldNumber % 1000 == 0) {
            qInfo() << "Processing field" << fieldNumber;
        }

        if (fieldMetadata.isFirstField) qDebug() << "VbiLineDecoder::process(): Getting metadata for field" << fieldNumber << "(first)";
        else qDebug() << "VbiLineDecoder::process(): Getting metadata for field" << fieldNumber << "(second)";

        // Get the 24-bit biphase-coded data from field lines 16-18
        BiphaseCode biphaseCode;
        biphaseCode.decodeLines(getFieldLine(sourceFieldData, 16, videoParameters),
                                getFieldLine(sourceFieldData, 17, videoParameters),
                                getFieldLine(sourceFieldData, 18, videoParameters),
                                videoParameters, fieldMetadata);

        // Process NTSC specific data if source type is NTSC
        if (videoParameters.system == NTSC) {
            // Get the 40-bit FM coded data from field line 10
            FmCode fmCode;
            fmCode.decodeLine(getFieldLine(sourceFieldData, 10, videoParameters), videoParameters, fieldMetadata);

            // Get the white flag from field line 11
            WhiteFlag whiteFlag;
            whiteFlag.decodeLine(getFieldLine(sourceFieldData, 11, videoParameters), videoParameters, fieldMetadata);

            fieldMetadata.ntsc.inUse = true;
        }

        // Get VITC data, trying each possible line and stopping when we find a valid one
        VitcCode vitcCode;
        for (qint32 lineNumber: vitcCode.getLineNumbers(videoParameters)) {
            if (vitcCode.decodeLine(getFieldLine(sourceFieldData, lineNumber, videoParameters),
                                    videoParameters, fieldMetadata)) {
                break;
            }
        }

        // Get Closed Caption data from line 21 (525-line) or 22 (625-line)
        ClosedCaption closedCaption;
        closedCaption.decodeLine(getFieldLine(sourceFieldData, (videoParameters.system == PAL) ? 22 : 21, videoParameters),
                                 videoParameters, fieldMetadata);

        // Write the result to the output metadata
        if (!decoderPool.setOutputField(fieldNumber, fieldMetadata)) {
            abort = true;
            break;
        }
    }
}

// Private method to get a single scanline of greyscale data
SourceVideo::Data VbiLineDecoder::getFieldLine(const SourceVideo::Data &sourceField, qint32 fieldLine,
                                               const LdDecodeMetaData::VideoParameters& videoParameters)
{
    // Range-check the field line
    if (fieldLine < startFieldLine || fieldLine > endFieldLine) {
        qWarning() << "Cannot generate field-line data, line number is out of bounds! Scan line =" << fieldLine;
        return SourceVideo::Data();
    }

    qint32 startPointer = (fieldLine - startFieldLine) * videoParameters.fieldWidth;
    return sourceField.mid(startPointer, videoParameters.fieldWidth);
}
