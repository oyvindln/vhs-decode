/************************************************************************

    dataconverter.cpp

    ld-lds-converter - 10-bit to 16-bit .lds converter for ld-decode
    Copyright (C) 2019 Simon Inns

    This file is part of ld-decode-tools.

    ld-lds-converter is free software: you can redistribute it and/or
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

#include "dataconverter.h"

DataConverter::DataConverter(QString inputFileNameParam, QString outputFileNameParam, bool isPackingParam, QObject *parent) : QObject(parent)
{
    // Store the configuration parameters
    inputFileName = inputFileNameParam;
    outputFileName = outputFileNameParam;
    isPacking = isPackingParam;
}

// Method to process the conversion of the file
bool DataConverter::process(void)
{
    // Open the input file
    if (!openInputFile()) {
        qCritical("Could not open input file!");
        return false;
    }

    // Open the output file
    if (!openOutputFile()) {
        qCritical("Could not open output file!");
        return false;
    }

    // Packing or unpacking?
    if (isPacking) packFile();
    else unpackFile();

    // Close the input file
    closeInputFile();

    // Close the output file
    closeOutputFile();

    // Exit with success
    return true;
}

// Method to open the input file for reading
bool DataConverter::openInputFile(void)
{
    // Do we have a file name for the input file?
    if (inputFileName.isEmpty()) {
        // No source input file name was specified, using stdin instead
        qDebug() << "No input filename was provided, using stdin";
        inputFileHandle = new QFile;
        if (!inputFileHandle->open(stdin, QIODevice::ReadOnly)) {
            // Failed to open stdin
            qWarning() << "Could not open stdin as input file";
            return false;
        }
        qDebug() << "Reading input data from stdin";
    } else {
        // Open input file for reading
        inputFileHandle = new QFile(inputFileName);
        if (!inputFileHandle->open(QIODevice::ReadOnly)) {
            // Failed to open source sample file
            qDebug() << "Could not open" << inputFileName << "as input file";
            return false;
        }
        qDebug() << "DataConverter::openInputFile(): Input file is" << inputFileName << "and is" << inputFileHandle->size() << "bytes in length";
    }

    // Exit with success
    return true;
}

// Method to close the input file
void DataConverter::closeInputFile(void)
{
    // Is an input file open?
    if (inputFileHandle != nullptr) {
        inputFileHandle->close();
    }

    // Clear the file handle pointer
    delete inputFileHandle;
    inputFileHandle = nullptr;
}

// Method to open the output file for writing
bool DataConverter::openOutputFile(void)
{
    // Do we have a file name for the output file?
    if (outputFileName.isEmpty()) {
        // No output file name was specified, using stdin instead
        qDebug() << "No output filename was provided, using stdout";
        outputFileHandle = new QFile;
        if (!outputFileHandle->open(stdout, QIODevice::WriteOnly)) {
            // Failed to open stdout
            qWarning() << "Could not open stdout as output file";
            return false;
        }
        qDebug() << "Writing output data to stdout";
    } else {
        // Open the output file for writing
        outputFileHandle = new QFile(outputFileName);
        if (!outputFileHandle->open(QIODevice::WriteOnly)) {
            // Failed to open output file
            qDebug() << "DataConverter::openOutputFile(): Could not open" << outputFileName << "as output file";
            return false;
        }
        qDebug() << "DataConverter::openOutputFile(): Output file is" << outputFileName;
    }

    // Exit with success
    return true;
}

// Method to close the output file
void DataConverter::closeOutputFile(void)
{
    // Is an output file open?
    if (outputFileHandle != nullptr) {
        outputFileHandle->close();
    }

    // Clear the file handle pointer
    delete outputFileHandle;
    outputFileHandle = nullptr;
}

// Method to pack 16-bit data into 10-bit data
void DataConverter::packFile(void)
{
    qDebug() << "DataConverter::packFile(): Packing";
    QByteArray inputBuffer;
    QByteArray outputBuffer;
    bool isComplete = false;

    while(!isComplete) {
        // Input buffer must be divisible by 5 bytes due to 10-bit data format
        qint32 bufferSizeInBytes = (20 * 1024 * 1024); // = 20MiBytes
        inputBuffer.resize(bufferSizeInBytes);

        // Every 4 input words (8 bytes) is 5 output bytes
        outputBuffer.resize((bufferSizeInBytes / 8) * 5);

        // Fill the input buffer with data
        qint64 receivedBytes = 0;
        qint32 totalReceivedBytes = 0;
        do {
            receivedBytes = inputFileHandle->read(reinterpret_cast<char *>(inputBuffer.data() + totalReceivedBytes), inputBuffer.size() - totalReceivedBytes);
            if (receivedBytes > 0) totalReceivedBytes += receivedBytes;
        } while (receivedBytes > 0 && totalReceivedBytes < bufferSizeInBytes);

        // Check for end of file
        if (receivedBytes == 0) isComplete = true;

        if (totalReceivedBytes != 0) {
            // If we didn't fill the input buffer, resize it
            if (bufferSizeInBytes != totalReceivedBytes) {
                inputBuffer.resize(totalReceivedBytes);
                outputBuffer.resize((totalReceivedBytes / 8) * 5);
            }
            qDebug() << "DataConverter::packFile(): Got" << totalReceivedBytes << "bytes from input file";

            qint32 word0, word1, word2, word3;
            qint32 outputBufferPointer = 0;

            qint16 *input = reinterpret_cast<qint16 *>(inputBuffer.data());

            for (qint32 wordPointer = 0; wordPointer < (totalReceivedBytes / 2); wordPointer += 4) {

                word0 = (input[wordPointer + 0] / 64) + 512;
                word1 = (input[wordPointer + 1] / 64) + 512;
                word2 = (input[wordPointer + 2] / 64) + 512;
                word3 = (input[wordPointer + 3] / 64) + 512;

                outputBuffer[outputBufferPointer + 0]  = static_cast<char>((word0 & 0x03FC) >> 2);
                outputBuffer[outputBufferPointer + 1]  = static_cast<char>(((word0 & 0x0003) << 6) + ((word1 & 0x03F0) >> 4));
                outputBuffer[outputBufferPointer + 2]  = static_cast<char>(((word1 & 0x000F) << 4) + ((word2 & 0x03C0) >> 6));
                outputBuffer[outputBufferPointer + 3]  = static_cast<char>(((word2 & 0x003F) << 2) + ((word3 & 0x0300) >> 8));
                outputBuffer[outputBufferPointer + 4]  = static_cast<char>(word3 & 0x00FF);

                // Increment the packed sample buffer pointer
                outputBufferPointer += 5;
            }

            // Write the output buffer to the output file
            if (!outputFileHandle->write(reinterpret_cast<char *>(outputBuffer.data()),
                                         outputBuffer.size())) {
                // File write failed
                qCritical("Could not write to output file!");
            }
            qDebug() << "DataConverter::packFile(): Wrote" << outputBuffer.size() << "bytes to output file";
        } else {
            // Input file is empty
            qDebug() << "DataConverter::packFile(): Got zero bytes from input file";
            isComplete = true;
        }
    }
}

// Method to unpack 10-bit data into 16-bit data
void DataConverter::unpackFile(void)
{
    qDebug() << "DataConversion::unpackFile(): Unpacking";
    QByteArray inputBuffer;
    QByteArray outputBuffer;
    bool isComplete = false;

    while(!isComplete) {
        // Input buffer must be divisible by 5 bytes due to 10-bit data format
        qint32 bufferSizeInBytes = (5 * 1024 * 1024) * 4; // 5MiB * 4 = 20MiBytes
        inputBuffer.fill(0, bufferSizeInBytes);

        // Every 5 input bytes is 4 output words (8 bytes)
        outputBuffer.resize((bufferSizeInBytes / 5) * 8);

        // Fill the input buffer with data
        qint64 receivedBytes = 0;
        qint32 totalReceivedBytes = 0;
        do {
            receivedBytes = inputFileHandle->read(reinterpret_cast<char *>(inputBuffer.data() + totalReceivedBytes), bufferSizeInBytes - totalReceivedBytes);
            if (receivedBytes > 0) totalReceivedBytes += receivedBytes;
        } while (receivedBytes > 0 && totalReceivedBytes < bufferSizeInBytes);

        // Check for end of file
        if (receivedBytes == 0) isComplete = true;

        if (totalReceivedBytes != 0) {
            // If we didn't fill the input buffer, resize it
            if (bufferSizeInBytes != totalReceivedBytes) {
                inputBuffer.resize(totalReceivedBytes);
                outputBuffer.resize((totalReceivedBytes / 5) * 8);
            }
            qDebug() << "DataConverter::unpackFile(): Got" << totalReceivedBytes << "bytes from input file";

            char byte0, byte1, byte2, byte3, byte4;
            qint32 word0, word1, word2, word3;
            qint32 outputBufferPointer = 0;

            qint16 *output = reinterpret_cast<qint16 *>(outputBuffer.data());

            for (qint32 bytePointer = 0; bytePointer < totalReceivedBytes; bytePointer += 5) {
                // Unpack the 5 bytes into 4x 10-bit values

                // Unpacked:                 Packed:
                // 0: xxxx xx00 0000 0000    0: 0000 0000 0011 1111
                // 1: xxxx xx11 1111 1111    2: 1111 2222 2222 2233
                // 2: xxxx xx22 2222 2222    4: 3333 3333
                // 3: xxxx xx33 3333 3333

                byte0 = inputBuffer[bytePointer + 0];
                byte1 = inputBuffer[bytePointer + 1];
                byte2 = inputBuffer[bytePointer + 2];
                byte3 = inputBuffer[bytePointer + 3];
                byte4 = inputBuffer[bytePointer + 4];

                // Use multiplication instead of left-shift to avoid implicit conversion issues
                word0  = ((byte0 & 0xFF) *   4) + ((byte1 & 0xC0) >> 6);
                word1  = ((byte1 & 0x3F) *  16) + ((byte2 & 0xF0) >> 4);
                word2  = ((byte2 & 0x0F) *  64) + ((byte3 & 0xFC) >> 2);
                word3  = ((byte3 & 0x03) * 256) + ((byte4 & 0xFF)     );

                output[outputBufferPointer + 0] = static_cast<qint16>((word0 - 512) * 64);
                output[outputBufferPointer + 1] = static_cast<qint16>((word1 - 512) * 64);
                output[outputBufferPointer + 2] = static_cast<qint16>((word2 - 512) * 64);
                output[outputBufferPointer + 3] = static_cast<qint16>((word3 - 512) * 64);

                // Increment the sample buffer pointer
                outputBufferPointer += 4;
            }

            // Write the output buffer to the output file
            if (!outputFileHandle->write(reinterpret_cast<char *>(outputBuffer.data()),
                                         outputBuffer.size())) {
                // File write failed
                qCritical("Could not write to output file!");
            }
            qDebug() << "DataConverter::unpackFile(): Wrote" << outputBuffer.size() << "bytes to output file";
        } else {
            // Input file is empty
            qDebug() << "DataConverter::unpackFile(): Got zero bytes from input file";
            isComplete = true;
        }
    }
}
