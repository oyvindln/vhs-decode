/************************************************************************

    closedcaptionsdialog.cpp

    ld-analyse - TBC output analysis
    Copyright (C) 2018-2022 Simon Inns

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

#include "closedcaptionsdialog.h"
#include "ui_closedcaptionsdialog.h"

ClosedCaptionsDialog::ClosedCaptionsDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::ClosedCaptionsDialog)
{
    ui->setupUi(this);

    // Set flag to indicate waiting for preamble control code
    waitingForPreamble = true;

    // Set the last frame number to invalid
    lastFrameNumber = -1;

    lastNonDisplayCommand = -1;
    lastDisplayCommand = -1;
}

ClosedCaptionsDialog::~ClosedCaptionsDialog()
{
    delete ui;
}

void ClosedCaptionsDialog::addData(qint32 frameNumber, qint32 data0, qint32 data1)
{
    qDebug() << "ClosedCaptionsDialog::addData: Frame number" << frameNumber << "data0 =" << data0 << "data1 =" << data1;

    // Check that we have a continuous stream of frames
    if (frameNumber == lastFrameNumber) return;
    if (frameNumber != (lastFrameNumber+1)) resetCaptions();
    lastFrameNumber = frameNumber;

    // Check incoming data is valid
    if (data0 == -1 || data1 == -1) return;

    // Check for a non-display control code
    if (data0 >= 0x10 && data0 <= 0x1F) {
        if (data0 == lastNonDisplayCommand && data1 == lastDisplayCommand) {
            // This is a command repeat; ignore
        } else {
            qDebug() << "ClosedCaptionsDialog::addData(): Got non-display control code of" << data0;
            processCommand(data0, data1);
            lastNonDisplayCommand = data0;
            lastDisplayCommand = data1;
        }
    } else {
        // Normal text - display

        // Create a string from the two characters
        char string[3];
        string[0] = static_cast<char>(data0);
        string[1] = static_cast<char>(data1);
        string[2] = static_cast<char>(0);

        // Display in the text edit widget
        ui->captionTextEdit->moveCursor (QTextCursor::End);
        ui->captionTextEdit->insertPlainText(QString::fromLocal8Bit(string));
        ui->captionTextEdit->moveCursor (QTextCursor::End);

        lastNonDisplayCommand = -1;
        lastDisplayCommand = -1;
    }
}

void ClosedCaptionsDialog::processCommand(qint32 data0, qint32 data1)
{
    // Verify display control code
    if (data1 >= 0x20 && data1 <= 0x7F) {

        // Check for miscellaneous control codes (indicated by data0 & 01110110 == 00010100)
        if ((data0 & 0x76) == 0x14) {
            // Miscellaneous
            //qint32 channel = (data0 & 0x08) >> 3; // 0b00001000 >> 3
            qint32 commandGroup = (data0 & 0x02) >> 1; // 0b00000010 >> 1
            //qint32 fieldLine = (data0 & 0x01); // 0b00000001 : 1 = 284, 0 = 21
            qint32 commandType = (data1 & 0x0F); // 0b00001111

            if (commandGroup == 0) {
                // Normal command
                switch (commandType) {
                case 0:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Resume caption loading";
                    ui->captionTextEdit->moveCursor (QTextCursor::End);
                    ui->captionTextEdit->insertPlainText(" ");
                    ui->captionTextEdit->moveCursor (QTextCursor::End);
                    break;
                case 1:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Backspace";
                    break;
                case 2:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Reserved 1";
                    ui->captionTextEdit->moveCursor (QTextCursor::End);
                    ui->captionTextEdit->insertPlainText(" ");
                    ui->captionTextEdit->moveCursor (QTextCursor::End);
                    break;
                case 3:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Reserved 2";
                    break;
                case 4:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Delete to end of row";
                    ui->captionTextEdit->moveCursor (QTextCursor::End);
                    ui->captionTextEdit->insertPlainText(" ");
                    ui->captionTextEdit->moveCursor (QTextCursor::End);
                    break;
                case 5:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Roll-up captions, 2 rows";
                    break;
                case 6:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Roll-up captions, 3 rows";
                    break;
                case 7:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Roll-up captions, 4 rows";
                    break;
                case 8:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Flash on";
                    break;
                case 9:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Resume direct captioning";
                    break;
                case 10:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Text restart";
                    break;
                case 11:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Resume text display";
                    break;
                case 12:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Erase displayed memory";
                    break;
                case 13:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Carriage return";
                    break;
                case 14:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Erase non-displayed memory";
                    break;
                case 15:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - End of caption (flip memories)";
                    ui->captionTextEdit->append("");
                    break;
                }
            } else {
                // Tab offset command
                switch (commandType) {
                case 1:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Tab offset (1 column)";
                    break;
                case 2:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Tab offset (2 columns)";
                    break;
                case 3:
                    qDebug() << "ClosedCaptionsDialog::processCommand(): Miscellaneous command - Tab offset (3 columns)";
                    break;
                }
            }

            // Done
            return;
        }

        // Check for midrow command code (indicated by data0 & 01110111 == 00010001)
        if ((data0 & 0x77) == 0x11) {
            qDebug() << "ClosedCaptionsDialog::processCommand(): Midrow command";

            // Done
            return;
        }


        // Preamble address code
        // Mid-row control code
        // White non-underlined is default

        // The italics control code does not change the colour
        // A colour control code turns off italics and flashing

        //qint32 dataChannel =       (data0 & 0x08) >> 3; // 0b00001000 >> 3;
        //qint32 nonDisplayCommand = (data0 & 0x07);      // 0b00000111 >> 0;

        //qint32 displayCommand =    (data1 & 0x60) >> 5; // 0b01100000 >> 5;
        //qint32 displayData =       (data1 & 0x1E) >> 1; // 0b00011110;
        //qint32 displayU =          (data1 & 0x01);      // 0b00000001;

    } else {
        qDebug() << "ClosedCaptionsDialog::addData(): Display control code invalid!" << data1;
    }
}

void ClosedCaptionsDialog::resetCaptions()
{
    ui->captionTextEdit->clear();
}
