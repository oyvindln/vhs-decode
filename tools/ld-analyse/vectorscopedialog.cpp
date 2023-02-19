/************************************************************************

    vectorscopedialog.cpp

    ld-analyse - TBC output analysis
    Copyright (C) 2018-2022 Simon Inns
    Copyright (C) 2022 Adam Sampson

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

#include "vectorscopedialog.h"
#include "ui_vectorscopedialog.h"

#include <cmath>
#include <random>

#include <QDebug>
#include <QPainter>

VectorscopeDialog::VectorscopeDialog(QWidget *parent) :
    QDialog(parent),
    ui(new Ui::VectorscopeDialog)
{
    ui->setupUi(this);
    setWindowFlags(Qt::Window);
}

VectorscopeDialog::~VectorscopeDialog()
{
    delete ui;
}

void VectorscopeDialog::showTraceImage(const ComponentFrame &componentFrame, const LdDecodeMetaData::VideoParameters &videoParameters)
{
    qDebug() << "VectorscopeDialog::showTraceImage(): Called";

    // Draw the image
    QImage traceImage = getTraceImage(componentFrame, videoParameters);

    // Add the QImage to the QLabel in the dialogue
    ui->scopeLabel->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Expanding);
    ui->scopeLabel->setAlignment(Qt::AlignCenter);
    ui->scopeLabel->setScaledContents(true);
    ui->scopeLabel->setPixmap(QPixmap::fromImage(traceImage));

    // QT Bug workaround for some macOS versions
    #if defined(Q_OS_MACOS)
    	repaint();
    #endif
}

QImage VectorscopeDialog::getTraceImage(const ComponentFrame &componentFrame, const LdDecodeMetaData::VideoParameters &videoParameters)
{
    // Scope size and scale
    constexpr qint32 SIZE = 1024;
    constexpr qint32 SCALE = 65536 / SIZE;
    constexpr qint32 HALF_SIZE = SIZE / 2;

    // Define image with width, height and format
    QImage scopeImage(SIZE, SIZE, QImage::Format_RGB888);
    QPainter scopePainter;

    // Set the background to black
    scopeImage.fill(Qt::black);

    // Attach the scope image to the painter
    scopePainter.begin(&scopeImage);

    // Initialise a cheap, predictable random number generator, for defocussing
    std::minstd_rand randomEngine(12345);
    std::normal_distribution<double> normalDist(0.0, 100.0);

    // For each sample in the active area, plot its U/V values on the chart
    scopePainter.setPen(Qt::green);
    bool defocus = ui->defocusCheckBox->isChecked();
    for (qint32 lineNumber = videoParameters.firstActiveFrameLine; lineNumber < videoParameters.lastActiveFrameLine; lineNumber++) {
        const auto &uLine = componentFrame.u(lineNumber);
        const auto &vLine = componentFrame.v(lineNumber);

        for (qint32 xPosition = videoParameters.activeVideoStart; xPosition < videoParameters.activeVideoEnd; xPosition++) {
            // If defocussing, add a random (but normally-distributed) value to U/V
            double uOffset = defocus ? normalDist(randomEngine) : 0.0;
            double vOffset = defocus ? normalDist(randomEngine) : 0.0;

            // On a real vectorscope, U is positive to the right, and V is positive *upwards*
            qint32 x = HALF_SIZE + (static_cast<qint32>(uLine[xPosition] + uOffset) / SCALE);
            qint32 y = HALF_SIZE - (static_cast<qint32>(vLine[xPosition] + vOffset) / SCALE);

            scopePainter.drawPoint(x, y);
        }
    }

    // Overlay the graticule, unless it's disabled
    if (!ui->graticuleNoneRadioButton->isChecked()) {
        scopePainter.setPen(Qt::white);

        // Draw the vertical/horizontal graticule lines and circle
        scopePainter.drawLine(HALF_SIZE, 0, HALF_SIZE, SIZE - 1);
        scopePainter.drawLine(0, HALF_SIZE, SIZE - 1, HALF_SIZE);
        scopePainter.drawEllipse(0, 0, SIZE - 1, SIZE - 1);

        // For NTSC: draw I/Q graticule lines, 33 degrees offset from the axes
        if (videoParameters.system == NTSC) {
            double theta = (-33.0 * M_PI) / 180;
            for (qint32 i = 0; i < 4; i++) {
                scopePainter.drawLine(HALF_SIZE + (0.2 * HALF_SIZE * cos(theta)),
                                      HALF_SIZE + (0.2 * HALF_SIZE * sin(theta)),
                                      HALF_SIZE + (HALF_SIZE * cos(theta)),
                                      HALF_SIZE + (HALF_SIZE * sin(theta)));
                theta += M_PI / 2.0;
            }
        }

        // Scaling factor for which graticule
        const double percent = ui->graticule75RadioButton->isChecked() ? 0.75 : 1.0;

        // Draw graticule targets for the six colour bars
        for (qint32 rgb = 1; rgb < 7; rgb++) {
            // R'G'B' for this bar
            const double R = percent * static_cast<double>((rgb >> 2) & 1);
            const double G = percent * static_cast<double>((rgb >> 1) & 1);
            const double B = percent * static_cast<double>(rgb & 1);

            // Convert R'G'B' to Y'UV [Poynton p337 eq 28.5]
            const double U = (R * -0.147141) + (G * -0.288869) + (B * 0.436010);
            const double V = (R * 0.614975)  + (G * -0.514965) + (B * -0.100010);

            // Convert to angle and magnitude, scaled to match scope coords
            const double barTheta = atan2(-V, U);
            const double barMag = sqrt((V * V) + (U * U)) * (videoParameters.white16bIre - videoParameters.black16bIre) / SCALE;

            // Draw the target grid, with 10 degree angle and 10% magnitude steps
            const double stepTheta = (10.0 * M_PI) / 180.0;
            const double stepMag = 0.1 * barMag;
            for (qint32 step = -1; step < 2; step++) {
                // XXX These should really be curved lines
                const double theta = barTheta + (step * stepTheta);
                scopePainter.drawLine(HALF_SIZE + ((barMag - stepMag) * cos(theta)), HALF_SIZE + ((barMag - stepMag) * sin(theta)),
                                      HALF_SIZE + ((barMag + stepMag) * cos(theta)), HALF_SIZE + ((barMag + stepMag) * sin(theta)));
            }
            for (qint32 step = -1; step < 2; step++) {
                const double mag = barMag + (step * stepMag);
                scopePainter.drawLine(HALF_SIZE + (mag * cos(barTheta - stepTheta)), HALF_SIZE + (mag * sin(barTheta - stepTheta)),
                                      HALF_SIZE + (mag * cos(barTheta + stepTheta)), HALF_SIZE + (mag * sin(barTheta + stepTheta)));

            }
        }

        // XXX Draw a line for the colourburst -- we don't decode it at the moment
    }

    // Return the QImage
    scopePainter.end();
    return scopeImage;
}

// GUI signal handlers ------------------------------------------------------------------------------------------------

void VectorscopeDialog::on_defocusCheckBox_clicked()
{
    emit scopeChanged();
}

void VectorscopeDialog::on_graticuleButtonGroup_buttonClicked(QAbstractButton *button)
{
    (void) button;
    emit scopeChanged();
}
