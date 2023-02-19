/************************************************************************

    whitesnranalysisdialog.h

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

#ifndef WHITESNRANALYSISDIALOG_H
#define WHITESNRANALYSISDIALOG_H

#include <cmath>

#include <QDialog>
#include <qwt_plot.h>
#include <qwt_plot_canvas.h>
#include <qwt_legend.h>
#include <qwt_plot_grid.h>
#include <qwt_plot_curve.h>
#include <qwt_plot_zoomer.h>
#include <qwt_plot_panner.h>
#include <qwt_scale_widget.h>
#include <qwt_scale_draw.h>
#include <qwt_plot_marker.h>

#include "lddecodemetadata.h"

namespace Ui {
class WhiteSnrAnalysisDialog;
}

class WhiteSnrAnalysisDialog : public QDialog
{
    Q_OBJECT

public:
    explicit WhiteSnrAnalysisDialog(QWidget *parent = nullptr);
    ~WhiteSnrAnalysisDialog();

    void startUpdate(qint32 _numberOfFrames);
    void addDataPoint(qint32 frameNumber, double whiteSnr);
    void finishUpdate(qint32 _currentFrameNumber);
    void updateFrameMarker(qint32 _currentFrameNumber);

private slots:
    void scaleDivChangedSlot();

private:
    void removeChartContents();
    void generateTrendLine();

    Ui::WhiteSnrAnalysisDialog *ui;
    QwtPlotZoomer *zoomer;
    QwtPlotPanner *panner;
    QwtPlot *plot;
    QwtLegend *legend;
    QwtPlotGrid *grid;
    QPolygonF *whitePoints;
    QwtPlotCurve *whiteCurve;
    QPolygonF *trendPoints;
    QwtPlotCurve *trendCurve;
    QwtPlotMarker *plotMarker;

    double maxY;
    qint32 numberOfFrames;
    QVector<double> tlPoint;
};

#endif // WHITESNRANALYSISDIALOG_H
