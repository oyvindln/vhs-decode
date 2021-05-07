/************************************************************************

    tbcsource.h

    ld-analyse - TBC output analysis
    Copyright (C) 2018-2021 Simon Inns

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

#ifndef TBCSOURCE_H
#define TBCSOURCE_H

#include <QObject>
#include <QImage>
#include <QPainter>
#include <QtConcurrent/QtConcurrent>
#include <QDebug>

// TBC library includes
#include "sourcevideo.h"
#include "lddecodemetadata.h"
#include "vbidecoder.h"
#include "filters.h"

// Chroma decoder includes
#include "configuration.h"
#include "palcolour.h"
#include "comb.h"

class TbcSource : public QObject
{
    Q_OBJECT
public:
    explicit TbcSource(QObject *parent = nullptr);

    struct ScanLineData {
        QVector<qint32> data;
        QVector<bool> isDropout;
        qint32 blackIre;
        qint32 whiteIre;
        qint32 colourBurstStart;
        qint32 colourBurstEnd;
        qint32 activeVideoStart;
        qint32 activeVideoEnd;
        bool isSourcePal;
    };

    void loadSource(QString inputFileName);
    void unloadSource();
    bool getIsSourceLoaded();
    QString getCurrentSourceFilename();

    void setHighlightDropouts(bool _state);
    void setChromaDecoder(bool _state);
    void setFieldOrder(bool _state);
    bool getHighlightDropouts();
    bool getChromaDecoder();
    bool getFieldOrder();

    QImage getFrameImage(qint32 frameNumber);
    qint32 getNumberOfFrames();
    qint32 getNumberOfFields();
    bool getIsSourcePal();
    qint32 getFrameHeight();
    qint32 getFrameWidth();

    VbiDecoder::Vbi getFrameVbi(qint32 frameNumber);
    bool getIsFrameVbiValid(qint32 frameNumber);

    QVector<qreal> getBlackSnrGraphData();
    QVector<qreal> getWhiteSnrGraphData();
    QVector<qreal> getDropOutGraphData();
    qint32 getGraphDataSize();

    bool getIsDropoutPresent(qint32 frameNumber);
    ScanLineData getScanLineData(qint32 frameNumber, qint32 scanLine);

    qint32 getFirstFieldNumber(qint32 frameNumber);
    qint32 getSecondFieldNumber(qint32 frameNumber);

    qint32 getCcData0(qint32 frameNumber);
    qint32 getCcData1(qint32 frameNumber);

    void setChromaConfiguration(const PalColour::Configuration &palConfiguration, const Comb::Configuration &ntscConfiguration,
                                const OutputWriter::Configuration &outputConfiguration);
    const PalColour::Configuration &getPalConfiguration();
    const Comb::Configuration &getNtscConfiguration();
    const OutputWriter::Configuration &getOutputConfiguration();

    qint32 startOfNextChapter(qint32 currentFrameNumber);
    qint32 startOfChapter(qint32 currentFrameNumber);

signals:
    void busyLoading(QString information);
    void finishedLoading();

private slots:
    void finishBackgroundLoad();

private:
    bool sourceReady;

    // Frame data
    QVector<qreal> blackSnrGraphData;
    QVector<qreal> whiteSnrGraphData;
    QVector<qreal> dropoutGraphData;

    // Frame image options
    bool chromaOn;
    bool dropoutsOn;
    bool reverseFoOn;

    // Source globals
    SourceVideo sourceVideo;
    LdDecodeMetaData ldDecodeMetaData;
    QString currentSourceFilename;
    QString lastLoadError;

    // Chroma decoder objects
    PalColour palColour;
    Comb ntscColour;
    OutputWriter outputWriter;

    // VBI decoder
    VbiDecoder vbiDecoder;

    // Background loader globals
    QFutureWatcher<void> watcher;
    QFuture <void> future;

    // Cache of current frame QImage
    QImage frameCache;
    qint32 frameCacheFrameNumber;

    // Chroma decoder configuration
    PalColour::Configuration palConfiguration;
    Comb::Configuration ntscConfiguration;
    OutputWriter::Configuration outputConfiguration;
    bool decoderConfigurationChanged;

    // Chapter map
    QVector<qint32> chapterMap;

    QImage generateQImage(qint32 frameNumber);
    void generateData();
    void startBackgroundLoad(QString sourceFilename);
};

#endif // TBCSOURCE_H
