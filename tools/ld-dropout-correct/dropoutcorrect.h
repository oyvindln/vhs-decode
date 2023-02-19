/************************************************************************

    dropoutcorrect.h

    ld-dropout-correct - Dropout correction for ld-decode
    Copyright (C) 2018-2020 Simon Inns
    Copyright (C) 2019-2020 Adam Sampson

    This file is part of ld-decode-tools.

    ld-dropout-correct is free software: you can redistribute it and/or
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

#ifndef DROPOUTCORRECT_H
#define DROPOUTCORRECT_H

#include <QObject>
#include <QElapsedTimer>
#include <QAtomicInt>
#include <QThread>
#include <QDebug>

#include "sourcevideo.h"
#include "lddecodemetadata.h"

class CorrectorPool;

class DropOutCorrect : public QThread
{
    Q_OBJECT
public:
    explicit DropOutCorrect(QAtomicInt& _abort, CorrectorPool& _correctorPool, QObject *parent = nullptr);

protected:
    void run() override;

private:
    enum Location {
        visibleLine,
        colourBurst,
        unknown
    };

    struct DropOutLocation {
        qint32 fieldLine;
        qint32 startx;
        qint32 endx;
        Location location;
    };

    struct Replacement {
        // The default value is no replacement
        Replacement() : isSameField(true), fieldLine(-1) {}

        bool isSameField;
        qint32 fieldLine;

        qint32 sourceNumber;
        double quality;

        qint32 distance;
    };

    // Statistics
    struct Statistics {
        qint32 sameSourceConcealment;
        qint32 multiSourceConcealment;
        qint32 multiSourceCorrection;
        qint32 totalReplacementDistance;
    };

    // Decoder pool
    QAtomicInt& abort;
    CorrectorPool& correctorPool;

    QVector<LdDecodeMetaData::VideoParameters> videoParameters;

    void correctField(const QVector<QVector<DropOutLocation> > &thisFieldDropouts,
                      const QVector<QVector<DropOutLocation> > &otherFieldDropouts,
                      QVector<SourceVideo::Data> &thisFieldData, const QVector<SourceVideo::Data> &otherFieldData,
                      bool thisFieldIsFirst, bool intraField, const QVector<qint32> &availableSourcesForFrame,
                      const QVector<double> &sourceFrameQuality, Statistics &statistics);
    QVector<DropOutLocation> populateDropoutsVector(LdDecodeMetaData::Field field, bool overCorrect);
    QVector<DropOutLocation> setDropOutLocations(QVector<DropOutLocation> dropOuts);
    Replacement findReplacementLine(const QVector<QVector<DropOutLocation>> &thisFieldDropouts,
                                    const QVector<QVector<DropOutLocation>> &otherFieldDropouts,
                                    qint32 dropOutIndex, bool thisFieldIsFirst, bool matchChromaPhase,
                                    bool isColourBurst, bool intraField, const QVector<qint32> &availableSourcesForFrame,
                                    const QVector<double> &sourceFrameQuality);
    void findPotentialReplacementLine(const QVector<QVector<DropOutLocation>> &targetDropouts, qint32 targetIndex,
                                      const QVector<QVector<DropOutLocation>> &sourceDropouts, bool isSameField,
                                      qint32 sourceOffset, qint32 stepAmount,
                                      qint32 sourceNo, const QVector<double> &sourceFrameQuality,
                                      QVector<Replacement> &candidates);
    void correctDropOut(const DropOutLocation &dropOut,
                        const Replacement &replacement, const Replacement &chromaReplacement,
                        QVector<SourceVideo::Data> &thisFieldData, const QVector<SourceVideo::Data> &otherFieldData,
                        Statistics &statistics);
};

#endif // DROPOUTCORRECT_H
