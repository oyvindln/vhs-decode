/************************************************************************

    efmprocess.h

    ld-process-efm - EFM data decoder
    Copyright (C) 2019-2022 Simon Inns

    This file is part of ld-decode-tools.

    ld-process-efm is free software: you can redistribute it and/or
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

#ifndef EFMPROCESS_H
#define EFMPROCESS_H

#include <QString>
#include <QFile>
#include <QDebug>

#include "Decoders/efmtof3frames.h"
#include "Decoders/syncf3frames.h"
#include "Decoders/f3tof2frames.h"
#include "Decoders/f2tof1frames.h"
#include "Decoders/f1toaudio.h"
#include "Decoders/f1todata.h"

class EfmProcess
{
public:
    EfmProcess();

    using ErrorTreatment = F1ToAudio::ErrorTreatment;

    struct Statistics {
        EfmToF3Frames::Statistics efmToF3Frames;
        SyncF3Frames::Statistics syncF3Frames;
        F3ToF2Frames::Statistics f3ToF2Frames;
        F2ToF1Frames::Statistics f2ToF1Frames;
        F1ToAudio::Statistics f1ToAudio;
        F1ToData::Statistics f1ToData;
    };

    void setDebug(bool _debug_efmToF3Frames, bool _debug_syncF3Frames,
                  bool _debug_f3ToF2Frames, bool _debug_f2ToF1Frames,
                  bool _debug_f1ToAudio, bool _debug_f1ToData);
    void setAudioErrorTreatment(ErrorTreatment _errorTreatment);
    void setDecoderOptions(bool _padInitialDiscTime, bool _decodeAsData, bool _audioIsDts, bool _noTimeStamp);
    void reportStatistics() const;
    bool process(QString inputFilename, QString outputFilename);
    Statistics getStatistics();
    void reset();

private:
    // Debug
    bool debug_efmToF3Frames;
    bool debug_f3ToF2Frames;
    bool debug_syncF3Frames;
    bool debug_f2ToF1Frame;
    bool debug_f1ToAudio;
    bool debug_f1ToData;

    // Audio options
    F1ToAudio::ErrorTreatment errorTreatment;
    F1ToAudio::ConcealType concealType;

    // Class globals
    EfmToF3Frames efmToF3Frames;
    SyncF3Frames syncF3Frames;
    F3ToF2Frames f3ToF2Frames;
    F2ToF1Frames f2ToF1Frames;
    F1ToAudio f1ToAudio;
    F1ToData f1ToData;
    bool padInitialDiscTime;
    bool decodeAsAudio;
    bool decodeAsData;
    bool audioIsDts;
    bool noTimeStamp;

    Statistics statistics;
};

#endif // EFMPROCESS_H
