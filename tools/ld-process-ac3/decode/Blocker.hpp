/*******************************************************************************
 * Blocker.hpp
 *
 * ld-process-ac3 - AC3-RF decoder
 * Copyright (C) 2022-2022 Leighton Smallshire & Ian Smallshire
 *
 * Derived from prior work by Staffan Ulfberg with feedback
 * to original author. (Copyright (C) 2021-2022)
 * https://bitbucket.org/staffanulfberg/ldaudio/src/master/
 *
 * This file is part of ld-decode-tools.
 *
 * ld-process-ac3 is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 ******************************************************************************/

#pragma once

#include "QPSKFramer.hpp"
#include "cstring"


struct QPSKBlock {
    uint8_t bytes[72 * 37];
};


template<class DATA_SRC>
struct Blocker {
    explicit Blocker(DATA_SRC &source) : source(source) {}

    DATA_SRC &source;
    bool initialized = false;

    QPSKBlock currentBlock{};
    int framesConsumed = 0;
    int expectedSeq = 0;
    int consecutiveInSequence = 0;

    // collects QPSK frames into blocks. Must be numbered 0 to 71 and in order (with some tolerance)
    QPSKBlock next() {
        while (true) {
            QPSKFrame frame = source.next();

            // todo move outwards / to a real initializer. requires buffer/seek/peek on previous stage
            // Drop frames from the start until frameNumber is zero
            // could be moved to constructor, but then previous stage needs a putback() method & buffer
            while (!initialized) {
                if (frame.frameNumber == 0)
                    break;
                frame = source.next();
            }
            initialized = true;

            int usedFrameNo;
            if (frame.frameNumber != expectedSeq) { // frame arrived out-of-order / mislabeled
                // todo; (from scala) this throws out WAY too much data.
                // if (consecutiveInSequence < 3) { // within first 3 of block
                //     framesConsumed = 0; // clear block
                //     consecutiveInSequence = 0; // no longer consecutive correct sequence numbers
                //     usedFrameNo = -1; // used frame '-1'
                // } else { // after first 3 of block
                consecutiveInSequence = 0; // no longer consecutive correct sequence numbers
                usedFrameNo = expectedSeq; // pretend the frame has the expected number
                // }
            } else {
                consecutiveInSequence++;
                usedFrameNo = frame.frameNumber;
            }

            // copy from frame to block.
            if (framesConsumed < 72)
                std::memcpy(currentBlock.bytes + 37 * framesConsumed, frame.bytes, 37);
            framesConsumed++;

            if (usedFrameNo == 71 && framesConsumed == 72) {
                expectedSeq = 0;
                framesConsumed = 0;
                return currentBlock;
            } else {
                expectedSeq = usedFrameNo + 1;
            }
        }
    }
};
