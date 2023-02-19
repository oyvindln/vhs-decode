/*******************************************************************************
 * QPSKFramer.hpp
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


struct QPSKFrame {
    int frameNumber;
    uint8_t bytes[37];
};


template<class DATA_SRC>
struct QPSKFramer {
    explicit QPSKFramer(DATA_SRC &source) : source(source) {}

    DATA_SRC &source;

    int syncFrameSymbolsSeen = 0;
    char syncFrameNo[4]{};
    int symbolInFrameCounter = 0;
    char symbolsInFrame[37 * 4]{0};
    int consecutiveSynced = 0;
    int autoSyncAt = -1;
    int prevFrameNo = 0;

    int index = -1;

    int n_frames = 0;

    // arrangeInFrames(symbols)
    // searches for the pattern 0113xxxx0000 in the symbols, then constructs and outputs a qpsk frame
    QPSKFrame next() {
        while (true) {
            index++;
            auto symbol = source.get();
            if (symbol == EOF)
                throw std::range_error("EOF");
            symbol -= 48;

            // fileout << (int) symbol << "\t" << (int) index << "\t" << (int) syncFrameSymbolsSeen << "\n";

            if (syncFrameSymbolsSeen < 12 && autoSyncAt < index) {
                // looking for the pattern 0113 ???? 0000
                bool isNextSyncSymbol;
                if (syncFrameSymbolsSeen < 1) { // NOLINT(bugprone-branch-clone)
                    isNextSyncSymbol = symbol == 0;
                } else if (syncFrameSymbolsSeen < 3) {
                    isNextSyncSymbol = symbol == 1;
                } else if (syncFrameSymbolsSeen < 4) {
                    isNextSyncSymbol = symbol == 3;
                } else if (syncFrameSymbolsSeen < 8) {
                    syncFrameNo[syncFrameSymbolsSeen - 4] = (char) symbol;
                    isNextSyncSymbol = true;
                } else {
                    isNextSyncSymbol = symbol == 0;
                }

                if (isNextSyncSymbol)
                    syncFrameSymbolsSeen += 1;
                else {
                    if (consecutiveSynced > 0) {
                        Logger(WARN, "WARN") << "Missing sync at symbol " << index << " (consecutive="
                                             << consecutiveSynced << ")";
                        consecutiveSynced--;
                        for (int j = 0; j < 4; ++j)
                            syncFrameNo[j] = (char) (((prevFrameNo + 1) % 72) >> (6 - 2 * j) & 3);
                        autoSyncAt = index + 12 - syncFrameSymbolsSeen;
                    } else {
                        Logger(WARN, "SYNC") << "Lost sync at symbol " << index;
                        syncFrameSymbolsSeen = 0;
                    }
                }
            } else if (index >= autoSyncAt) {
                if (syncFrameSymbolsSeen == 12 && symbolInFrameCounter == 0)
                    consecutiveSynced = std::min(consecutiveSynced + 1, 3);
                else if (index == autoSyncAt)
                    syncFrameSymbolsSeen = 12;

                symbolsInFrame[symbolInFrameCounter] = (char) symbol;
                symbolInFrameCounter += 1;
                if (symbolInFrameCounter == 37 * 4) {
                    prevFrameNo = syncFrameNo[0] << 6 | syncFrameNo[1] << 4 | syncFrameNo[2] << 2 | syncFrameNo[3] << 0;

                    QPSKFrame frame{};
                    frame.frameNumber = prevFrameNo;
                    for (int i = 0; i < 37; ++i) {
                        frame.bytes[i] = symbolsInFrame[4 * i + 0] << 6 | symbolsInFrame[4 * i + 1] << 4 |
                                         symbolsInFrame[4 * i + 2] << 2 | symbolsInFrame[4 * i + 3] << 0;
                    }

                    symbolInFrameCounter = 0;
                    syncFrameSymbolsSeen = 0;
                    n_frames++;
                    return frame;
                }
            }
        }
    }
};
