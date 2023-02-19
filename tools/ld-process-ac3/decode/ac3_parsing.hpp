/*******************************************************************************
 * syncframe.hpp
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

// Note that section/page numbers refer to "ATSC Standard: Digital Audio Compression (AC-3, E-AC-3)"
// https://www.atsc.org/wp-content/uploads/2015/03/A52-201212-17.pdf


#pragma once

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <vector>


const double fscod_lookup[]{48, 44.1, 32};


const int acmod_chans_lookup[]{2, 1, 2, 3, 3, 4, 4, 5};

// Section 7.10.1 in crc checking, Pg 103. Offsets indicate polynomial powers; 16, 15, 2 and 0.
constexpr int AC3_POLY = 0x18005;


// all bitstream elements arrive most significant (or left) bit first
struct bitbuffer {
    uint8_t *bufStart;
    uint8_t *bufEnd;
    int pos = 0;

    explicit bitbuffer(uint8_t *bufStart, uint8_t *bufEnd) : bufStart(bufStart), bufEnd(bufEnd) {}

    //http://osteras.info/personal/2014/10/27/parse-bitstream.html
    uint32_t get(uint8_t len) { // NOLINT(misc-no-recursion)
        size_t byte_offset = pos / 8;
        uint8_t bit_pos = pos % 8;
        uint8_t bit_avail = 8 - bit_pos;

        if (bufStart + byte_offset >= bufEnd)
            throw std::out_of_range("No more bits"); // todo; trap appropriately

        if (bit_avail >= len) {
            uint32_t value = bufStart[byte_offset];
            value &= 0xff >> bit_pos; // remove everything to the left of the bits we need
            value >>= (bit_avail - len);
            pos += len;
            return value;
        } else {
            uint32_t value = get(bit_avail);
            value = value << (len - bit_avail);
            value = value | get(len - bit_avail);
            return value;
        }
    }
};


// Exception thrown when constructing a SyncFrame from invalid data
class InvalidFrameError : public std::runtime_error
{
public:
    InvalidFrameError(std::string message) : std::runtime_error(message) {}
};


struct SyncInfo {
    uint16_t syncword; // 0x0B77
    uint16_t crc1; // crc for first 5/8ths of the block
    uint8_t fscod; // sampling frequency code
    uint8_t frmsizecod; // frame size code

    explicit SyncInfo(bitbuffer &source) {
        syncword = source.get(16);
        if (syncword != 0x0B77) throw InvalidFrameError("invalid syncword");
        crc1 = source.get(16);
        fscod = source.get(2);
        // code 3 unused
        if (fscod == 0b11) throw InvalidFrameError("invalid fscod");
        frmsizecod = source.get(6);
        // max frmsizecod is 36
        if (frmsizecod > 36) throw InvalidFrameError("invalid frmsizcod");
        // used to determine the number of 16-bit words before the next sync word
    }
};


struct BitStreamInformation {
    uint8_t bsid;       // bit stream identification
    uint8_t bsmod;      // bit stream mode
    uint8_t acmod;      // audio coding mode
    uint8_t cmixlev;    // center mix level
    uint8_t surmixlev;  // surround mix level
    uint8_t dsurmod;    // dolby surround mode
    uint8_t lfeon;      // low frequency effects on
    uint8_t dialnorm;   // dialogue normalization word
    uint8_t compre;     // compression gain word exists
    uint8_t compr;      // compression gain word
    uint8_t langcode;   // language code exists
    uint8_t langcod;    // language code
    uint8_t audprodie;  // audio production information exists
    uint8_t mixlevel;   // mixing level
    uint8_t roomtyp;    // room type

    uint8_t dialnorm2;  // dialogue normalization word exists, ch2
    uint8_t compr2e;    // compression gain word exists, ch2
    uint8_t compr2;     // compression gain word
    uint8_t langcod2e;  // language code exists, ch2
    uint8_t langcod2;   // language code, ch2
    uint8_t audprodi2e; // audio production information exists, ch2
    uint8_t mixlevel2;  // mixing level, ch2
    uint8_t roomtyp2;   // room type, ch2
    uint8_t copyrightb; // copyright bit
    uint8_t origbs;     // origional bit stream
    uint8_t timecod1e;  // time code first half exists
    uint8_t timecod1;   // time code first half
    uint8_t timecod2e;  // time code second half exists
    uint8_t timecod2;   // time code second half
    uint8_t addbsie;    // additional bit stream information exists
    uint8_t addbsil;    // additional bit stream information length
    uint8_t addbsi;     // additional bit stream information


    explicit BitStreamInformation(bitbuffer &source) {
        bsid = source.get(5);
        bsmod = source.get(3);
        acmod = source.get(3);

        if ((acmod & 0x1) && (acmod != 0x1))// if 3 front channels
            cmixlev = source.get(2);
        if (acmod & 0x4)// if a surround channel exists
            surmixlev = source.get(2);
        if (acmod == 0x2) // if in 2/0 mode
            dsurmod = source.get(2);
        lfeon = source.get(1);
        dialnorm = source.get(5); // 1-31

        compre = source.get(1);
        if (compre)
            compr = source.get(8); //
        langcode = source.get(1);
        if (langcode)
            langcod = source.get(8); // needs a lookup table, missing in recent documentation
        audprodie = source.get(1);
        if (audprodie) {
            mixlevel = source.get(5); // 0-31
            roomtyp = source.get(2);
        }
        if (acmod == 0) {
            dialnorm2 = source.get(5);
            compr2e = source.get(1);
            if (compr2e)
                compr2 = source.get(8);
            langcod2e = source.get(1);
            if (langcod2e)
                langcod2 = source.get(8);
            audprodi2e = source.get(1);
            if (audprodi2e) {
                mixlevel2 = source.get(5);
                roomtyp2 = source.get(2);
            }
        }
        copyrightb = source.get(1);
        origbs = source.get(1);
        timecod1e = source.get(1);
        if (timecod1e)
            timecod1 = source.get(14);
        timecod2e = source.get(1);
        if (timecod2e)
            timecod2 = source.get(14);
        addbsie = source.get(1);
        if (addbsie) {
            addbsil = source.get(6);
            addbsi = source.get((addbsil + 1) * 8);
        }
    }
};


struct SyncFrame {
    std::vector<uint8_t> &frameData;
    bitbuffer bs;
    SyncInfo syncInfo;
    BitStreamInformation bsi;

    // Constructor. May throw InvalidFrameError.
    explicit SyncFrame(std::vector<uint8_t> &f) : frameData(f),
                                                  bs(f.data(), f.data() + f.size()),
                                                  syncInfo(bs), bsi(bs) {}

    // See page 106, 7.10.2 Checking Bit Stream Consistency
    // these checks almost entirely apply to audio blocks, which are not unpacked here, so cant be checked.

    // Check both CRCs in the frame are valid.
    // May throw InvalidFrameError.
    uint8_t check_crc() {
        int frameSize, frameSize58;
        uint8_t *frame = frameData.data();

        if (syncInfo.frmsizecod != 28) throw InvalidFrameError("invalid frmsizecod");
        if (syncInfo.fscod != 0b00) throw InvalidFrameError("invalid fscod");

        frameSize = 768; // frame size from frmsizecod & fscod. todo: lookup / calculate
        frameSize58 = (frameSize >> 1) + (frameSize >> 3); // 1/8 + 4/8

        // CRC1 covers first 5/8ths of the frame.
        bool pass1 = calc_crc16(frame + 2, (frameSize58 << 1) - 2);
        // CRC2 covers last 3/8ths of the frame. Mostly useless if crc1 failed.
        bool pass2 = calc_crc16(frame + (frameSize58 << 1), ((frameSize - frameSize58) << 1));

        return pass1 | (pass2 << 1);
    }

private:
    static std::array<uint16_t, 256> crcLookup;

    static bool calc_crc16(const uint8_t *data, uint32_t len) {
        union {
            uint16_t word;
            uint8_t bytes[2];
        } crc{};

        // See LFSR diagram, page 104
        uint16_t a, b;
        while (len--) {
            a = crc.bytes[1];
            b = (crc.word & 255) ^ *data++; // xor & increment pointer
            crc.word = a ^ crcLookup[b]; // precalculated values
        }
        return crc.word == 0; // should be zero if crc worked.
    }
};
