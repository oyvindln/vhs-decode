#include "ac3_parsing.hpp"


constexpr std::array<uint16_t, 256> init_table() {
    std::array<uint16_t, 256> table{}; // 256x 16-bit ints.

    constexpr int n_bits = 16;
    int poly = (AC3_POLY + (1 << n_bits));

    for (int i = 0; i < 256; i++) { // for each index:
        uint16_t crc = i;
        for (int j = 0; j < n_bits; j++) { // for each bit:
            if (crc & (1 << (n_bits - 1))) // if highest allowed bit set
                crc = (crc << 1) ^ poly; // left-shift, then xor with poly
            else // otherwise
                crc <<= 1; // left-shift
        }
        table[i] = (crc << 8) | (crc >> 8);
    }
    return table;
}


std::array<uint16_t, 256> SyncFrame::crcLookup = init_table();
