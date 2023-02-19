/*******************************************************************************
 * AC3Filter.hpp
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

// #include <string>
// #include <algorithm>
// #include <map>
// #include <complex>
// #include <cassert>
// #include <fstream>
//
// // https://bitbucket.org/staffanulfberg/ldaudio/src/master/
//
// using complex = std::complex<double>;
//
// // samples per carrier cycle
// const int samplesPerCarrierCycle = 16;
//
//
// struct AC3Filter {
//     AC3Filter(std::istream &file, double sampleRate) : file(file) {
//         build_filter(sampleRate);
//         file.read((char *) buffer, buffer_size); // Fill the buffer
//         assert(file.good());
//         file_start_pos = file.tellg();
//     }
//
//     long file_start_pos; // todo; allow setting & jumping to start position in file
//
//     std::istream &file;
//
//     static const int N = 1024 * samplesPerCarrierCycle;
//     static const int filterSize = 64;
//     static const int filterCoefficientSize = filterSize * 2 - 1;
//     double filterCoefficients[filterCoefficientSize]{};
//     static const int buffer_size = filterCoefficientSize;
//     uint8_t buffer[buffer_size]{};
//     int buffer_pos = 0;
//
//     void build_filter(double sampleRate) {
//         complex filterResponse[N];
//
//         for (auto i = 0; i < N; ++i) {
//             double f = ((i < N / 2) ? i : N - i) * sampleRate / N;
//             if (std::abs(f - 2.88e6) < 150e3)
//                 filterResponse[i] = {1, 0};
//             else
//                 filterResponse[i] = {0, 0};
//         }
//
//         rfft(filterResponse, N);
//
//         for (auto &bucket: filterResponse) // ensure only real coefficients
//             assert(bucket.imag() < 1e-10);
//
//         for (int i = 1; i < N; ++i) // ensure filter is symmetric
//             assert(std::abs(filterResponse[i].real() - filterResponse[N - i].real()) < 1e-10);
//
//         for (int i = 0; i < filterSize - 1; ++i) // takeRight
//             filterCoefficients[i] = filterResponse[N - filterSize + i + 1].real();
//         for (int i = 0; i < filterSize; ++i) // take (left)
//             filterCoefficients[i + filterSize - 1] = filterResponse[i].real();
//
//         // Apply Hann function. Magic number 0.53836
//         for (int i = 0; i < filterCoefficientSize; ++i)
//             filterCoefficients[i] = hann(0.53836, i - filterSize, N) * filterCoefficients[i];
//     }
//
//
//     // to-do; (probably) loses buffer_size samples from end
//     // ~50% time spent here
//     uint8_t next() {
//         auto value = file.get();
//         if (value == EOF)
//             throw std::range_error("EOF");
//
//         // if (file.tellg() >= file_start_pos + 40 * 1048576) // end pos
//         //     throw std::range_error("DEBUG");
//
//         buffer[buffer_pos] = (char) value & 0xff;
//         buffer_pos = (buffer_pos + 1) % buffer_size;
//
//         double o = 0.0;
//         for (int j = 0; j < filterCoefficientSize; ++j) {
//             auto sample = buffer[(buffer_pos + 127 - 1 - j) % buffer_size];
//             o += filterCoefficients[j] * sample;
//         }
//
//         return uint8_t(o * 64 + 128) & 0xff;
//     }
//
//
//     static double inline hann(double a0, int i, int n) {
//         return a0 - (1 - a0) * cos(2. * M_PI * double(i) / double(n));
//     }
//
//     static void inline rfft(complex *x, int size) { // real fft?
//         return transform(x, size); // , complex{0, -2}, 2
//     }
//
//     // derived from https://gist.github.com/lukicdarkoo/3f0d056e9244784f8b4a
//     static void transform(complex *x, int size) { // NOLINT(misc-no-recursion)
//         // Check if it is split enough
//         if (size <= 1)
//             return;
//
//         // Split even and odd
//         complex odd[size / 2];
//         complex even[size / 2];
//         for (int i = 0; i < size / 2; i++) {
//             even[i] = x[i * 2];
//             odd[i] = x[i * 2 + 1];
//         }
//
//         // Split on tasks
//         transform(even, size / 2);
//         transform(odd, size / 2);
//
//         const complex direction{0, -2};
//         const double scalar = 2;
//
//         // Calculate DFT
//         for (int j = 0; j < size / 2; j++) {
//             complex base = even[j] / scalar;
//             complex offset = exp(direction * (M_PI * j / size)) * odd[j] / scalar;
//             x[j] = base + offset;
//             x[size / 2 + j] = base - offset;
//         }
//     }
// };
