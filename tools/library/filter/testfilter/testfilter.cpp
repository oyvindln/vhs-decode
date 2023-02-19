/************************************************************************

    testfilter.cpp

    ld-decode-tools filter library
    Copyright (C) 2014-5 Chad Page
    Copyright (C) 2019 Adam Sampson

    This file is part of ld-decode-tools.

    ld-chroma-decoder is free software: you can redistribute it and/or
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

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

using std::array;
using std::cerr;
using std::fill;
using std::string;
using std::to_string;
using std::vector;

#include "deemp.h"
#include "firfilter.h"

// This is the original filter code from ld-decoder.h.
class SimpleFilter
{
protected:
    int order;
    vector<double> b, a;
    vector<double> y, x;

public:
    template <typename BSrc, typename ASrc>
    SimpleFilter(const BSrc &_b, const ASrc &_a)
        : b(_b.begin(), _b.end()), a(_a.begin(), _a.end())
    {
        order = b.size();

        x.resize(b.size() + 1);
        y.resize(a.size() + 1);

        clear();
    }

    void clear(double val = 0)
    {
        for (unsigned i = 0; i < a.size(); i++) {
            y[i] = val;
        }
        for (unsigned i = 0; i < b.size(); i++) {
            x[i] = val;
        }
    }

    inline double feed(double val)
    {
        double a0 = a[0];
        double y0;

        double *x_data = x.data();
        double *y_data = y.data();

        memmove(&x_data[1], x_data, sizeof(double) * (b.size() - 1));
        memmove(&y_data[1], y_data, sizeof(double) * (a.size() - 1));

        x[0] = val;
        y0 = 0;

        for (unsigned o = 0; o < b.size(); o++) {
            y0 += ((b[o] / a0) * x[o]);
        }
        for (unsigned o = 1; o < a.size(); o++) {
            y0 -= ((a[o] / a0) * y[o]);
        }

        y[0] = y0;
        return y[0];
    }
};

// Check that two filters produce the same output given the same input.
template <typename FN, typename FO>
void testIIRFilter(const char *name, FN& fn, FO& fo)
{
    cerr << "Testing IIRFilter: " << name << "\n";

    for (int i = 0; i < 100; ++i) {
        double input = i - 40;
        double out_n = fn.feed(input);
        double out_o = fo.feed(input);
        if (fabs(out_n - out_o) > 0.000001) {
            cerr << "Mismatch on " << name << " at " << i << ": " << input << " -> " << out_n << ", " << out_o << "\n";
            exit(1);
        }
    }
}

// Test IIRFilter for the sets of coefficients used in the code
void testIIRFilters()
{
    auto f0(f_colorlp);
    SimpleFilter g0(c_colorlp_b, c_colorlp_a);
    testIIRFilter("colorlp", f0, g0);

    auto f2(f_nrc);
    SimpleFilter g2(c_nrc_b, c_nrc_a);
    testIIRFilter("nrc", f2, g2);

    auto f3(f_nr);
    SimpleFilter g3(c_nr_b, c_nr_a);
    testIIRFilter("nr", f3, g3);

    auto f4(f_a500_48k);
    SimpleFilter g4(c_a500_48k_b, c_a500_48k_a);
    testIIRFilter("a500_48k", f4, g4);

    auto f5(f_a40h_48k);
    SimpleFilter g5(c_a40h_48k_b, c_a40h_48k_a);
    testIIRFilter("a40h_48k", f5, g5);
}

// Check that FIRFilter's output matches SimpleFilter in FIR mode.
template <typename Input, typename Output, typename Coeffs>
void testFIRFilter(const string &name, const Input& input, const Output& output, const Coeffs &coeffs, double epsilon = 0.000001)
{
    cerr << "Testing FIRFilter: " << name << "\n";

    const array<typename Coeffs::value_type, 1> one {1};
    SimpleFilter refFilter(coeffs, one);

    // SimpleFilter has a delay, so pre-feed an appropriate number of samples
    const unsigned delay = coeffs.size() / 2;
    for (unsigned i = 0; i < delay; i++) {
        const double input_o = (i >= input.size()) ? 0 : input[i];
        refFilter.feed(input_o);
    }

    for (unsigned i = 0; i < input.size(); i++) {
        // Feed 0s once we reach the end of the input data
        const unsigned j = delay + i;
        const double input_o = (j >= input.size()) ? 0 : input[j];

        const double out_o = refFilter.feed(input_o);
        const double out_n = output[i];
        if (fabs(out_n - out_o) >= epsilon) {
            cerr << "Mismatch on " << name << " at " << i << ": " << input_o << " -> " << out_n << ", " << out_o << "\n";
            exit(1);
        }
    }
}

// Test FIRFilter with a set of coefficients for various types
template <typename Coeffs>
void testFIRCoeffs(const string &name, const Coeffs &coeffs)
{
    const auto f = makeFIRFilter(coeffs);
    vector<double> input, output;

    // Vectors with lengths from 0 to slightly more than the coefficients size.
    // This tests that making up samples outside the bounds of the input works
    // correctly for all combinations of sizes.

    for (int i = 0; i < static_cast<int>(coeffs.size()) + 3; i++) {
        f.apply(input, output);
        testFIRFilter(name + " length " + to_string(i) + " separate", input, output, coeffs);

        output = input;
        f.apply(output);
        testFIRFilter(name + " length " + to_string(i) + " in-place", input, output, coeffs);

        input.push_back(i + 42);
        output.push_back(0);
    }

    // Typical length double vectors

    input.clear();
    for (int i = 0; i < 100; i++) {
        input.push_back(i - 40);
    }
    output.resize(input.size());

    fill(output.begin(), output.end(), 0);
    f.apply(input, output);
    testFIRFilter(name + " double separate", input, output, coeffs);

    output = input;
    f.apply(output);
    testFIRFilter(name + " double in-place", input, output, coeffs);

    // int16_t vectors

    vector<int16_t> input16, output16;
    for (int i = 0; i < 100; i++) {
        input16.push_back(i - 40);
    }
    output16.resize(input16.size());

    fill(output16.begin(), output16.end(), 0);
    f.apply(input16, output16);
    testFIRFilter(name + " int16_t separate", input16, output16, coeffs, 1);

    output16 = input16;
    f.apply(output16);
    testFIRFilter(name + " int16_t in-place", input16, output16, coeffs, 1);

    // Different types for input and output

    fill(output16.begin(), output16.end(), 0);
    f.apply(input, output16);
    testFIRFilter(name + " double->int16_t", input, output16, coeffs, 1);

    fill(output.begin(), output.end(), 0);
    f.apply(input16, output);
    testFIRFilter(name + " int16_t->double", input16, output, coeffs);
}

// Test FIRFilter
void testFIRFilters()
{
    const array<double, 1> one {1};
    testFIRCoeffs("one", one);

    assert(c_nrc_a.size() == 1);
    testFIRCoeffs("nrc", c_nrc_b);

    assert(c_nr_a.size() == 1);
    testFIRCoeffs("nr", c_nr_b);

    assert(c_a500_44k_a.size() == 1);
    testFIRCoeffs("a500_44k", c_a500_44k_b);
}

int main()
{
    testIIRFilters();
    testFIRFilters();

    return 0;
}
