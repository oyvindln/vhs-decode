/*******************************************************************************
 * logger.hpp
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

#include <iostream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include <map>
#include <sstream>

enum LogLevel {
    DEBU = 0,
    INFO = 1,
    WARN = 2,
    ERRR = 3,
}; // unsure if different compilers might assign different values / ordering?

int constexpr MAX_LOGLEVEL = 3;

std::map<std::string, int> LOG_LEVELS{
    {"DEBUG", DEBU},
    {"INFO",  INFO},
    {"WARN",  WARN},
    {"ERROR", ERRR},
};


class Logger {
public:
    using Clock = std::chrono::steady_clock;
    using TimePoint = Clock::time_point;

    static int GLOBAL_LOG_LEVEL;
    static std::ostream *LOG_STREAM;
    static const TimePoint GLOBAL_START;

    const std::string &logLabel;

    explicit Logger(const LogLevel &level, const std::string &label) : logLabel(label), logLevel(level) {}

    ~Logger() {
        if (logLevel < GLOBAL_LOG_LEVEL)
            return; // do nothing
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
            Clock::now() - GLOBAL_START);

        // if (logLabel == "CRC1" || logLabel == "CRC2")
        *LOG_STREAM << "[" << logLabel << "]\t" << duration.count() << "ms\t" << _buffer.str() << "\n";
    }

    template<typename T>
    Logger &operator<<(T const &value) {
        _buffer << value;
        return *this;
    }

private:
    std::ostringstream _buffer;
    int logLevel;
};


// set static vars (should be in .cpp)
int Logger::GLOBAL_LOG_LEVEL = INFO;

std::ostream *Logger::LOG_STREAM = &std::cerr; // todo; fix this warning (set as stderr in main.cpp?)

const Logger::TimePoint Logger::GLOBAL_START = Logger::Clock::now();
