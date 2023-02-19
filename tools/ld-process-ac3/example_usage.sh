#
# example_usage.sh
#
# ld-process-ac3 - AC3-RF decoder
# Copyright (C) 2022 Leighton Smallshire & Ian Smallshire
#
# Derived from prior work by Staffan Ulfberg with feedback
# to original author. (Copyright (C) 2021-2022)
# https://bitbucket.org/staffanulfberg/ldaudio/src/master/
#
# This file is part of ld-decode-tools.
#
# ld-process-ac3 is free software: you can redistribute it and/or
# modify it under the terms of the GNU General Public License as
# published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

path=$1
outpath=${2:-"$path.ac3"}

# Helpful while developing; automatically build before running
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_MAKE_PROGRAM=ninja -G Ninja -S . -B ./cmake-build-debug
cmake --build ./cmake-build-debug -j 3

# individual steps (useful to cache while developing)
#ffmpeg -hide_banner -y -i "$path" -f s16le -c:a pcm_s16le TP0
#sox -r 40000000 -b 16 -c 1 -e signed -t raw TP0 -b 8 -r 46080000 -e unsigned -c 1 -t raw TP1 sinc -n 500 2600000-3160000
#time cmake-build-debug/demodulate/ld-ac3-demodulate TP1 TP2 demodulate_log
#time cmake-build-debug/decode/ld-ac3-decode TP2 "$outpath" decode_log

time (
  ffmpeg -hide_banner -loglevel error -y -i "$path" -f s16le -c:a pcm_s16le - |
    sox -r 40000000 -b 16 -c 1 -e signed -t raw - -b 8 -r 46080000 -e unsigned -c 1 -t raw - sinc -n 500 2600000-3160000 |
    cmake-build-debug/demodulate/ld-ac3-demodulate - - demodulate_log |
    cmake-build-debug/decode/ld-ac3-decode - "$outpath" decode_log
)
ffplay -codec:a:0 ac3 "$outpath"
