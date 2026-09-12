# Functional tests for ld-decode Python tools
#
# These tests verify that ld-decode can correctly ingest PAL and NTSC files,
# producing expected output (TBC, metadata, audio, etc.)
#
# Most tests expect the ld-decode-testdata repo within the source directory as "testdata".

set(TESTDATA_DIR ${CMAKE_SOURCE_DIR}/testdata)

# Python unit tests, including the .lds converter's bit-exactness tests against
# the C++ ld-lds-converter it replaces.  Invoked as "python -m pytest" so the
# working tree is used rather than any installed copy of lddecode.
add_test(
    NAME python-unit-tests
    COMMAND ${Python3_EXECUTABLE} -m pytest -q ${CMAKE_SOURCE_DIR}/tests
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
)

# Test that ld-decode can decode NTSC files and produce TBC output
add_test(
    NAME decode-ntsc-basic
    COMMAND ${CMAKE_SOURCE_DIR}/ld-decode
        ${TESTDATA_DIR}/ntsc/ve-snw-cut.ldf
        ${CMAKE_BINARY_DIR}/testout/ntsc-basic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test that ld-decode can decode PAL files and produce TBC output
add_test(
    NAME decode-pal-basic
    COMMAND ${CMAKE_SOURCE_DIR}/ld-decode
        --PAL
        ${TESTDATA_DIR}/pal/ggv-mb-1khz.ldf
        ${CMAKE_BINARY_DIR}/testout/pal-basic
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Test that ld-cut can extract a segment from NTSC file
add_test(
    NAME cut-ntsc-segment
    COMMAND ${CMAKE_SOURCE_DIR}/ld-cut
        -S 30255 -l 4
        ${TESTDATA_DIR}/ntsc/ve-snw-cut.ldf
        ${CMAKE_BINARY_DIR}/testout/ntsc-cut.ldf
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(cut-ntsc-segment PROPERTIES TIMEOUT 120)

# Test that ld-cut can extract a segment from PAL file
add_test(
    NAME cut-pal-segment
    COMMAND ${CMAKE_SOURCE_DIR}/ld-cut
        --pal -S 760 -l 4
        ${TESTDATA_DIR}/pal/ggv-mb-1khz.ldf
        ${CMAKE_BINARY_DIR}/testout/pal-cut.ldf
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(cut-pal-segment PROPERTIES TIMEOUT 120)

# Test decode of NTSC cut segment
add_test(
    NAME decode-ntsc-cut
    COMMAND ${CMAKE_SOURCE_DIR}/ld-decode
        ${CMAKE_BINARY_DIR}/testout/ntsc-cut.ldf
        ${CMAKE_BINARY_DIR}/testout/ntsc-cut-decoded
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(decode-ntsc-cut PROPERTIES DEPENDS cut-ntsc-segment)

# Test decode of PAL cut segment
add_test(
    NAME decode-pal-cut
    COMMAND ${CMAKE_SOURCE_DIR}/ld-decode
        --PAL
        ${CMAKE_BINARY_DIR}/testout/pal-cut.ldf
        ${CMAKE_BINARY_DIR}/testout/pal-cut-decoded
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(decode-pal-cut PROPERTIES DEPENDS cut-pal-segment)

# Test that ld-cut can write packed .lds output
add_test(
    NAME cut-ntsc-lds
    COMMAND ${CMAKE_SOURCE_DIR}/ld-cut
        -S 30255 -l 4
        ${TESTDATA_DIR}/ntsc/ve-snw-cut.ldf
        ${CMAKE_BINARY_DIR}/testout/ntsc-cut.lds
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(cut-ntsc-lds PROPERTIES TIMEOUT 120)

# Test that the .lds ld-cut produced is actually decodable
add_test(
    NAME decode-ntsc-lds
    COMMAND ${CMAKE_SOURCE_DIR}/ld-decode
        ${CMAKE_BINARY_DIR}/testout/ntsc-cut.lds
        ${CMAKE_BINARY_DIR}/testout/ntsc-lds-decoded
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)
set_tests_properties(decode-ntsc-lds PROPERTIES DEPENDS cut-ntsc-lds)

# Test that ld-compress can compress and uncompress a .lds file without
# changing a single byte.  flac is the one external program ld-compress still
# needs - everything else it does is in process - so skip if it is missing.
find_program(FLAC_EXECUTABLE NAMES flac)

if(FLAC_EXECUTABLE)
    add_test(
        NAME compress-lds-round-trip
        COMMAND ${CMAKE_COMMAND}
            -DPYTHON=${Python3_EXECUTABLE}
            -DLD_COMPRESS=${CMAKE_SOURCE_DIR}/ld-compress
            -DSOURCE_LDS=${CMAKE_BINARY_DIR}/testout/ntsc-cut.lds
            -DWORK_DIR=${CMAKE_BINARY_DIR}/testout/lds-round-trip
            -P ${CMAKE_SOURCE_DIR}/cmake_modules/LdsRoundTrip.cmake
    )
    set_tests_properties(compress-lds-round-trip PROPERTIES
        DEPENDS cut-ntsc-lds
        TIMEOUT 300
    )
else()
    message(STATUS "Skipping compress-lds-round-trip test (flac not found)")
endif()
