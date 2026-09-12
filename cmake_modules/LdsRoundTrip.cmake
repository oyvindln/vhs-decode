# Check that a .lds file survives an ld-compress compress/uncompress cycle
# unchanged.  Run with cmake -P; expects PYTHON, LD_COMPRESS, SOURCE_LDS and
# WORK_DIR.
#
# This is the end-to-end check for the .lds path: it covers the lddecode.lds
# packing and unpacking, the flac encoder and the PyAV decode, all of which
# ld-compress drives.

foreach(var PYTHON LD_COMPRESS SOURCE_LDS WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} must be set")
    endif()
endforeach()

file(REMOVE_RECURSE "${WORK_DIR}")
file(MAKE_DIRECTORY "${WORK_DIR}")

# ld-compress always writes to the current directory, so work on copies there:
# sample.lds is what gets compressed, original.lds is kept for comparison
foreach(copy_name original.lds sample.lds)
    execute_process(
        COMMAND "${CMAKE_COMMAND}" -E copy "${SOURCE_LDS}" "${WORK_DIR}/${copy_name}"
        RESULT_VARIABLE copy_result
    )
    if(NOT copy_result EQUAL 0)
        message(FATAL_ERROR "Could not copy ${SOURCE_LDS} to ${WORK_DIR}/${copy_name}")
    endif()
endforeach()

# Invoked through the interpreter rather than directly, so that the in-tree
# script's shebang does not have to resolve to the Python being tested
execute_process(
    COMMAND "${PYTHON}" "${LD_COMPRESS}" sample.lds
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE compress_result
)
if(NOT compress_result EQUAL 0)
    message(FATAL_ERROR "ld-compress failed with status ${compress_result}")
endif()

if(NOT EXISTS "${WORK_DIR}/sample.ldf")
    message(FATAL_ERROR "ld-compress did not produce sample.ldf")
endif()

# The .lds has to go, or ld-compress -u would refuse to overwrite it
file(REMOVE "${WORK_DIR}/sample.lds")

execute_process(
    COMMAND "${PYTHON}" "${LD_COMPRESS}" -u sample.ldf
    WORKING_DIRECTORY "${WORK_DIR}"
    RESULT_VARIABLE uncompress_result
)
if(NOT uncompress_result EQUAL 0)
    message(FATAL_ERROR "ld-compress -u failed with status ${uncompress_result}")
endif()

file(MD5 "${WORK_DIR}/original.lds" original_md5)
file(MD5 "${WORK_DIR}/sample.lds" round_trip_md5)

if(NOT original_md5 STREQUAL round_trip_md5)
    file(SIZE "${WORK_DIR}/original.lds" original_size)
    file(SIZE "${WORK_DIR}/sample.lds" round_trip_size)
    message(FATAL_ERROR
        "Round trip changed the data:\n"
        "  original:   ${original_size} bytes, md5 ${original_md5}\n"
        "  round trip: ${round_trip_size} bytes, md5 ${round_trip_md5}")
endif()

message(STATUS "lds round trip is lossless (${original_md5})")
