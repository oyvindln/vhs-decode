cmake_minimum_required(VERSION 3.16)

project(ld-decode-tools)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")
include(CTest)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Options that can be specified with -D

option(USE_QWT
    "Use the Qwt library; if OFF, ld-analyse will not be built"
    ON
)

# Needed for ezpwd as it uses alternative operators
if(MSVC)
    add_compile_options(/permissive-)
endif()

set(USE_QT_VERSION "" CACHE STRING
    "Version of Qt to use, 5 or 6 (Which is used by default can vary due to how cmake find functions work, but will often default to Qt6)"
)

# Not working on Windows as of yet. The decoders are able to use ffmpeg binaries instead if needed.
if(NOT WIN32)
    option(BUILD_LDF_READER
        "build ld_ldf_reader"
        ON
    )
endif()

option(BUILD_PYTHON
    "Build and install ld-decode's Python library and tools"
    ON
)

# Check for dependencies

# When using Qt 6.3, you can replace the code block below with qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)
include(GNUInstallDirs)
set(CMAKE_AUTOUIC ON)

set(QT_PACKAGE_NAMES Qt5 Qt6)
if(USE_QT_VERSION)
    set(QT_PACKAGE_NAMES Qt${USE_QT_VERSION})
endif()
find_package(QT NAMES ${QT_PACKAGE_NAMES} REQUIRED COMPONENTS Core)
if(USE_QWT)
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Gui Widgets)
else()
    find_package(Qt${QT_VERSION_MAJOR} REQUIRED COMPONENTS Core)
endif()
message(STATUS "Qt Version: ${QT_VERSION}")

# For Qt < 5.15, emulate 5.15's Qt5CoreConfig.cmake, so we don't have to
# specify Qt5/Qt6 elsewhere.
if(QT_VERSION VERSION_LESS 5.15)
    function(qt_add_resources outfiles)
        qt5_add_resources("${outfiles}" ${ARGN})
        set("${outfiles}" "${${outfiles}}" PARENT_SCOPE)
    endfunction()
    foreach(library Core Gui Widgets)
        add_library(Qt::${library} INTERFACE IMPORTED)
        set_target_properties(Qt::${library} PROPERTIES
            INTERFACE_LINK_LIBRARIES "Qt5::${library}")
    endforeach()
endif()

find_package(PkgConfig REQUIRED)
if(USE_QWT)
    # try pkg-config first
    pkg_check_modules(QWT Qt${QT_VERSION_MAJOR}Qwt6)
    if(QWT_FOUND)
        # .....
        set(QWT_INCLUDE_DIR ${QWT_INCLUDE_DIRS})
        set(QWT_LIBRARY ${QWT_LIBRARIES})
    else()
        find_package(Qwt REQUIRED)
    endif()
endif()

pkg_check_modules(FFTW REQUIRED IMPORTED_TARGET
    fftw3
)

# Get the Git branch and revision

execute_process(
    COMMAND git rev-parse --abbrev-ref HEAD
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_BRANCH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
    COMMAND git log -1 --format=%h
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    OUTPUT_VARIABLE GIT_COMMIT_HASH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

add_compile_definitions(APP_BRANCH=\"${GIT_BRANCH}\")
add_compile_definitions(APP_COMMIT=\"${GIT_COMMIT_HASH}\")

# Subdirectories

if(USE_QWT)
    add_subdirectory(tools/ld-analyse)
endif()
add_subdirectory(tools/ld-chroma-decoder)
add_subdirectory(tools/ld-chroma-decoder/encoder)
add_subdirectory(tools/ld-disc-stacker)
add_subdirectory(tools/ld-discmap)
add_subdirectory(tools/ld-dropout-correct)
add_subdirectory(tools/ld-export-metadata)
add_subdirectory(tools/ld-lds-converter)
add_subdirectory(tools/ld-process-ac3)
add_subdirectory(tools/ld-process-efm)
add_subdirectory(tools/ld-process-vbi)
add_subdirectory(tools/ld-process-vits)
add_subdirectory(tools/library)

if(BUILD_TESTING)
    add_subdirectory(tools/library/filter/testfilter)
    add_subdirectory(tools/library/tbc/testlinenumber)
    add_subdirectory(tools/library/tbc/testmetadata)
    add_subdirectory(tools/library/tbc/testvbidecoder)
    add_subdirectory(tools/library/tbc/testvitcdecoder)
    include(LdDecodeTests)
endif()

# ld-ldf-reader

if(BUILD_LDF_READER)
    pkg_check_modules(LIBAV REQUIRED IMPORTED_TARGET
        libavcodec
        libavformat
        libavutil
    )

    add_executable(ld-ldf-reader
      ld-ldf-reader.c)

    target_link_libraries(ld-ldf-reader PkgConfig::LIBAV)

    install(TARGETS ld-ldf-reader)
endif()

# Python library and tools

if(BUILD_PYTHON)
    find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)

    set(PYTHON_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR}/python-build)

    # python setup.py install seems to behave differently if prefix
    # is supplied, requiring setting pythonpath after install.
    # If a custom prefix is not set, don't supply prefix at allocate_lock
    # /usr/local is the default on linux.
    # Install not really tested on windows as of yet.
    if(${CMAKE_INSTALL_PREFIX} STREQUAL "/usr/local")
        set(PYTHON_PREFIX_LINE "")
    else()
        message(STATUS "Custom install prefix supplied - you may have to manually specify PYTHONPATH")
        set(PYTHON_PREFIX_LINE "--prefix ${CMAKE_INSTALL_PREFIX}")
    endif()

    add_custom_target(python-build ALL
        COMMAND ${PYTHON} ./setup.py --quiet
            build --build-base ${PYTHON_BUILD_DIR}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    )

    # We must run build/egg_info here to specify the directories.
    install(CODE "
        if(NOT DEFINED ENV{DESTDIR})
            set(ENV{DESTDIR} /)
        endif()
        execute_process(
            COMMAND ${PYTHON} ./setup.py --quiet
                build --build-base ${PYTHON_BUILD_DIR}
                egg_info --egg-base ${CMAKE_CURRENT_BINARY_DIR}
                install ${PYTHON_PREFIX_LINE}
                        --root \$ENV{DESTDIR}
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        )
    ")
endif()

install(PROGRAMS gen_chroma_vid.sh TYPE BIN)
