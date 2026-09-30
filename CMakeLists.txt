cmake_minimum_required(VERSION 3.16)
project(ld-decode)

# Enable testing
include(CTest)

# Find Python
find_package(Python3 3.6 REQUIRED COMPONENTS Interpreter)

# Generate version file from git information
function(generate_version_file)
    set(VERSION_FILE "${CMAKE_SOURCE_DIR}/lddecode/version")
    set(DEFAULT_VERSION "main:unknown")
    
    # Try to get git information
    set(BRANCH "release")
    set(COMMIT "unknown")
    set(DIRTY "")
    
    # Check if we're in a git repository
    if(EXISTS "${CMAKE_SOURCE_DIR}/.git")
        # Try to get exact tag
        execute_process(
            COMMAND git describe --tags --exact-match
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_TAG
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(GIT_TAG)
            # Remove 'v' prefix if present
            string(REGEX REPLACE "^v" "" COMMIT "${GIT_TAG}")
            set(BRANCH "release")
        else()
            # Try to get describe output with commits since tag
            execute_process(
                COMMAND git describe --tags --always
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                OUTPUT_VARIABLE GIT_DESCRIBE
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            
            if(GIT_DESCRIBE)
                string(REGEX REPLACE "^v" "" COMMIT "${GIT_DESCRIBE}")
            endif()
            
            # Get branch name
            execute_process(
                COMMAND git rev-parse --abbrev-ref HEAD
                WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
                OUTPUT_VARIABLE GIT_BRANCH
                ERROR_QUIET
                OUTPUT_STRIP_TRAILING_WHITESPACE
            )
            
            if(GIT_BRANCH)
                set(BRANCH "${GIT_BRANCH}")
                if(BRANCH STREQUAL "HEAD")
                    set(BRANCH "release")
                endif()
            endif()
        endif()
        
        # Check if dirty
        execute_process(
            COMMAND git status --porcelain
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
            OUTPUT_VARIABLE GIT_STATUS
            ERROR_QUIET
            OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        
        if(GIT_STATUS)
            set(DIRTY ":dirty")
        endif()
    endif()
    
    # Generate version string
    set(VERSION_STRING "${BRANCH}:${COMMIT}${DIRTY}")
    
    # Write to version file
    file(WRITE "${VERSION_FILE}" "${VERSION_STRING}\n")
    
    message(STATUS "Generated version file: ${VERSION_STRING}")
endfunction()

# Generate version file during configuration
generate_version_file()

# Include test definitions
include(cmake_modules/LdDecodeTests.cmake)
