if(BUILD_PYTHON_WHEEL)
    set(CMAKE_CXX_EXTENSIONS OFF)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)

    # You can add development-specific options here, such as warnings, tests,
    # coverage, etc.

    list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake_modules")

    # Needed for ezpwd as it uses alternative operators
    if(MSVC)
        add_compile_options(/permissive-)
    endif()

    if(MSVC)
        find_package(Getopt REQUIRED)
    endif()
endif()
add_subdirectory(demodulate)
add_subdirectory(decode)
