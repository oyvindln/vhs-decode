if(MSVC)
    find_package(Getopt REQUIRED)
endif()

add_subdirectory(demodulate)
add_subdirectory(decode)
