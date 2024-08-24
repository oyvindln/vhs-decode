cmake_minimum_required(VERSION 3.21)
#Workaround to make sure getopt is linked statically on msvc since including dlls is janky
#won¨'t work on 32-bit and arm windows so would still need some solution for that if needed..
if(WIN32 AND NOT VCPKG_TARGET_TRIPLET)
    set(VCPKG_TARGET_TRIPLET "x64-windows-static")
endif()

project(ld-ac3-decode-wheel)

add_subdirectory(ld-process-ac3)
