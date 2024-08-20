cmake_minimum_required(VERSION 3.21)

#message(STATUS "toolchain file: ${CMAKE_TOOLCHAIN_FILE}")
#message(STATUS "linkage: ${VCPKG_LIBRARY_LINKAGE}")
#message(STATUS "triplet: ${VCPKG_TARGET_TRIPLET}")

project(ld-ac3-decode-wheel)

add_subdirectory(ld-process-ac3)
