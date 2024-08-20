add_executable(ld-ac3-demodulate
    main.cpp
)
if(MSVC)
    target_link_libraries(ld-ac3-demodulate PRIVATE ${Getopt_LIBRARIES})
    target_include_directories(ld-ac3-demodulate PRIVATE ${Getopt_INCLUDE_DIRS})
endif()

if(BUILD_PYTHON_TOOL)
    install(TARGETS ld-ac3-demodulate
            EXCLUDE_FROM_ALL
            COMPONENT python_binaries
            RUNTIME DESTINATION ${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/scripts
            LIBRARY DESTINATION ${PY_BUILD_CMAKE_PACKAGE_NAME}-${PY_BUILD_CMAKE_PACKAGE_VERSION}.data/platlib)
else()
    install(TARGETS ld-ac3-demodulate)
endif()
