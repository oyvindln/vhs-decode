add_executable(ld-ac3-demodulate
    main.cpp
)
if(MSVC)
    target_link_libraries(ld-ac3-demodulate PRIVATE ${Getopt_LIBRARIES})
    target_include_directories(ld-ac3-demodulate PRIVATE ${Getopt_INCLUDE_DIRS})
endif()


install(TARGETS ld-ac3-demodulate)
