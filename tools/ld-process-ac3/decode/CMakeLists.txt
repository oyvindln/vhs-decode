add_executable(ld-ac3-decode
    ac3_parsing.cpp
    main.cpp
)
if(MSVC)
    target_link_libraries(ld-ac3-decode PRIVATE ${Getopt_LIBRARIES})
    target_include_directories(ld-ac3-decode PRIVATE ${Getopt_INCLUDE_DIRS})
endif()
target_include_directories(ld-ac3-decode PRIVATE . ../../ld-process-efm)

install(TARGETS ld-ac3-decode)
