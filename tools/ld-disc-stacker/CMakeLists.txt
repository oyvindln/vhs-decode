add_executable(ld-disc-stacker
    main.cpp
    stacker.cpp
    stackingpool.cpp
)

target_link_libraries(ld-disc-stacker PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-disc-stacker)
