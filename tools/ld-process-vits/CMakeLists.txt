add_executable(ld-process-vits
    main.cpp
    processingpool.cpp
    vitsanalyser.cpp
)

target_link_libraries(ld-process-vits PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-process-vits)
