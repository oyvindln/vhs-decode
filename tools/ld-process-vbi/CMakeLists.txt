add_executable(ld-process-vbi
    biphasecode.cpp
    closedcaption.cpp
    decoderpool.cpp
    main.cpp
    fmcode.cpp
    vbilinedecoder.cpp
    videoid.cpp
    vitccode.cpp
    whiteflag.cpp
)

target_link_libraries(ld-process-vbi PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-process-vbi)
