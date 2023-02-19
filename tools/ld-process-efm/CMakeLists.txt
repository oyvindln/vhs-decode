add_executable(ld-process-efm
    efmprocess.cpp
    main.cpp
    Datatypes/audio.cpp
    Datatypes/f1frame.cpp
    Datatypes/f2frame.cpp
    Datatypes/f3frame.cpp
    Datatypes/section.cpp
    Datatypes/sector.cpp
    Datatypes/tracktime.cpp
    Decoders/c1circ.cpp
    Decoders/c2circ.cpp
    Decoders/c2deinterleave.cpp
    Decoders/efmtof3frames.cpp
    Decoders/f1toaudio.cpp
    Decoders/f1todata.cpp
    Decoders/f2tof1frames.cpp
    Decoders/f3tof2frames.cpp
    Decoders/syncf3frames.cpp
)

target_include_directories(ld-process-efm PRIVATE .)

target_link_libraries(ld-process-efm PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-process-efm)
