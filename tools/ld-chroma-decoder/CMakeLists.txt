# Library for chroma decoders

add_library(lddecode-chroma STATIC
    comb.cpp
    componentframe.cpp
    framecanvas.cpp
    outputwriter.cpp
    palcolour.cpp
    secamdecoder.cpp
    sourcefield.cpp
    transformpal.cpp
    transformpal2d.cpp
    transformpal3d.cpp
)

target_include_directories(lddecode-chroma PUBLIC .)

target_link_libraries(lddecode-chroma PRIVATE Qt::Core PkgConfig::FFTW lddecode-library)

# ld-chroma-decoder

add_executable(ld-chroma-decoder
    decoder.cpp
    decoderpool.cpp
    main.cpp
    monodecoder.cpp
    ntscdecoder.cpp
    paldecoder.cpp
    secamdecoder.cpp
)

target_link_libraries(ld-chroma-decoder PRIVATE Qt::Core lddecode-library lddecode-chroma)

install(TARGETS ld-chroma-decoder)
