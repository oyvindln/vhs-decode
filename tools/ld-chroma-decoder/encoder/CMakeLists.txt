# For M_PI constant
add_compile_definitions(_USE_MATH_DEFINES)


add_executable(ld-chroma-encoder
    main.cpp
    encoder.cpp
    ntscencoder.cpp
    palencoder.cpp
)

target_link_libraries(ld-chroma-encoder PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-chroma-encoder)
