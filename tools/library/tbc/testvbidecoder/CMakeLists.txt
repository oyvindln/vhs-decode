add_executable(testvbidecoder
    testvbidecoder.cpp
)

target_link_libraries(testvbidecoder PRIVATE Qt::Core lddecode-library)

add_test(NAME testvbidecoder COMMAND testvbidecoder)
