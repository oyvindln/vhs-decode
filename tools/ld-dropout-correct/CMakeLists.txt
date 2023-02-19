add_executable(ld-dropout-correct
    correctorpool.cpp
    main.cpp
    dropoutcorrect.cpp
)

target_link_libraries(ld-dropout-correct PRIVATE Qt::Core lddecode-library)

install(TARGETS ld-dropout-correct)
