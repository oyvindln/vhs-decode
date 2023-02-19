add_executable(testlinenumber
    testlinenumber.cpp
)

target_link_libraries(testlinenumber PRIVATE Qt::Core lddecode-library)

add_test(NAME testlinenumber COMMAND testlinenumber)
