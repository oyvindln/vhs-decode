add_executable(testmetadata
    testmetadata.cpp
)

target_link_libraries(testmetadata PRIVATE Qt::Core lddecode-library)

add_test(NAME testmetadata COMMAND testmetadata)
