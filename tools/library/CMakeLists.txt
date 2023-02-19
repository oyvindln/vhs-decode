add_library(lddecode-library STATIC
    tbc/dropouts.cpp
    tbc/filters.cpp
    tbc/jsonio.cpp
    tbc/lddecodemetadata.cpp
    tbc/logging.cpp
    tbc/navigation.cpp
    tbc/sourceaudio.cpp
    tbc/sourcevideo.cpp
    tbc/vbidecoder.cpp
    tbc/vitcdecoder.cpp
)

target_include_directories(lddecode-library PUBLIC filter tbc)

target_link_libraries(lddecode-library PRIVATE Qt::Core)
