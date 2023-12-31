# For M_PI constant
add_compile_definitions(_USE_MATH_DEFINES)

set(ld-analyse_SOURCES
    blacksnranalysisdialog.cpp blacksnranalysisdialog.ui
    busydialog.cpp busydialog.ui
    closedcaptionsdialog.cpp closedcaptionsdialog.ui
    main.cpp
    mainwindow.cpp mainwindow.ui
    oscilloscopedialog.cpp oscilloscopedialog.ui
    vectorscopedialog.cpp vectorscopedialog.ui
    aboutdialog.cpp aboutdialog.ui
    videoparametersdialog.cpp videoparametersdialog.ui
    chromadecoderconfigdialog.cpp chromadecoderconfigdialog.ui
    tbcsource.cpp
    vbidialog.cpp vbidialog.ui
    configuration.cpp
    dropoutanalysisdialog.cpp dropoutanalysisdialog.ui
    visibledropoutanalysisdialog.cpp visibledropoutanalysisdialog.ui
    whitesnranalysisdialog.cpp whitesnranalysisdialog.ui
)

qt_add_resources(ld-analyse_SOURCES ld-analyse-resources.qrc)

if(WIN32)
set(APP_ICON_RESOURCE_WINDOWS "${CMAKE_CURRENT_SOURCE_DIR}/appicon.rc")

add_executable(ld-analyse MACOSX_BUNDLE
    ${ld-analyse_SOURCES}
	${APP_ICON_RESOURCE_WINDOWS})
elseif(UNIX)
	add_executable(ld-analyse MACOSX_BUNDLE
    ${ld-analyse_SOURCES})
endif()


target_include_directories(ld-analyse PRIVATE
    ${QWT_INCLUDE_DIR}  ${FFTW_INCLUDE_DIR}
)

target_link_libraries(ld-analyse PRIVATE
    Qt::Core Qt::Gui Qt::Widgets
    ${QWT_LIBRARY}
    lddecode-library lddecode-chroma
)

if(WIN32)
    target_compile_definitions(ld-analyse PRIVATE
        NOMINMAX
        QUSB_LIBRARY
    )
    set_property(TARGET ld-analyse PROPERTY WIN32_EXECUTABLE true)
endif()

if(UNIX)
	install(TARGETS ld-analyse BUNDLE DESTINATION bin)
	#add .tbc mime type
	install(CODE "execute_process(COMMAND xdg-mime install --novendor --mode system ${CMAKE_CURRENT_SOURCE_DIR}/install/application-video-tbc.xml)")
	#add ld-analyse icon
	install(CODE "execute_process(COMMAND xdg-icon-resource install --novendor --mode system --context apps --size 16 ${CMAKE_CURRENT_SOURCE_DIR}/Graphics/16-analyse.png ld-analyse)")
	install(CODE "execute_process(COMMAND xdg-icon-resource install --novendor --mode system --context apps --size 32 ${CMAKE_CURRENT_SOURCE_DIR}/Graphics/32-analyse.png ld-analyse)")
	install(CODE "execute_process(COMMAND xdg-icon-resource install --novendor --mode system --context apps --size 64 ${CMAKE_CURRENT_SOURCE_DIR}/Graphics/64-analyse.png ld-analyse)")
	install(CODE "execute_process(COMMAND xdg-icon-resource install --novendor --mode system --context apps --size 128 ${CMAKE_CURRENT_SOURCE_DIR}/Graphics/128-analyse.png ld-analyse)")
	#add ld-analyse to application menu
	install(CODE "execute_process(COMMAND xdg-desktop-menu install --novendor --mode system ${CMAKE_CURRENT_SOURCE_DIR}/install/ld-analyse.desktop)")
else()
	install(
			TARGETS ld-analyse
			BUNDLE DESTINATION bin
	)
endif()
