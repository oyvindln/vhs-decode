#-------------------------------------------------
#
# Project created by QtCreator 2018-11-03T13:40:24
#
#-------------------------------------------------

QT       += core gui

greaterThan(QT_MAJOR_VERSION, 4): QT += widgets

TARGET = ld-analyse
TEMPLATE = app

# The following define makes your compiler emit warnings if you use
# any feature of Qt which has been marked as deprecated (the exact warnings
# depend on your compiler). Please consult the documentation of the
# deprecated API in order to know how to port your code away from it.
DEFINES += QT_DEPRECATED_WARNINGS

# You can also make your code fail to compile if you use deprecated APIs.
# In order to do so, uncomment the following line.
# You can also select to disable deprecated APIs only up to a certain version of Qt.
#DEFINES += QT_DISABLE_DEPRECATED_BEFORE=0x060000    # disables all the APIs deprecated before Qt 6.0.0

CONFIG += c++11

SOURCES += \
    blacksnranalysisdialog.cpp \
    busydialog.cpp \
    closedcaptionsdialog.cpp \
    main.cpp \
    mainwindow.cpp \
    oscilloscopedialog.cpp \
    aboutdialog.cpp \
    chromadecoderconfigdialog.cpp \
    tbcsource.cpp \
    vbidialog.cpp \
    configuration.cpp \
    dropoutanalysisdialog.cpp \
    ../ld-chroma-decoder/palcolour.cpp \
    ../ld-chroma-decoder/comb.cpp \
    ../ld-chroma-decoder/rgb.cpp \
    ../ld-chroma-decoder/ycbcr.cpp \
    ../ld-chroma-decoder/transformpal.cpp \
    ../ld-chroma-decoder/transformpal2d.cpp \
    ../ld-chroma-decoder/transformpal3d.cpp \
    ../ld-chroma-decoder/framecanvas.cpp \
    ../ld-chroma-decoder/sourcefield.cpp \
    ../library/tbc/lddecodemetadata.cpp \
    ../library/tbc/sourcevideo.cpp \
    ../library/tbc/vbidecoder.cpp \
    ../library/tbc/filters.cpp \
    ../library/tbc/logging.cpp \
    ../library/tbc/dropouts.cpp \
    whitesnranalysisdialog.cpp

HEADERS += \
    blacksnranalysisdialog.h \
    busydialog.h \
    closedcaptionsdialog.h \
    mainwindow.h \
    oscilloscopedialog.h \
    aboutdialog.h \
    chromadecoderconfigdialog.h \
    tbcsource.h \
    vbidialog.h \
    configuration.h \
    dropoutanalysisdialog.h \
    ../ld-chroma-decoder/palcolour.h \
    ../ld-chroma-decoder/comb.h \
    ../ld-chroma-decoder/rgb.h \
    ../ld-chroma-decoder/outputframe.h \
    ../ld-chroma-decoder/ycbcr.h \
    ../ld-chroma-decoder/yiq.h \
    ../ld-chroma-decoder/transformpal.h \
    ../ld-chroma-decoder/transformpal2d.h \
    ../ld-chroma-decoder/transformpal3d.h \
    ../ld-chroma-decoder/framecanvas.h \
    ../ld-chroma-decoder/sourcefield.h \
    ../library/filter/firfilter.h \
    ../library/tbc/lddecodemetadata.h \
    ../library/tbc/sourcevideo.h \
    ../library/tbc/vbidecoder.h \
    ../library/tbc/filters.h \
    ../library/tbc/logging.h \
    ../library/tbc/dropouts.h \
    whitesnranalysisdialog.h

FORMS += \
    blacksnranalysisdialog.ui \
    busydialog.ui \
    closedcaptionsdialog.ui \
    mainwindow.ui \
    oscilloscopedialog.ui \
    aboutdialog.ui \
    chromadecoderconfigdialog.ui \
    vbidialog.ui \
    dropoutanalysisdialog.ui \
    whitesnranalysisdialog.ui

# Add external includes to the include path
INCLUDEPATH += ../library/filter
INCLUDEPATH += ../library/tbc
INCLUDEPATH += ../ld-chroma-decoder

# Include git information definitions
isEmpty(BRANCH) {
    BRANCH = "unknown"
}
isEmpty(COMMIT) {
    COMMIT = "unknown"
}
DEFINES += APP_BRANCH=\"\\\"$${BRANCH}\\\"\" \
    APP_COMMIT=\"\\\"$${COMMIT}\\\"\"

# Rules for installation
isEmpty(PREFIX) {
    PREFIX = /usr/local
}
unix:!android: target.path = $$PREFIX/bin/
!isEmpty(target.path): INSTALLS += target

RESOURCES += \
    ld-analyse-resources.qrc

# Additional include paths to support MacOS compilation
macx {
ICON = Graphics/ld-analyse.icns
INCLUDEPATH += "/usr/local/include"
}

# Normal open-source OS goodness
LIBS += -L"/usr/local/lib"
LIBS += -lfftw3

# Include the QWT library (used for charting)
unix:!macx {
#INCLUDEPATH += $(QWT)/include
INCLUDEPATH += /usr/include/qwt
LIBS += -lqwt-qt5 #Distrubutions other than Ubuntu may be -lqwt
}
macx {
INCLUDEPATH += "/usr/local/lib/qwt.framework/Versions/6/Headers"
LIBS += -F"/usr/local/lib" -framework qwt
}
