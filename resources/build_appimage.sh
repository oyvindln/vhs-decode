#!/bin/sh


mkdir build_appimage
cd build_appimage
cmake .. -DUSE_QT_VERSION=5 -DCMAKE_INSTALL_PREFIX=/usr -DBUILD_PYTHON=OFF -DBUILD_LDF_READER=OFF
make -j3
# Full path needed due to python command
make install DESTDIR="$PWD/AppDir"

wget -nc https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz -O - | tar Jxvf - --wildcards --strip-components=1 ffmpeg-*-amd64-static/ffmpeg
cp  ffmpeg ./AppDir/usr/bin/

# Needs linuxdeploy-qt appimage alongside linuxdeploy one
wget -nc https://github.com/linuxdeploy/linuxdeploy/releases/download/1-alpha-20220822-1/linuxdeploy-x86_64.AppImage
wget -nc https://github.com/linuxdeploy/linuxdeploy-plugin-qt/releases/download/continuous/linuxdeploy-plugin-qt-x86_64.AppImage
chmod +x linuxdeploy-x86_64.AppImage
chmod +x linuxdeploy-plugin-qt-x86_64.AppImage
./linuxdeploy-x86_64.AppImage --plugin qt -d ../resources/dd86.desktop -i ../tools/ld-analyse/Graphics/64-analyse.png --appdir AppDir/ --output appimage --custom-apprun ../resources/AppRun

###
#does not re-copy desktop file?
