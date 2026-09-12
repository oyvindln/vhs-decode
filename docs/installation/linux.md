# Linux installation

Linux installation is performed using appImage.  The pre-compiled installation package is here:

[ld-decode release](https://github.com/happycube/ld-decode/releases/latest){target="_blank"}

Download the appImage which will have a filename similar to ***ld-decode-dev-x86_64.AppImage***

This AppImage contains all required ld-decode tools:

- ld-decode
- ld-cut
- ld-compress
- ld-ldf-reader-py
- ld-lds-converter-py

It also bundles the flac 1.5.0 that ld-compress needs for multithreaded encoding, along with its own Python interpreter, so nothing has to be installed separately.

## Usage

Run ld-decode (default):

```
./ld-decode-*.AppImage [arguments]
```

To run any of the other tools, create a symlink named after it:

```
ln -s ld-decode-*.AppImage ld-compress
./ld-compress [arguments]
```

Or extract the AppImage and add its `bin` directory to your PATH:

```
./ld-decode-*.AppImage --appimage-extract
export PATH="$PWD/squashfs-root/usr/bin:$PATH"
ld-compress [arguments]
```
