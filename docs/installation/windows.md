# Windows Installation

Windows installation is performed using a portable ZIP archive. The pre-compiled installation package is here:

[ld-decode release](https://github.com/happycube/ld-decode/releases/latest){target="_blank"}

Download the ZIP file which will have a filename similar to ***ld-decode-dev-windows.zip***

This archive contains all required ld-decode tools, in its `bin` directory:

- ld-decode.bat
- ld-cut.bat
- ld-compress.bat
- ld-ldf-reader-py.bat
- ld-lds-converter-py.bat

`bin\flac.exe` is the flac 1.5.0 that ld-compress needs for multithreaded encoding; ld-compress finds it there without any setup, so nothing has to be installed separately.

## Installation

1. Download the ZIP file
2. Extract the contents to a location of your choice (e.g., `C:\ld-decode`)
3. The installation is portable - no installation wizard or registry entries required

## Usage

Run ld-decode from Command Prompt or PowerShell:

```
C:\path\to\ld-decode\bin\ld-decode.bat [arguments]
```

For easier access, add the `bin` directory to your PATH:

**Command Prompt:**
```
set PATH=%PATH%;C:\path\to\ld-decode\bin
ld-decode [arguments]
```

**PowerShell:**
```
$env:Path += ";C:\path\to\ld-decode\bin"
ld-decode [arguments]
```

To permanently add to PATH, edit your system environment variables through System Properties.
