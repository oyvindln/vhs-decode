#!/bin/sh
# entrypoint.sh — AppImage entry point with desktop integration.
#
# AppImages are not automatically registered with the desktop environment.
# Without registration the taskbar shows a generic icon and pinning doesn't
# work. This script extracts the .desktop file and icon from the AppImage on
# first run and installs them to ~/.local/share so GNOME/KDE/Cinnamon/Mint
# taskbars show the correct icon and allow pinning.

APPDIR="$(dirname "$(readlink -f "$0")")"
APPIMAGE_PATH="${APPIMAGE:-}"

# If APPIMAGE env is set (AppImage runtime), use it; otherwise use APPDIR.
if [ -z "${APPIMAGE_PATH}" ]; then
  APPIMAGE_PATH="${APPDIR}"
fi

# Desktop integration: install .desktop file and icon to user dirs.
# Only do this once per AppImage path to avoid repeated work.
INTEGRATION_MARKER="${HOME}/.local/share/applications/.vhs-decode-appimage-$(echo "${APPIMAGE_PATH}" | md5sum | cut -c1-8)"
if [ ! -f "${INTEGRATION_MARKER}" ]; then
  APPS_DIR="${HOME}/.local/share/applications"
  ICONS_DIR="${HOME}/.local/share/icons/hicolor/256x256/apps"
  mkdir -p "${APPS_DIR}" "${ICONS_DIR}" 2>/dev/null || true

  # Install the icon
  if [ -f "${APPDIR}/vhs-decode.png" ]; then
    cp "${APPDIR}/vhs-decode.png" "${ICONS_DIR}/vhs-decode.png" 2>/dev/null || true
  fi

  # Generate a .desktop file with the absolute AppImage path for Exec.
  DESKTOP_FILE="${APPS_DIR}/vhs-decode.desktop"
  cat > "${DESKTOP_FILE}" << DESKEOF
[Desktop Entry]
Name=vhs-decode
Comment=Software defined VHS/LaserDisc/CVBS decoder
Exec=${APPIMAGE_PATH} %f
Terminal=false
Icon=vhs-decode
Type=Application
Categories=AudioVideo;Video;AudioVideoEditing;
Keywords=vhs;laserdisc;ld;cvbs;rf;decode;video;tape;
DESKEOF

  # Update desktop database (best-effort, may not be installed)
  update-desktop-database "${APPS_DIR}" 2>/dev/null || true
  gtk-update-icon-cache -f -t "${HOME}/.local/share/icons/hicolor" 2>/dev/null || true

  # Mark as integrated
  echo "${APPIMAGE_PATH}" > "${INTEGRATION_MARKER}" 2>/dev/null || true
fi

# Run the actual decode binary
exec {{ python-executable }} -u "${APPDIR}/opt/python{{ python-version }}/bin/decode.py" "$@"
