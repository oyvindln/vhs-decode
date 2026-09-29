"""Application identity for Qt windows.

Desktop environments (Cinnamon/Mint, GNOME, KDE, ...) match a running window
to its launcher/pinned icon through the window class (X11 ``WM_CLASS``) or the
desktop file name (Wayland ``app_id``).  Qt derives both from ``argv[0]``, and
in the self-contained builds that is ``decode.py`` inside a temporary AppImage
mount.  The window then matches no ``.desktop`` file, so the taskbar shows a
placeholder icon and the desktop generates a stray ``Decode.Py`` entry.

Calling :func:`apply_app_identity` right after the ``QApplication`` is created
pins the identity to ``vhs-decode``, which matches ``vhs-decode.desktop``
(``Icon=vhs-decode`` and ``StartupWMClass=vhs-decode``).
"""

APP_IDENTITY = "vhs-decode"


def apply_app_identity(app=None):
    """Set the application name / desktop file name on the running QApplication.

    Must be called after the QApplication exists and before any window is
    shown, because Qt reads the name when the native window is created.
    Never raises: a failure here must not stop the GUI from starting.
    """
    try:
        try:
            from PyQt6.QtCore import QCoreApplication
            from PyQt6.QtGui import QGuiApplication
        except ImportError:
            from PyQt5.QtCore import QCoreApplication
            from PyQt5.QtGui import QGuiApplication

        QCoreApplication.setApplicationName(APP_IDENTITY)
        # Wayland app_id and the freedesktop desktop-file association.
        QGuiApplication.setDesktopFileName(APP_IDENTITY)
    except Exception as exc:  # pragma: no cover - cosmetic only
        print(f"WARN: could not set application identity: {exc}")
