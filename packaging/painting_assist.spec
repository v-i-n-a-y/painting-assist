# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for Painting Assist (onedir, windowed).

Build with:  uv run pyinstaller packaging/painting_assist.spec

Produces:
  * macOS   -> dist/Painting Assist.app  (windowed .app bundle)
  * Windows -> dist/Painting Assist/PaintingAssist.exe
  * Linux   -> dist/Painting Assist/PaintingAssist

The app version is hardcoded below; keep it in sync with
painting_assist/__init__.py (__version__) and pyproject.toml.
"""

import sys
from pathlib import Path

# ``__file__`` is not defined when PyInstaller execs the spec, but ``SPECPATH``
# (the directory containing this spec) is injected into the namespace.
PROJECT_ROOT = Path(SPECPATH).resolve().parent
RESOURCES = PROJECT_ROOT / "painting_assist" / "resources"

APP_VERSION = "0.12.0"  # keep in sync with pyproject.toml / __init__.py

# Display name for the bundle; binary name differs on win/linux (no spaces).
APP_DISPLAY_NAME = "Painting Assist"
BINARY_NAME = "PaintingAssist"

if sys.platform == "darwin":
    icon_file = str(RESOURCES / "icon.icns")
elif sys.platform == "win32":
    icon_file = str(RESOURCES / "icon.ico")
else:
    icon_file = None  # AppImage/desktop icon handled by CI packaging

block_cipher = None


a = Analysis(
    [str(PROJECT_ROOT / "packaging" / "launch.py")],
    pathex=[str(PROJECT_ROOT)],
    binaries=[],
    # Ship the whole resources/ dir as package data under painting_assist/.
    datas=[(str(RESOURCES), "painting_assist/resources")],
    # mixbox is imported lazily (guarded) by painting_assist.mixing, so name it
    # explicitly to be sure PyInstaller bundles it (single self-contained module).
    hiddenimports=["mixbox"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # Trim heavy stdlib/3rd-party modules that this app never imports. Kept
    # conservative: PySide6, cv2, numpy and Pillow must stay fully functional.
    excludes=[
        "tkinter",
        "matplotlib",
        "pytest",
        "PySide6.QtWebEngineCore",
        "PySide6.QtWebEngineWidgets",
        "PySide6.QtQuick",
        "PySide6.QtQml",
        "PySide6.Qt3DCore",
        "PySide6.QtCharts",
        "PySide6.QtDataVisualization",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

# On macOS the EXE is named for display (it lives inside the .app); on
# win/linux use the space-free binary name.
exe_name = APP_DISPLAY_NAME if sys.platform == "darwin" else BINARY_NAME

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,  # windowed / GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon_file,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name=APP_DISPLAY_NAME,
)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name=f"{APP_DISPLAY_NAME}.app",
        icon=icon_file,
        bundle_identifier="com.vinaywilliams.painting-assist",
        version=APP_VERSION,
        info_plist={
            "CFBundleName": APP_DISPLAY_NAME,
            "CFBundleDisplayName": APP_DISPLAY_NAME,
            "CFBundleShortVersionString": APP_VERSION,
            "CFBundleVersion": APP_VERSION,
            "NSHighResolutionCapable": True,
            "LSMinimumSystemVersion": "11.0",
        },
    )
