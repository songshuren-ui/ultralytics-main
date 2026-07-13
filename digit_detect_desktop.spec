# -*- mode: python ; coding: utf-8 -*-

from pathlib import Path
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

root = Path.cwd()
hiddenimports = [
    "ultralytics",
    "ultralytics.nn.tasks",
    "ultralytics.utils",
    "ultralytics.cfg",
    "PySide6.QtCore",
    "PySide6.QtGui",
    "PySide6.QtWidgets",
]
binaries = collect_dynamic_libs("PySide6")
data = collect_data_files("PySide6") + [
    (str(root / 'dist' / 'best.pt'), '.'),
]


a = Analysis(
    ['digit_detect_desktop.py'],
    pathex=[str(root)],
    binaries=binaries,
    data=data,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['streamlit', 'matplotlib', 'pandas', 'tensorflow', 'tensorboard', 'IPython', 'notebook'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.data,
    [],
    name='digit_detect_desktop',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
