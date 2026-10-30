# PyInstaller spec: ScanLake Desktop, full build (adds PyTorch: CUDA/CPU engine and dev-mode
# training). Several GB; the ORT build is the normal player download.
#   python packaging/build.py full
# Output: dist/full/ScanLake Desktop/ScanLake Desktop.exe
# ruff: noqa
# pyright: basic

from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

ROOT = Path(SPECPATH).parent
APP = ROOT / "apps" / "desktop" / "src" / "scanlake_desktop"

datas = [(str(ROOT / "assets"), "assets")]
datas += collect_data_files("scanlake_core", includes=["domain/data/*.json"])
datas += collect_data_files("scanlake_training", includes=["resources/**/*"])
binaries = collect_dynamic_libs("onnxruntime")  # includes DirectML.dll

hiddenimports = [
    "pynput.keyboard._win32",
    "pynput.mouse._win32",
    "keyring.backends.Windows",
    "scanlake_inference.ort_backend",
    "scanlake_inference.torch_backend",
    "scanlake_training.cli",
    "scanlake_training.evaluate",
]

excludes = [
    "torchvision", "tkinter", "matplotlib", "IPython", "pytest",
    "PySide6.QtWebEngineCore", "PySide6.QtWebEngineWidgets", "PySide6.QtQml", "PySide6.QtQuick",
    "PySide6.QtMultimedia", "PySide6.Qt3DCore", "PySide6.QtCharts", "PySide6.QtPdf",
    "pyarrow.flight", "pyarrow.gandiva", "pyarrow.substrait", "pyarrow.cuda",
]

a = Analysis(
    [str(APP / "__main__.py")],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    runtime_hooks=[str(ROOT / "packaging" / "rthooks" / "rth_dlls.py")],
    noarchive=False,
    # Keep Qt and pynput as separate files so they stay replaceable (LGPL)
    module_collection_mode={"PySide6": "py", "pynput": "py"},
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="ScanLake Desktop",
    console=False,
    disable_windowed_traceback=False,
    upx=False,
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False, name="ScanLake Desktop")
