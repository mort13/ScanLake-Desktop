# PyInstaller runtime hook: prefer the bundled DirectML.dll / onnxruntime DLLs over older copies
# that may sit in System32 or on PATH.
import os
import sys

base = getattr(sys, "_MEIPASS", None)
if base and hasattr(os, "add_dll_directory"):
    for sub in ("onnxruntime/capi", ""):
        path = os.path.join(base, sub)
        if os.path.isdir(path):
            os.add_dll_directory(path)
