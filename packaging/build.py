"""Build the player app with PyInstaller in a clean, player-only environment.

    python packaging/build.py ort      # ONNX Runtime build (default player download)
    python packaging/build.py full     # + PyTorch engine and dev-mode training (several GB)

Steps: fetch + verify the pinned assets, create `.venv-<flavor>` with only the desktop app and
its runtime dependencies (no dev tools; PyTorch only for `full`), add PyInstaller, build, then
run the built exe with --self-test. Then, if Inno Setup is installed, build the installer.
Needs uv on PATH.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYINSTALLER = "pyinstaller>=6.11"


def run(cmd: list[str], env: dict[str, str] | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=ROOT, check=True, env=env)


def main(argv: list[str]) -> int:
    flavor = argv[0] if argv else "ort"
    if flavor not in ("ort", "full"):
        print("flavor must be 'ort' or 'full'", file=sys.stderr)
        return 2
    venv = ROOT / f".venv-{flavor}"
    env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
    run([sys.executable, "packaging/fetch_assets.py"])
    extras = ["--extra", "torch"] if flavor == "full" else []
    run(["uv", "sync", "--package", "scanlake-desktop", "--no-dev", "--frozen", *extras], env)
    python = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    run(["uv", "pip", "install", "--python", str(python), PYINSTALLER], env)
    dist = ROOT / "dist" / ("full" if flavor == "full" else "")
    run(
        [
            str(python),
            "-m",
            "PyInstaller",
            "--noconfirm",
            "--clean",
            "--distpath",
            str(dist),
            "--workpath",
            f"build/{flavor}",
            f"packaging/desktop-{flavor}.spec",
        ]
    )

    exe = dist / "ScanLake Desktop" / "ScanLake Desktop.exe"
    report = Path(tempfile.mkdtemp()) / "self-test.json"
    code = subprocess.run([str(exe), "--self-test", str(report)], check=False).returncode
    print(report.read_text(encoding="utf-8") if report.exists() else "no self-test report")
    if code != 0 or not json.loads(report.read_text(encoding="utf-8")).get("ok"):
        print("self-test FAILED", file=sys.stderr)
        return 1
    size = sum(f.stat().st_size for f in exe.parent.rglob("*") if f.is_file())
    print(f"built {exe} ({size / 1e6:.0f} MB)")
    iscc = shutil.which("iscc") or next(
        (
            str(p)
            for p in Path(os.environ.get("ProgramFiles(x86)", "")).glob("Inno Setup */ISCC.exe")
        ),
        None,
    )
    if iscc is None:
        print("Inno Setup not found: skipping the installer (https://jrsoftware.org/isinfo.php)")
        return 0
    version = _app_version()
    run(
        [
            iscc,
            f"/DAppVersion={version}",
            f"/DFlavor={flavor}",
            "packaging/installer/scanlake-desktop.iss",
        ]
    )
    return 0


def _app_version() -> str:
    import tomllib

    data = tomllib.loads((ROOT / "apps" / "desktop" / "pyproject.toml").read_text("utf-8"))
    return str(data["project"]["version"])


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
