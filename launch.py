"""One-step launcher for ImageMixer on Windows, macOS and Linux.

People who just want to use the tool should not have to create a virtual environment, install packages or
pick a port by hand. The wrappers (ImageMixer.bat, ImageMixer.command, ImageMixer.sh) all call this file so the
behaviour is identical everywhere and lives in one place.

    python launch.py            start, installing what is missing on first run
    python launch.py --full     also install the optional trained models (cutout and surface extension)
    python launch.py --no-browser
"""
from __future__ import annotations

import argparse
import hashlib
import os
import socket
import subprocess
import sys
import time
import urllib.request
import venv
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MIN_PY = (3, 10)


def venv_python() -> Path:
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def run(cmd: list[str]) -> None:
    subprocess.check_call(cmd, cwd=ROOT)


def ensure_env(full: bool) -> Path:
    py = venv_python()
    if not py.exists():
        print("First run: creating a private Python environment (about a minute)...")
        venv.EnvBuilder(with_pip=True).create(VENV)
    files = [ROOT / "requirements.txt"] + ([ROOT / "requirements-optional.txt"] if full else [])
    digest = hashlib.sha256(b"".join(f.read_bytes() for f in files)).hexdigest()
    stamp = VENV / ("installed-full" if full else "installed")
    if not stamp.exists() or stamp.read_text() != digest:
        print("Installing packages" + (" including the trained models (large download)" if full else "") + "...")
        run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-q", "--upgrade", "pip"])
        for f in files:
            run([str(py), "-m", "pip", "install", "--disable-pip-version-check", "-r", str(f)])
        stamp.write_text(digest)
    return py


def free_port(start: int) -> int:
    # Falling back to the next port keeps a second launch (or another app on 5057) from being a dead end.
    for port in range(start, start + 50):
        with socket.socket() as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise SystemExit("No free port found near %d." % start)


def wait_until_up(url: str, proc: subprocess.Popen, timeout: float = 60) -> bool:
    end = time.time() + timeout
    while time.time() < end and proc.poll() is None:
        try:
            urllib.request.urlopen(url + "/api/devices", timeout=1)
            return True
        except Exception:
            time.sleep(0.4)
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="Launch ImageMixer")
    ap.add_argument("--full", action="store_true", help="install optional trained models too")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--port", type=int, default=5057)
    a = ap.parse_args()

    if sys.version_info < MIN_PY:
        print("ImageMixer needs Python %d.%d or newer (this is %d.%d). Get it from https://www.python.org/downloads/"
              % (MIN_PY + sys.version_info[:2]))
        return 1
    try:
        py = ensure_env(a.full)
    except (subprocess.CalledProcessError, OSError) as e:
        print("Setup failed:", e)
        print("On Debian/Ubuntu, `sudo apt install python3-venv` is often what is missing.")
        return 1

    port = free_port(a.port)
    url = f"http://127.0.0.1:{port}"
    proc = subprocess.Popen([str(py), "-m", "app.server", "--port", str(port)], cwd=ROOT)
    try:
        if wait_until_up(url, proc):
            print(f"\nImageMixer is running at {url}  (close this window or press Ctrl+C to stop)\n")
            if not a.no_browser:
                webbrowser.open(url)
        return proc.wait()
    except KeyboardInterrupt:
        proc.terminate()
        return 0


if __name__ == "__main__":
    sys.exit(main())
