"""
launcher.py — start dashboard.py and app.py in separate console windows.

Usage:
    python launcher.py
"""
import os
import sys
import time
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))

APPS = [
    ("dashboard.py", 8501, "Nifty Dashboard"),
    ("app.py",       8502, "Quant Screener"),
]

SEP = "=" * 48


def start_app(script: str, port: int) -> subprocess.Popen:
    cmd = [sys.executable, "-m", "streamlit", "run", script, f"--server.port={port}"]
    kwargs: dict = {"cwd": HERE}
    if sys.platform == "win32":
        # Each app gets its own visible console window
        kwargs["creationflags"] = subprocess.CREATE_NEW_CONSOLE
    return subprocess.Popen(cmd, **kwargs)


def main() -> None:
    print(f"\n  {SEP}")
    print("   Nifty Quant Screener — Launcher")
    print(f"  {SEP}\n")

    procs: list[subprocess.Popen] = []
    for script, port, label in APPS:
        print(f"  Starting {label:20s} → http://localhost:{port}")
        procs.append(start_app(script, port))
        time.sleep(2)   # stagger startup so ports don't race

    print(f"\n  {SEP}")
    print("  Both apps are running in separate windows.")
    print(f"  {SEP}")
    print("  dashboard.py  →  http://localhost:8501")
    print("  app.py        →  http://localhost:8502")
    print(f"  {SEP}")
    print("  Close the individual app windows to stop each server.")
    print("  Press Ctrl+C here to stop both at once.\n")

    try:
        # Block until both child processes exit (user closes their windows)
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n  Shutting down both servers...")
        for p in procs:
            p.terminate()
        for p in procs:
            p.wait()
        print("  Done.\n")


if __name__ == "__main__":
    main()
