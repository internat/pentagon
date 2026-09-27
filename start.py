"""
Intelligent Data Analysis System — One-Click Launcher
=====================================================
Запускает Flask-сервер дашборда и открывает браузер.
Двойной клик по start.bat (Windows) или: python start.py
"""

import os
import sys
import time
import threading
import webbrowser

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

PORT = int(os.environ.get("PORT", 5000))
URL = f"http://localhost:{PORT}"


def open_browser():
    """Открывает браузер после короткой паузы (пока сервер поднимается)."""
    time.sleep(2)
    print("  Opening browser...")
    webbrowser.open(URL)


def main():
    print("=" * 64)
    print("  INTELLIGENT DATA ANALYSIS SYSTEM — AutoML Dashboard")
    print("=" * 64)
    print(f"  Dashboard: {URL}")
    print("  Stop:      Ctrl+C")
    print("=" * 64)

    threading.Thread(target=open_browser, daemon=True).start()

    from server import app
    try:
        app.run(host="127.0.0.1", port=PORT, debug=False, threaded=True)
    except KeyboardInterrupt:
        print("\n  Server stopped.")


if __name__ == "__main__":
    main()
