"""Launch the AI Study Assistant web app.

Usage:
    python run_app.py                       # http://localhost:8501
    python run_app.py --server.port 8600    # any extra flags go straight to Streamlit
"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main(extra_args=None) -> int:
    extra_args = sys.argv[1:] if extra_args is None else extra_args
    command = [sys.executable, "-m", "streamlit", "run", str(ROOT / "src" / "app" / "main.py"), *extra_args]
    print("Launching the AI Study Assistant...")
    try:
        return subprocess.run(command, cwd=ROOT).returncode
    except FileNotFoundError:
        print("Could not start Python/Streamlit. Install dependencies with: pip install -r requirements.txt")
        return 1
    except KeyboardInterrupt:
        print("\nStopped.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
