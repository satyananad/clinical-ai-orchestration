import os
import sys
import subprocess

def main():
    # 1. Clean up any previous server holding port 8000
    os.system("lsof -ti :8000 | xargs kill -9 2>/dev/null")

    # 2. Set PYTHONPATH to include project root and venv site-packages
    base_dir = os.path.dirname(os.path.abspath(__file__))
    venv_site = os.path.join(base_dir, "venv", "lib", "python3.14", "site-packages")
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{base_dir}:{venv_site}:{existing_pythonpath}".strip(":")

    print("=" * 65)
    print("  DENTAL CLINICAL WORKSTATION SERVER")
    print("=" * 65)
    print("• Interactive Web Dashboard: http://127.0.0.1:8000")
    print("• Interactive API Docs:      http://127.0.0.1:8000/docs")
    print("• Press Ctrl+C to stop the server")
    print("=" * 65)

    python_bin = os.path.join(base_dir, "venv", "bin", "python")
    if not os.path.exists(python_bin):
        python_bin = sys.executable

    try:
        subprocess.run(
            [python_bin, "-m", "uvicorn", "backend.app.main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"],
            env=env
        )
    except KeyboardInterrupt:
        print("\nServer gracefully stopped.")

if __name__ == "__main__":
    main()
