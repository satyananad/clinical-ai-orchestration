import os
import sys

root_dir = os.path.abspath(os.path.dirname(__file__))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

for py_ver in [f"python{sys.version_info.major}.{sys.version_info.minor}", "python3.14", "python3.13", "python3.12", "python3.11"]:
    venv_site = os.path.join(root_dir, "venv", "lib", py_ver, "site-packages")
    if os.path.isdir(venv_site) and venv_site not in sys.path:
        sys.path.insert(0, venv_site)
        break
