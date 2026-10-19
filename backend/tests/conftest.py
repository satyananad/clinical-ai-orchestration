import os
import sys

# Ensure assignment root and backend directory are in sys.path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
backend_dir = os.path.join(root_dir, "backend")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

# Auto-detect and include virtual environment site-packages if running outside venv
for py_ver in [f"python{sys.version_info.major}.{sys.version_info.minor}", "python3.14", "python3.13", "python3.12", "python3.11"]:
    venv_site = os.path.join(root_dir, "venv", "lib", py_ver, "site-packages")
    if os.path.isdir(venv_site) and venv_site not in sys.path:
        sys.path.insert(0, venv_site)
        break

import pytest
from backend.app.services.clock import set_frozen_clock

@pytest.fixture(autouse=True)
def deterministic_clinical_clock():
    """Ensures appointment dates (Oct 19-20, 2026) are deterministic across timezones and calendar days."""
    set_frozen_clock("2026-10-19 12:00:00")
    yield
    set_frozen_clock(None)

