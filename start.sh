#!/bin/bash
# Free port 8000 if occupied
lsof -ti :8000 | xargs kill -9 2>/dev/null

if [ -f "./venv/bin/uvicorn" ]; then
    ./venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
else
    export PYTHONPATH=".:venv/lib/python3.14/site-packages"
    python3 -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000 --reload
fi
