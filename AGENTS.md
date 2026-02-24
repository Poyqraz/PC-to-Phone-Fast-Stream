## Cursor Cloud specific instructions

This is a Python Flask screen-sharing app that captures the host screen and streams it as MJPEG over HTTP on port 5000, with an adaptive streaming control algorithm.

### Dependencies

- Runtime: `pip install -r requirements.txt`
- Dev tools (linter + tests): `pip install -r requirements-dev.txt`

### Running the app

The app requires a display environment for `mss` screen capture. In headless/cloud environments, start Xvfb first:

```
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
python3 web_vers.py
```

The server listens on `0.0.0.0:5000`. Visit `/` for the dashboard with live stream, metrics, and controls.

### Lint and test commands

- Lint: `ruff check .`
- Tests: `pytest tests/ -v` (requires `DISPLAY` env var set to a running X server)
- Auto-fix lint: `ruff check --fix .`

### Key caveats

- `opencv-python-headless` is used instead of `opencv-python` to avoid unnecessary GUI library dependencies in headless environments.
- The `mss` library requires a running X server (real or virtual). Without `DISPLAY` set, the app will crash on startup.
- Tests that exercise `process_frame()` also need `DISPLAY` set because OpenCV color conversion runs, though no actual screen capture occurs in unit tests.
- The `ruff` and `pytest` binaries install to `~/.local/bin` — ensure it's on `PATH`.
