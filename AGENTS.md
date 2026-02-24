## Cursor Cloud specific instructions

This is a minimal single-file Python Flask app (`web_vers.py`) that captures the host screen and streams it as MJPEG over HTTP on port 5000.

### Dependencies

Install with: `pip install -r requirements.txt`

### Running the app

The app requires a display environment for `mss` screen capture. In headless/cloud environments, start Xvfb first:

```
Xvfb :99 -screen 0 1920x1080x24 &
export DISPLAY=:99
python3 web_vers.py
```

The server listens on `0.0.0.0:5000`. Visit `/` to see the stream viewer, or `/stream` for the raw MJPEG feed.

### Key caveats

- `opencv-python-headless` is used instead of `opencv-python` to avoid unnecessary GUI library dependencies in headless environments.
- The `mss` library requires a running X server (real or virtual). Without `DISPLAY` set, the app will crash on startup.
- There are no automated tests, linter config, or build steps in this repo. Validation is done by running the server and verifying the stream via `curl` or browser.
