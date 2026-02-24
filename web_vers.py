"""MJPEG screen-sharing server with adaptive streaming control."""

import time

import mss
import numpy as np
from flask import Flask, Response, jsonify, request

from stream_controller import StreamConfig, StreamController

app = Flask(__name__)
controller = StreamController(StreamConfig())


@app.route("/")
def index():
    """Serve the viewer page with live stream, controls, and metrics dashboard."""
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Screen Stream</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: system-ui, sans-serif; background: #111; color: #eee; }
  .container { max-width: 1400px; margin: 0 auto; padding: 16px; }
  h1 { font-size: 1.2rem; margin-bottom: 12px; color: #8cf; }
  .stream-wrap { background: #000; border-radius: 8px; overflow: hidden;
                 margin-bottom: 16px; text-align: center; }
  .stream-wrap img { max-width: 100%; height: auto; }
  .panel { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .card { background: #1a1a2e; border-radius: 8px; padding: 16px; }
  .card h2 { font-size: 0.95rem; color: #8cf; margin-bottom: 10px; }
  .metric { display: flex; justify-content: space-between; padding: 4px 0;
            font-size: 0.85rem; border-bottom: 1px solid #2a2a3e; }
  .metric:last-child { border-bottom: none; }
  .metric .val { font-weight: 600; color: #6f6; font-family: monospace; }
  label { display: block; font-size: 0.82rem; margin: 8px 0 3px; }
  input[type=range] { width: 100%; }
  select, button { padding: 6px 14px; border-radius: 4px; border: 1px solid #444;
                   background: #222; color: #eee; cursor: pointer; }
  button { background: #2563eb; border-color: #2563eb; font-weight: 600; }
  button:hover { background: #1d4ed8; }
  .range-val { float: right; font-family: monospace; color: #6f6; }
  @media (max-width: 700px) { .panel { grid-template-columns: 1fr; } }
</style>
</head>
<body>
<div class="container">
  <h1>PC-to-Phone Fast Stream</h1>
  <div class="stream-wrap"><img src="/stream" alt="Live stream"></div>
  <div class="panel">
    <div class="card">
      <h2>Metrics</h2>
      <div id="metrics"></div>
    </div>
    <div class="card">
      <h2>Controls</h2>
      <label>Mode:
        <select id="mode" onchange="setMode(this.value)">
          <option value="adaptive">Adaptive</option>
          <option value="manual">Manual</option>
        </select>
      </label>
      <label>FPS: <span class="range-val" id="fps-val">11</span></label>
      <input type="range" id="fps" min="1" max="30" value="11" step="1"
             oninput="document.getElementById('fps-val').textContent=this.value">
      <label>Quality: <span class="range-val" id="q-val">70</span></label>
      <input type="range" id="quality" min="10" max="95" value="70" step="5"
             oninput="document.getElementById('q-val').textContent=this.value">
      <label>Width: <span class="range-val" id="w-val">1280</span></label>
      <input type="range" id="width" min="320" max="1920" value="1280" step="80"
             oninput="document.getElementById('w-val').textContent=this.value">
      <label>Height: <span class="range-val" id="h-val">920</span></label>
      <input type="range" id="height" min="240" max="1080" value="920" step="40"
             oninput="document.getElementById('h-val').textContent=this.value">
      <br>
      <button onclick="applySettings()">Apply</button>
      <button onclick="resetCtrl()" style="background:#666;border-color:#666">Reset</button>
    </div>
  </div>
</div>
<script>
function poll() {
  fetch('/api/metrics').then(r=>r.json()).then(d=>{
    let h='';
    for(let[k,v]of Object.entries(d))
      h+=`<div class="metric"><span>${k}</span><span class="val">${v}</span></div>`;
    document.getElementById('metrics').innerHTML=h;
    document.getElementById('mode').value=d.mode||'adaptive';
  }).catch(()=>{});
}
setInterval(poll,1000); poll();

function setMode(m){fetch('/api/mode',{method:'POST',headers:{'Content-Type':'application/json'},
  body:JSON.stringify({mode:m})})}

function applySettings(){
  fetch('/api/settings',{method:'POST',headers:{'Content-Type':'application/json'},
    body:JSON.stringify({
      fps:+document.getElementById('fps').value,
      quality:+document.getElementById('quality').value,
      width:+document.getElementById('width').value,
      height:+document.getElementById('height').value
    })
  }).then(r=>r.json()).then(d=>console.log('Applied:',d));
}
function resetCtrl(){fetch('/api/reset',{method:'POST'}).then(()=>poll())}
</script>
</body>
</html>"""


def generate():
    """Yield MJPEG frames using the adaptive stream controller."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]

        while True:
            start = time.time()

            img = sct.grab(monitor)
            frame = np.array(img)

            jpeg_bytes, frame_delay = controller.process_frame(frame)

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )

            elapsed = time.time() - start
            time.sleep(max(0, frame_delay - elapsed))


@app.route("/stream")
def stream():
    """MJPEG multipart stream endpoint."""
    return Response(
        generate(), mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/api/metrics")
def api_metrics():
    """Return current streaming metrics as JSON."""
    return jsonify(controller.get_metrics().to_dict())


@app.route("/api/mode", methods=["POST"])
def api_mode():
    """Switch between adaptive and manual mode."""
    data = request.get_json(force=True)
    mode = data.get("mode", "adaptive")
    try:
        controller.set_mode(mode)
        return jsonify({"mode": mode})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.route("/api/settings", methods=["POST"])
def api_settings():
    """Apply manual parameter overrides (FPS, quality, resolution)."""
    data = request.get_json(force=True)
    applied = controller.set_manual_params(
        fps=data.get("fps"),
        quality=data.get("quality"),
        width=data.get("width"),
        height=data.get("height"),
    )
    return jsonify({"applied": applied})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    """Reset controller state and metrics to defaults."""
    controller.reset()
    return jsonify({"status": "reset"})


if __name__ == "__main__":
    print("SERVER BASLADI")
    app.run(host="0.0.0.0", port=5000, threaded=True)
