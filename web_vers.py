from flask import Flask, Response
import mss
import cv2
import numpy as np
import time

app = Flask(__name__)

@app.route('/')
def index():
    return '<img src="/stream">'

def generate():
    with mss.mss() as sct:
        monitor = sct.monitors[1]

        target_width = 1280   # 480p genişlik
        target_height = 920  # 480p yükseklik
        fps = 11
        frame_delay = 1 / fps

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 70]  # kalite

        while True:
            start = time.time()

            img = sct.grab(monitor)
            frame = np.array(img)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

            # Resize (en büyük kazanç burada)
            frame = cv2.resize(frame, (target_width, target_height))

            _, buffer = cv2.imencode('.jpg', frame, encode_param)
            frame_bytes = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   frame_bytes + b'\r\n')

            elapsed = time.time() - start
            time.sleep(max(0, frame_delay - elapsed))

@app.route('/stream')
def stream():
    return Response(generate(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    print("SERVER BASLADI")
    app.run(host='0.0.0.0', port=5000, threaded=True)
