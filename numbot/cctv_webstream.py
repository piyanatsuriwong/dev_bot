#!/usr/bin/env python3
"""CCTV Web Stream - Flask MJPEG Live Streaming Server
เปิดดู live feed ผ่าน browser ที่ http://<pi-ip>:8080
"""

import cv2
import time
import logging
import threading
from flask import Flask, Response, render_template_string

logger = logging.getLogger("cctv.webstream")

HTML_PAGE = """<!DOCTYPE html>
<html><head>
<title>NumBot CCTV - Live Stream</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>
body { background:#1a1a2e; color:#eee; font-family:Arial; text-align:center; margin:0; padding:20px; }
h1 { color:#00ff88; }
img { max-width:100%; border:2px solid #00ff88; border-radius:8px; }
.info { margin:15px; padding:10px; background:#16213e; border-radius:8px; }
.status { color:#00ff88; font-weight:bold; }
footer { margin-top:20px; color:#666; font-size:0.8em; }
</style>
<script>
setInterval(function(){
  fetch('/api/status').then(r=>r.json()).then(d=>{
    document.getElementById('det-count').textContent=d.total_detections;
    document.getElementById('fps').textContent=d.fps.toFixed(1);
    document.getElementById('uptime').textContent=d.uptime;
  }).catch(()=>{});
}, 2000);
</script>
</head><body>
<h1>📹 NumBot CCTV</h1>
<img src="/video_feed" alt="Live Stream">
<div class="info">
  <span class="status">● LIVE</span> |
  FPS: <span id="fps">--</span> |
  Detections: <span id="det-count">0</span> |
  Uptime: <span id="uptime">--</span>
</div>
<footer>NumBot CCTV System - Raspberry Pi 5</footer>
</body></html>"""


class WebStream:
    """Flask MJPEG streaming server"""
    
    def __init__(self, host="0.0.0.0", port=8080):
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self._frame = None
        self._frame_lock = threading.Lock()
        self._running = False
        self._thread = None
        self._start_time = time.time()
        self._total_detections = 0
        self._fps = 0.0
        
        self._setup_routes()
    
    def _setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template_string(HTML_PAGE)
        
        @self.app.route('/video_feed')
        def video_feed():
            return Response(
                self._generate_frames(),
                mimetype='multipart/x-mixed-replace; boundary=frame'
            )
        
        @self.app.route('/api/status')
        def api_status():
            import json
            uptime_sec = int(time.time() - self._start_time)
            h, m, s = uptime_sec // 3600, (uptime_sec % 3600) // 60, uptime_sec % 60
            return json.dumps({
                "total_detections": self._total_detections,
                "fps": self._fps,
                "uptime": f"{h:02d}:{m:02d}:{s:02d}",
                "running": self._running
            }), 200, {'Content-Type': 'application/json'}
        
        @self.app.route('/api/snapshot')
        def snapshot():
            with self._frame_lock:
                if self._frame is not None:
                    _, buf = cv2.imencode('.jpg', self._frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    return Response(buf.tobytes(), mimetype='image/jpeg')
            return "No frame available", 503
    
    def update_frame(self, frame):
        """Update the current frame (called from detection loop)"""
        with self._frame_lock:
            self._frame = frame
    
    def update_stats(self, fps, total_detections):
        self._fps = fps
        self._total_detections = total_detections
    
    def _generate_frames(self):
        """MJPEG frame generator"""
        while self._running:
            with self._frame_lock:
                if self._frame is None:
                    time.sleep(0.1)
                    continue
                _, buffer = cv2.imencode('.jpg', self._frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   buffer.tobytes() + b'\r\n')
            time.sleep(0.05)  # ~20 fps max
    
    def start(self):
        """Start web server in background thread"""
        if self._running:
            return
        self._running = True
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(f"Web stream started at http://{self.host}:{self.port}")
    
    def _run(self):
        self.app.run(host=self.host, port=self.port, threaded=True, use_reloader=False)
    
    def stop(self):
        self._running = False
        logger.info("Web stream stopped")
