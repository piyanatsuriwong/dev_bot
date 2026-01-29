#!/usr/bin/env python3
"""
Mecanum Motor Web Controller
Control the robot via web browser at http://<pi_ip>:8080
"""

import http.server
import socketserver
import json
import threading
import time

# Try to import Motor, mock if not available
try:
    from Motor import PWM
    MOTOR_AVAILABLE = True
except ImportError:
    MOTOR_AVAILABLE = False
    print("Warning: Motor module not available, running in simulation mode")

PORT = 8080
SPEED = 1500  # Default speed

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Mecanum Robot Control</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body {
            font-family: Arial, sans-serif;
            background: #1a1a2e;
            color: white;
            min-height: 100vh;
            display: flex;
            flex-direction: column;
            align-items: center;
            padding: 20px;
        }
        h1 { margin-bottom: 20px; color: #00d4ff; }
        .status {
            background: #16213e;
            padding: 10px 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            font-size: 18px;
        }

        .main-container {
            display: flex;
            gap: 30px;
            align-items: flex-start;
            flex-wrap: wrap;
            justify-content: center;
        }

        /* Robot Wheel Display */
        .robot-display {
            background: #16213e;
            padding: 20px;
            border-radius: 15px;
            text-align: center;
        }
        .robot-display h3 {
            margin-bottom: 15px;
            color: #00d4ff;
        }
        .robot-body {
            position: relative;
            width: 160px;
            height: 200px;
            background: #2d3a4f;
            border-radius: 20px;
            margin: 0 auto;
            border: 3px solid #4a5568;
        }
        .robot-front {
            position: absolute;
            top: 10px;
            left: 50%;
            transform: translateX(-50%);
            font-size: 12px;
            color: #888;
        }
        .wheel {
            position: absolute;
            width: 35px;
            height: 55px;
            background: #1a1a2e;
            border: 3px solid #4a5568;
            border-radius: 8px;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            transition: all 0.2s;
        }
        .wheel-label {
            font-size: 10px;
            font-weight: bold;
            color: #888;
        }
        .wheel-arrow {
            font-size: 20px;
            color: #888;
            transition: all 0.2s;
        }
        .wheel.forward {
            background: #00aa00;
            border-color: #00ff00;
            box-shadow: 0 0 15px #00ff00;
        }
        .wheel.forward .wheel-arrow {
            color: white;
        }
        .wheel.backward {
            background: #aa0000;
            border-color: #ff0000;
            box-shadow: 0 0 15px #ff0000;
        }
        .wheel.backward .wheel-arrow {
            color: white;
        }
        .wheel.stopped {
            background: #333;
            border-color: #555;
        }
        .wheel.stopped .wheel-arrow {
            color: #555;
        }
        .wheel-fl { top: 15px; left: -20px; }
        .wheel-fr { top: 15px; right: -20px; }
        .wheel-rl { bottom: 15px; left: -20px; }
        .wheel-rr { bottom: 15px; right: -20px; }

        .direction-arrow {
            position: absolute;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            font-size: 50px;
            color: #00d4ff;
            text-shadow: 0 0 20px #00d4ff;
            opacity: 0;
            transition: opacity 0.2s;
        }
        .direction-arrow.visible {
            opacity: 1;
        }

        .control-pad {
            display: grid;
            grid-template-columns: repeat(3, 70px);
            grid-template-rows: repeat(3, 70px);
            gap: 8px;
            margin-bottom: 15px;
        }
        .btn {
            background: #0f3460;
            border: 2px solid #00d4ff;
            border-radius: 12px;
            color: white;
            font-size: 22px;
            cursor: pointer;
            transition: all 0.2s;
            display: flex;
            align-items: center;
            justify-content: center;
        }
        .btn:hover { background: #1a5490; }
        .btn:active, .btn.active {
            background: #00d4ff;
            color: #1a1a2e;
            transform: scale(0.95);
        }
        .btn-stop {
            background: #e94560;
            border-color: #ff6b6b;
        }
        .btn-stop:hover { background: #ff6b6b; }

        .rotate-section {
            display: flex;
            gap: 15px;
            margin-bottom: 15px;
        }
        .rotate-btn {
            width: 90px;
            height: 50px;
            font-size: 14px;
        }

        .speed-control {
            background: #16213e;
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        .speed-control input {
            width: 180px;
            margin: 8px 0;
        }
        .speed-value {
            font-size: 20px;
            color: #00d4ff;
        }

        .commands {
            margin-top: 15px;
            display: flex;
            gap: 10px;
        }
        .cmd-btn {
            padding: 12px 20px;
            font-size: 13px;
        }

        .legend {
            margin-top: 15px;
            font-size: 12px;
            color: #888;
        }
        .legend span {
            margin: 0 10px;
        }
        .legend .fwd { color: #00ff00; }
        .legend .bwd { color: #ff0000; }
    </style>
</head>
<body>
    <h1>🤖 Mecanum Robot Control</h1>
    <div class="status">Status: <span id="status">Ready</span></div>

    <div class="main-container">
        <!-- Robot Wheel Display -->
        <div class="robot-display">
            <h3>Wheel Status</h3>
            <div class="robot-body">
                <div class="robot-front">▲ FRONT</div>
                <div class="wheel wheel-fl" id="wheel-fl">
                    <span class="wheel-label">FL</span>
                    <span class="wheel-arrow">●</span>
                </div>
                <div class="wheel wheel-fr" id="wheel-fr">
                    <span class="wheel-label">FR</span>
                    <span class="wheel-arrow">●</span>
                </div>
                <div class="wheel wheel-rl" id="wheel-rl">
                    <span class="wheel-label">RL</span>
                    <span class="wheel-arrow">●</span>
                </div>
                <div class="wheel wheel-rr" id="wheel-rr">
                    <span class="wheel-label">RR</span>
                    <span class="wheel-arrow">●</span>
                </div>
                <div class="direction-arrow" id="dirArrow">↑</div>
            </div>
            <div class="legend">
                <span class="fwd">■ Forward</span>
                <span class="bwd">■ Backward</span>
                <span>■ Stop</span>
            </div>
        </div>

        <!-- Controls -->
        <div class="controls">
            <!-- Main D-Pad with Diagonals -->
            <div class="control-pad">
                <button class="btn" onmousedown="send('diagonal_forward_left')" onmouseup="send('stop')" ontouchstart="send('diagonal_forward_left')" ontouchend="send('stop')">↖</button>
                <button class="btn" onmousedown="send('forward')" onmouseup="send('stop')" ontouchstart="send('forward')" ontouchend="send('stop')">↑</button>
                <button class="btn" onmousedown="send('diagonal_forward_right')" onmouseup="send('stop')" ontouchstart="send('diagonal_forward_right')" ontouchend="send('stop')">↗</button>

                <button class="btn" onmousedown="send('strafe_left')" onmouseup="send('stop')" ontouchstart="send('strafe_left')" ontouchend="send('stop')">←</button>
                <button class="btn btn-stop" onclick="send('stop')">⬛</button>
                <button class="btn" onmousedown="send('strafe_right')" onmouseup="send('stop')" ontouchstart="send('strafe_right')" ontouchend="send('stop')">→</button>

                <button class="btn" onmousedown="send('diagonal_backward_left')" onmouseup="send('stop')" ontouchstart="send('diagonal_backward_left')" ontouchend="send('stop')">↙</button>
                <button class="btn" onmousedown="send('backward')" onmouseup="send('stop')" ontouchstart="send('backward')" ontouchend="send('stop')">↓</button>
                <button class="btn" onmousedown="send('diagonal_backward_right')" onmouseup="send('stop')" ontouchstart="send('diagonal_backward_right')" ontouchend="send('stop')">↘</button>
            </div>

            <!-- Rotation Buttons -->
            <div class="rotate-section">
                <button class="btn rotate-btn" onmousedown="send('rotate_left')" onmouseup="send('stop')" ontouchstart="send('rotate_left')" ontouchend="send('stop')">↺ Left</button>
                <button class="btn rotate-btn" onmousedown="send('rotate_right')" onmouseup="send('stop')" ontouchstart="send('rotate_right')" ontouchend="send('stop')">↻ Right</button>
            </div>

            <!-- Speed Control -->
            <div class="speed-control">
                <div>Speed: <span class="speed-value" id="speedValue">1500</span></div>
                <input type="range" id="speedSlider" min="500" max="4000" value="1500" oninput="updateSpeed(this.value)">
                <div style="display:flex; justify-content:space-between; width:180px; margin:auto;">
                    <span>Slow</span><span>Fast</span>
                </div>
            </div>

            <!-- Quick Commands -->
            <div class="commands">
                <button class="btn cmd-btn" onclick="send('demo')">🎬 Demo</button>
                <button class="btn cmd-btn" onclick="send('circle')">⭕ Circle</button>
            </div>
        </div>
    </div>

    <script>
        let currentSpeed = 1500;

        // Wheel patterns: [FL, RL, FR, RR] where 1=forward, -1=backward, 0=stop
        const wheelPatterns = {
            'stop':                    [0, 0, 0, 0, '●'],
            'forward':                 [1, 1, 1, 1, '↑'],
            'backward':                [-1, -1, -1, -1, '↓'],
            'strafe_left':             [-1, 1, 1, -1, '←'],
            'strafe_right':            [1, -1, -1, 1, '→'],
            'rotate_left':             [-1, -1, 1, 1, '↺'],
            'rotate_right':            [1, 1, -1, -1, '↻'],
            'diagonal_forward_left':   [0, 1, 1, 0, '↖'],
            'diagonal_forward_right':  [1, 0, 0, 1, '↗'],
            'diagonal_backward_left':  [-1, 0, 0, -1, '↙'],
            'diagonal_backward_right': [0, -1, -1, 0, '↘'],
        };

        function updateWheelDisplay(cmd) {
            const pattern = wheelPatterns[cmd] || wheelPatterns['stop'];
            const wheels = ['wheel-fl', 'wheel-rl', 'wheel-fr', 'wheel-rr'];
            const arrows = ['↑', '↑', '↑', '↑'];

            wheels.forEach((id, i) => {
                const el = document.getElementById(id);
                const arrowEl = el.querySelector('.wheel-arrow');
                el.classList.remove('forward', 'backward', 'stopped');

                if (pattern[i] === 1) {
                    el.classList.add('forward');
                    arrowEl.textContent = '↑';
                } else if (pattern[i] === -1) {
                    el.classList.add('backward');
                    arrowEl.textContent = '↓';
                } else {
                    el.classList.add('stopped');
                    arrowEl.textContent = '●';
                }
            });

            // Update direction arrow
            const dirArrow = document.getElementById('dirArrow');
            if (cmd === 'stop') {
                dirArrow.classList.remove('visible');
            } else {
                dirArrow.textContent = pattern[4];
                dirArrow.classList.add('visible');
            }
        }

        function send(cmd) {
            const status = document.getElementById('status');
            status.textContent = cmd.replace(/_/g, ' ').toUpperCase();

            updateWheelDisplay(cmd);

            fetch('/cmd?action=' + cmd + '&speed=' + currentSpeed)
                .then(r => r.json())
                .then(data => {
                    if (cmd === 'stop') status.textContent = 'Ready';
                })
                .catch(e => status.textContent = 'Error: ' + e);
        }

        function updateSpeed(val) {
            currentSpeed = parseInt(val);
            document.getElementById('speedValue').textContent = val;
            fetch('/cmd?action=set_speed&speed=' + val);
        }

        // Keyboard controls
        document.addEventListener('keydown', (e) => {
            if (e.repeat) return;
            const keyMap = {
                'ArrowUp': 'forward',
                'ArrowDown': 'backward',
                'ArrowLeft': 'strafe_left',
                'ArrowRight': 'strafe_right',
                'q': 'diagonal_forward_left',
                'e': 'diagonal_forward_right',
                'z': 'diagonal_backward_left',
                'c': 'diagonal_backward_right',
                'a': 'rotate_left',
                'd': 'rotate_right',
                ' ': 'stop'
            };
            if (keyMap[e.key]) {
                e.preventDefault();
                send(keyMap[e.key]);
            }
        });

        document.addEventListener('keyup', (e) => {
            const moveKeys = ['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight',
                             'q', 'e', 'z', 'c', 'a', 'd'];
            if (moveKeys.includes(e.key)) {
                send('stop');
            }
        });

        // Prevent context menu on long press (mobile)
        document.addEventListener('contextmenu', e => e.preventDefault());

        // Initialize wheel display
        updateWheelDisplay('stop');
    </script>
</body>
</html>
"""

class MotorController:
    def __init__(self):
        self.speed = SPEED
        self.running = False

    def execute(self, action, speed=None):
        if speed:
            self.speed = int(speed)

        if not MOTOR_AVAILABLE:
            print(f"[SIM] {action} at speed {self.speed}")
            return {"status": "simulated", "action": action}

        try:
            if action == "stop":
                PWM.stop()
            elif action == "forward":
                PWM.forward(self.speed)
            elif action == "backward":
                PWM.backward(self.speed)
            elif action == "strafe_left":
                PWM.strafe_left(self.speed)
            elif action == "strafe_right":
                PWM.strafe_right(self.speed)
            elif action == "rotate_left":
                PWM.rotate_left(self.speed)
            elif action == "rotate_right":
                PWM.rotate_right(self.speed)
            elif action == "diagonal_forward_left":
                PWM.diagonal_forward_left(self.speed)
            elif action == "diagonal_forward_right":
                PWM.diagonal_forward_right(self.speed)
            elif action == "diagonal_backward_left":
                PWM.diagonal_backward_left(self.speed)
            elif action == "diagonal_backward_right":
                PWM.diagonal_backward_right(self.speed)
            elif action == "set_speed":
                PWM.set_speed(self.speed)
            elif action == "demo":
                self.run_demo()
            elif action == "circle":
                self.run_circle()
            else:
                return {"status": "error", "message": f"Unknown action: {action}"}

            print(f"[MOTOR] {action} at speed {self.speed}")
            return {"status": "ok", "action": action, "speed": self.speed}

        except Exception as e:
            print(f"[ERROR] {e}")
            return {"status": "error", "message": str(e)}

    def run_demo(self):
        """Run quick demo in background thread"""
        def demo():
            moves = [
                ("forward", 0.5),
                ("backward", 0.5),
                ("strafe_left", 0.5),
                ("strafe_right", 0.5),
                ("diagonal_forward_left", 0.5),
                ("diagonal_forward_right", 0.5),
                ("rotate_left", 0.5),
                ("rotate_right", 0.5),
            ]
            for action, duration in moves:
                self.execute(action)
                time.sleep(duration)
                PWM.stop()
                time.sleep(0.2)

        thread = threading.Thread(target=demo, daemon=True)
        thread.start()

    def run_circle(self):
        """Move in a circle pattern"""
        def circle():
            for angle in range(0, 360, 30):
                PWM.move_angle(angle, speed=0.6)
                time.sleep(0.3)
            PWM.stop()

        thread = threading.Thread(target=circle, daemon=True)
        thread.start()

motor = MotorController()

class Handler(http.server.SimpleHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_PAGE.encode())

        elif self.path.startswith("/cmd"):
            # Parse query parameters
            from urllib.parse import urlparse, parse_qs
            query = parse_qs(urlparse(self.path).query)
            action = query.get("action", ["stop"])[0]
            speed = query.get("speed", [None])[0]

            result = motor.execute(action, speed)

            self.send_response(200)
            self.send_header("Content-type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode())

        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress logging for cleaner output
        pass

def get_ip():
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip

if __name__ == "__main__":
    ip = get_ip()
    print("=" * 50)
    print("  Mecanum Motor Web Controller")
    print("=" * 50)
    print(f"  Open in browser: http://{ip}:{PORT}")
    print()
    print("  Keyboard Controls:")
    print("    Arrow Keys = Move")
    print("    Q/E = Diagonal Forward")
    print("    Z/C = Diagonal Backward")
    print("    A/D = Rotate")
    print("    Space = Stop")
    print()
    print("  Press Ctrl+C to exit")
    print("=" * 50)

    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down...")
            if MOTOR_AVAILABLE:
                PWM.stop()
