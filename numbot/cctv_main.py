#!/usr/bin/env python3
"""NumBot CCTV Main - ระบบ CCTV ตรวจจับคน + แจ้งเตือน Telegram + Live View
Usage:
    python3 cctv_main.py                    # รันด้วย config default
    python3 cctv_main.py --config my.json   # ระบุ config file
    python3 cctv_main.py --test-telegram    # ทดสอบส่ง Telegram
"""

import json
import os
import sys
import time
import signal
import logging
import sqlite3
import argparse
import threading
from datetime import datetime

import cv2

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('cctv.log', encoding='utf-8')
    ]
)
logger = logging.getLogger("cctv.main")

# Local imports
from cctv_detector import PersonDetector
from cctv_telegram import TelegramAlert
from cctv_webstream import WebStream


class CCTVSystem:
    """Main CCTV System - ตรวจจับคน + แจ้งเตือน + Live View"""
    
    def __init__(self, config_path="cctv_config.json"):
        self.config = self._load_config(config_path)
        self._running = False
        self._camera = None
        self._total_detections = 0
        self._fps = 0.0
        
        # Initialize components
        self._init_detector()
        self._init_telegram()
        self._init_webstream()
        self._init_database()
        self._init_detections_dir()
    
    def _load_config(self, path):
        if not os.path.exists(path):
            logger.error(f"Config not found: {path}")
            sys.exit(1)
        with open(path, 'r') as f:
            config = json.load(f)
        logger.info(f"Config loaded from {path}")
        return config
    
    def _init_detector(self):
        self.detector = PersonDetector(
            method=self.config.get("detection_method", "hog"),
            threshold=self.config.get("detection_threshold", 0.5),
            min_area=self.config.get("min_area", 3000)
        )
        logger.info(f"Detector: {self.config.get('detection_method', 'hog')}")
    
    def _init_telegram(self):
        tg = self.config.get("telegram", {})
        self.telegram = TelegramAlert(
            bot_token=tg.get("bot_token", ""),
            chat_id=tg.get("chat_id", ""),
            cooldown=self.config.get("alert_cooldown_seconds", 300)
        )
    
    def _init_webstream(self):
        ws = self.config.get("web_stream", {})
        if ws.get("enabled", True):
            self.webstream = WebStream(
                host=ws.get("host", "0.0.0.0"),
                port=ws.get("port", 8080)
            )
        else:
            self.webstream = None
    
    def _init_database(self):
        db_path = self.config.get("database", "cctv_log.db")
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.execute("""
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                num_persons INTEGER,
                confidence REAL,
                image_path TEXT,
                alert_sent INTEGER DEFAULT 0
            )
        """)
        self.db.commit()
        logger.info(f"Database: {db_path}")
    
    def _init_detections_dir(self):
        if self.config.get("save_detections", True):
            det_dir = self.config.get("detections_dir", "detections")
            os.makedirs(det_dir, exist_ok=True)
            self._det_dir = det_dir
        else:
            self._det_dir = None
    
    def _init_camera(self):
        """Initialize Picamera2 or fallback to OpenCV"""
        cam_num = self.config.get("camera_num", 0)
        res = self.config.get("resolution", [640, 480])
        
        try:
            from picamera2 import Picamera2
            self._camera = Picamera2(camera_num=cam_num)
            config = self._camera.create_preview_configuration(
                main={"size": tuple(res), "format": "RGB888"}
            )
            self._camera.configure(config)
            self._camera.start()
            self._cam_type = "picamera2"
            logger.info(f"Picamera2 started (cam {cam_num}, {res[0]}x{res[1]})")
        except Exception as e:
            logger.warning(f"Picamera2 failed: {e}, trying OpenCV VideoCapture")
            self._camera = cv2.VideoCapture(cam_num)
            self._camera.set(cv2.CAP_PROP_FRAME_WIDTH, res[0])
            self._camera.set(cv2.CAP_PROP_FRAME_HEIGHT, res[1])
            if not self._camera.isOpened():
                logger.error("No camera available!")
                sys.exit(1)
            self._cam_type = "opencv"
            logger.info(f"OpenCV camera started (cam {cam_num})")
    
    def _capture_frame(self):
        """Capture one frame from camera"""
        if self._cam_type == "picamera2":
            frame = self._camera.capture_array("main")
            # Picamera2 returns RGB, OpenCV uses BGR
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return frame
        else:
            ret, frame = self._camera.read()
            return frame if ret else None
    
    def _is_monitoring_time(self):
        """Check if current time is within monitoring schedule"""
        if not self.config.get("monitoring_enabled", False):
            return True  # Always monitor if schedule disabled
        
        now = datetime.now().strftime("%H:%M")
        start = self.config.get("monitoring_start", "18:00")
        end = self.config.get("monitoring_end", "08:00")
        
        if start <= end:
            return start <= now <= end
        else:  # Overnight (e.g., 18:00 - 08:00)
            return now >= start or now <= end
    
    def _save_detection(self, frame, num_persons, max_conf):
        """Save detection image and log to database"""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_path = None
        
        if self._det_dir:
            img_path = os.path.join(self._det_dir, f"det_{ts}.jpg")
            cv2.imwrite(img_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        
        self.db.execute(
            "INSERT INTO detections (timestamp, num_persons, confidence, image_path) VALUES (?,?,?,?)",
            (ts, num_persons, max_conf, img_path)
        )
        self.db.commit()
        return img_path
    
    def _cleanup_old_detections(self):
        """Remove old detection images if exceeding max size"""
        max_mb = self.config.get("max_detections_mb", 500)
        if not self._det_dir or not os.path.exists(self._det_dir):
            return
        
        total_size = 0
        files = []
        for f in os.listdir(self._det_dir):
            fp = os.path.join(self._det_dir, f)
            if os.path.isfile(fp):
                sz = os.path.getsize(fp)
                total_size += sz
                files.append((fp, os.path.getmtime(fp), sz))
        
        if total_size > max_mb * 1024 * 1024:
            files.sort(key=lambda x: x[1])  # oldest first
            while total_size > max_mb * 1024 * 1024 * 0.8 and files:
                fp, _, sz = files.pop(0)
                os.remove(fp)
                total_size -= sz
                logger.info(f"Cleaned old detection: {fp}")
    
    def run(self):
        """Main detection loop"""
        logger.info("=" * 50)
        logger.info("NumBot CCTV System Starting...")
        logger.info("=" * 50)
        
        # Init camera
        self._init_camera()
        
        # Start web stream
        if self.webstream:
            self.webstream.start()
        
        # Notify Telegram
        self.telegram.send_startup_message()
        
        # Signal handlers
        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Detection loop
        frame_count = 0
        fps_start = time.time()
        cleanup_counter = 0
        
        logger.info("Detection loop started. Press Ctrl+C to stop.")
        
        while self._running:
            # Check monitoring schedule
            if not self._is_monitoring_time():
                time.sleep(10)
                continue
            
            # Capture frame
            frame = self._capture_frame()
            if frame is None:
                logger.warning("Failed to capture frame")
                time.sleep(1)
                continue
            
            # Detect persons
            detections = self.detector.detect(frame)
            annotated = self.detector.draw_detections(frame, detections)
            
            # Update web stream
            if self.webstream:
                self.webstream.update_frame(annotated)
            
            # FPS calculation
            frame_count += 1
            elapsed = time.time() - fps_start
            if elapsed >= 2.0:
                self._fps = frame_count / elapsed
                frame_count = 0
                fps_start = time.time()
                if self.webstream:
                    self.webstream.update_stats(self._fps, self._total_detections)
            
            # Person detected!
            if len(detections) > 0:
                num = len(detections)
                max_conf = max(d[4] for d in detections)
                self._total_detections += num
                
                logger.info(f"🚨 {num} person(s) detected! (conf: {max_conf:.1%})")
                
                # Save detection
                img_path = self._save_detection(annotated, num, max_conf)
                
                # Send Telegram alert
                if self.telegram.can_send():
                    self.telegram.send_photo(annotated, num_persons=num)
                    # Update DB
                    self.db.execute(
                        "UPDATE detections SET alert_sent=1 WHERE image_path=?",
                        (img_path,)
                    )
                    self.db.commit()
            
            # Periodic cleanup
            cleanup_counter += 1
            if cleanup_counter >= 1000:
                cleanup_counter = 0
                threading.Thread(target=self._cleanup_old_detections, daemon=True).start()
            
            # Control frame rate (~10 fps for detection)
            time.sleep(0.1)
        
        self._shutdown()
    
    def _signal_handler(self, sig, frame):
        logger.info("Shutdown signal received...")
        self._running = False
    
    def _shutdown(self):
        logger.info("Shutting down CCTV system...")
        
        if self.webstream:
            self.webstream.stop()
        
        if self._cam_type == "picamera2":
            try:
                self._camera.stop()
            except:
                pass
        else:
            try:
                self._camera.release()
            except:
                pass
        
        self.db.close()
        self.telegram.send_text("📹 CCTV System stopped.")
        logger.info("CCTV system stopped.")


def main():
    parser = argparse.ArgumentParser(description="NumBot CCTV Person Detection System")
    parser.add_argument("--config", default="cctv_config.json", help="Config file path")
    parser.add_argument("--test-telegram", action="store_true", help="Test Telegram connection")
    args = parser.parse_args()
    
    if args.test_telegram:
        config = json.load(open(args.config))
        tg = config.get("telegram", {})
        alert = TelegramAlert(tg.get("bot_token"), tg.get("chat_id"))
        if alert.test_connection():
            print("✅ Telegram bot connected!")
            alert.send_text("🧪 Test message from NumBot CCTV")
            print("✅ Test message sent!")
        else:
            print("❌ Telegram connection failed!")
        return
    
    cctv = CCTVSystem(config_path=args.config)
    cctv.run()


if __name__ == "__main__":
    main()
