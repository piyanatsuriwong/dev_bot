# 📹 NumBot CCTV - Person Detection System

ระบบกล้องวงจรปิดตรวจจับคน สำหรับ Raspberry Pi 5
พร้อมแจ้งเตือนผ่าน Telegram และดู Live ผ่าน Browser

## ✨ Features

- **🔍 Person Detection** — ตรวจจับคนด้วย HOG (เบา) หรือ YOLO (แม่นยำ)
- **📱 Telegram Alert** — ส่งรูป+แจ้งเตือนเมื่อตรวจพบคน
- **🌐 Live View** — ดูภาพ live ผ่าน browser (MJPEG streaming)
- **💾 Auto Save** — บันทึกรูปที่ตรวจพบอัตโนมัติ
- **⏰ Schedule** — ตั้งเวลาเฝ้าระวัง (เช่น 18:00-08:00)
- **🗄 Log Database** — เก็บ log ใน SQLite
- **🧹 Auto Cleanup** — ลบรูปเก่าอัตโนมัติเมื่อเกิน limit

## 🔧 Hardware

- Raspberry Pi 5
- กล้อง IMX708 (CSI) หรือ USB camera
- เชื่อมต่ออินเทอร์เน็ต (สำหรับ Telegram)

## 📦 Installation

```bash
cd /home/pi/dev_bot/numbot

# Install dependencies (ส่วนใหญ่มีอยู่แล้วบน RPi OS)
pip3 install flask requests

# ถ้ายังไม่มี opencv
pip3 install opencv-python
```

## ⚙️ Configuration

แก้ไข `cctv_config.json`:

```json
{
    "camera_num": 0,
    "resolution": [640, 480],
    "detection_method": "hog",
    "detection_threshold": 0.5,
    "alert_cooldown_seconds": 300,
    "web_stream": {
        "enabled": true,
        "port": 8080
    },
    "telegram": {
        "enabled": true,
        "bot_token": "123456:ABC-DEF...",
        "chat_id": "your_chat_id"
    }
}
```

### ตั้งค่า Telegram Bot

1. คุยกับ [@BotFather](https://t.me/BotFather) บน Telegram
2. สร้าง bot ด้วย `/newbot`
3. คัดลอก **Bot Token** มาใส่ใน config
4. หา **Chat ID** โดยส่งข้อความหา bot แล้วเปิด:
   `https://api.telegram.org/bot<TOKEN>/getUpdates`
5. ใส่ `chat_id` ใน config

## 🚀 Usage

```bash
# รันระบบ CCTV
python3 cctv_main.py

# ระบุ config อื่น
python3 cctv_main.py --config my_config.json

# ทดสอบ Telegram
python3 cctv_main.py --test-telegram
```

### เข้าดู Live View

เปิด browser แล้วไปที่:
```
http://<raspberry-pi-ip>:8080
```

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/` | หน้า Live Stream |
| `/video_feed` | MJPEG stream (สำหรับ embed) |
| `/api/status` | สถานะระบบ (JSON) |
| `/api/snapshot` | ภาพ snapshot ล่าสุด |

## 🏗 Architecture

```
cctv_main.py          ← Main loop + orchestration
├── cctv_detector.py   ← Person detection (HOG/YOLO)
├── cctv_telegram.py   ← Telegram Bot API alerts  
├── cctv_webstream.py  ← Flask MJPEG web server
└── cctv_config.json   ← Configuration
```

### Data Flow
```
Camera (Picamera2/OpenCV)
    ↓
Person Detection (HOG/YOLO)
    ↓
    ├→ Web Stream (Flask MJPEG) → Browser
    ├→ Telegram Alert (photo + text)
    ├→ Save to detections/ folder
    └→ Log to SQLite database
```

## 🔄 Auto-start (systemd)

```bash
sudo nano /etc/systemd/system/numbot-cctv.service
```

```ini
[Unit]
Description=NumBot CCTV Person Detection
After=network.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/dev_bot/numbot
ExecStart=/usr/bin/python3 /home/pi/dev_bot/numbot/cctv_main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable numbot-cctv
sudo systemctl start numbot-cctv
```

## 📋 Config Options

| Option | Default | Description |
|--------|---------|-------------|
| `camera_num` | 0 | Camera index (0=IMX708, 1=IMX500) |
| `resolution` | [640,480] | Capture resolution |
| `detection_method` | "hog" | "hog" (เบา) หรือ "yolo" (แม่นยำ) |
| `detection_threshold` | 0.5 | Confidence threshold |
| `min_area` | 3000 | Min detection area (pixels) |
| `monitoring_enabled` | false | เปิดตั้งเวลาเฝ้า |
| `monitoring_start` | "18:00" | เวลาเริ่มเฝ้า |
| `monitoring_end` | "08:00" | เวลาหยุดเฝ้า |
| `alert_cooldown_seconds` | 300 | Cooldown ระหว่างแจ้งเตือน (วินาที) |
| `save_detections` | true | บันทึกรูปที่ตรวจพบ |
| `max_detections_mb` | 500 | ขนาดสูงสุดของโฟลเดอร์ detections |

## 📝 License

Part of NumBot project — GPL License
