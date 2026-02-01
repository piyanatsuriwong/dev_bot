# Pibot Voice System 🎤🤖

ระบบสนทนาด้วยเสียงสำหรับ Pibot บน Raspberry Pi

## Features

- 🎯 **Wake Word Detection** - ตรวจจับคำว่า "พีบอท" 
- 🎤 **Speech-to-Text** - แปลงเสียงเป็นข้อความ (Whisper)
- 🔊 **Text-to-Speech** - แปลงข้อความเป็นเสียง (Edge TTS)
- 🔗 **Clawdbot Integration** - ส่งข้อความเข้า Clawdbot Gateway

## Architecture

```
Microphone → Wake Word → STT → Clawdbot → TTS → Speaker
```

## Requirements

### Hardware (เสียบเมื่อพร้อม)
- USB Microphone หรือ I2S Mic HAT
- USB Speaker หรือ 3.5mm Audio

### Software
- Python 3.11+
- PortAudio (for PyAudio)
- ffmpeg

## Installation

```bash
cd /home/pi/clawd/pibot-voice
./scripts/install.sh
```

## Usage

```bash
# ทดสอบ TTS
python3 src/tts.py "สวัสดีครับ ผมคือพีบอท"

# ทดสอบ STT (ต้องมีไมค์)
python3 src/stt.py

# รันระบบเต็ม (ต้องมีไมค์+ลำโพง)
python3 src/main.py
```

## Configuration

แก้ไขได้ที่ `config/settings.yaml`

## Status

- [x] Project structure
- [x] TTS module (Edge TTS)
- [x] STT module (Whisper)
- [x] Clawdbot client
- [ ] Wake word detection
- [ ] Audio I/O management
- [ ] Main loop integration
