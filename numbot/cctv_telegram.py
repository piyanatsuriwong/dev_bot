#!/usr/bin/env python3
"""CCTV Telegram Alert - ส่งรูปแจ้งเตือนผ่าน Telegram Bot API
ใช้ requests เท่านั้น ไม่ต้องลง lib เพิ่ม
"""

import requests
import cv2
import time
import logging
import io

logger = logging.getLogger("cctv.telegram")

API_URL = "https://api.telegram.org/bot{token}/{method}"


class TelegramAlert:
    """Send alerts via Telegram Bot API"""
    
    def __init__(self, bot_token, chat_id, cooldown=300):
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.cooldown = cooldown
        self._last_alert_time = 0
        self._enabled = bool(bot_token and chat_id 
                            and bot_token != "YOUR_BOT_TOKEN_HERE"
                            and chat_id != "YOUR_CHAT_ID_HERE")
        
        if self._enabled:
            logger.info(f"Telegram alerts enabled (chat_id: {chat_id})")
        else:
            logger.warning("Telegram alerts DISABLED - set bot_token and chat_id in config")
    
    @property
    def is_enabled(self):
        return self._enabled
    
    def can_send(self):
        """Check if cooldown has passed"""
        return time.time() - self._last_alert_time >= self.cooldown
    
    def send_text(self, text):
        """Send text message"""
        if not self._enabled:
            return False
        
        try:
            url = API_URL.format(token=self.bot_token, method="sendMessage")
            resp = requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text,
                "parse_mode": "HTML"
            }, timeout=10)
            resp.raise_for_status()
            logger.info(f"Telegram text sent: {text[:50]}")
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False
    
    def send_photo(self, frame, caption=None, num_persons=0):
        """Send photo with detection results"""
        if not self._enabled:
            return False
        
        if not self.can_send():
            remaining = self.cooldown - (time.time() - self._last_alert_time)
            logger.debug(f"Cooldown active, {remaining:.0f}s remaining")
            return False
        
        try:
            # Encode frame to JPEG
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            photo_bytes = io.BytesIO(buffer.tobytes())
            photo_bytes.name = "detection.jpg"
            
            if caption is None:
                ts = time.strftime("%Y-%m-%d %H:%M:%S")
                caption = (
                    f"🚨 <b>ตรวจพบคน!</b>\n"
                    f"👤 จำนวน: {num_persons} คน\n"
                    f"🕐 เวลา: {ts}"
                )
            
            url = API_URL.format(token=self.bot_token, method="sendPhoto")
            resp = requests.post(url, data={
                "chat_id": self.chat_id,
                "caption": caption,
                "parse_mode": "HTML"
            }, files={
                "photo": photo_bytes
            }, timeout=30)
            resp.raise_for_status()
            
            self._last_alert_time = time.time()
            logger.info(f"Telegram photo sent: {num_persons} person(s) detected")
            return True
        except Exception as e:
            logger.error(f"Telegram photo send failed: {e}")
            return False
    
    def send_startup_message(self):
        """Send system startup notification"""
        import socket
        hostname = socket.gethostname()
        try:
            ip = socket.gethostbyname(hostname)
        except:
            ip = "unknown"
        
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        text = (
            f"📹 <b>CCTV System Started</b>\n"
            f"🖥 Host: {hostname}\n"
            f"🌐 IP: {ip}\n"
            f"🕐 Time: {ts}\n"
            f"✅ Person detection active"
        )
        return self.send_text(text)
    
    def test_connection(self):
        """Test bot connection"""
        if not self._enabled:
            return False
        try:
            url = API_URL.format(token=self.bot_token, method="getMe")
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("ok"):
                bot_name = data["result"].get("username", "unknown")
                logger.info(f"Telegram bot connected: @{bot_name}")
                return True
        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
        return False
