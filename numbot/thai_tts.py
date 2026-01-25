#!/usr/bin/env python3
"""
Thai TTS - Text-to-Speech for Object Detection
Speaks detected objects in Thai with queue to prevent overlapping

Usage:
    tts = ThaiTTS()
    tts.speak("cat")  # พูด "แมว"
    tts.speak("dog")  # รอคิว แล้วพูด "หมา"
"""

import threading
import queue
import time
import os

# Try to import TTS library
try:
    from gtts import gTTS
    import pygame
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False
    print("Warning: gTTS not available. Install with: pip install gtts")

# Try to import pydub for robot voice effect
try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:
    PYDUB_AVAILABLE = False
    print("Warning: pydub not available. Install with: pip install pydub")

# COCO labels to Thai translation
ENGLISH_TO_THAI = {
    # Animals
    "bird": "นก",
    "cat": "แมว",
    "dog": "หมา",
    "horse": "ม้า",
    "sheep": "แกะ",
    "cow": "วัว",
    "elephant": "ช้าง",
    "bear": "หมี",
    "zebra": "ม้าลาย",
    "giraffe": "ยีราฟ",
    
    # Vehicles
    "bicycle": "จักรยาน",
    "car": "รถยนต์",
    "motorcycle": "มอเตอร์ไซค์",
    "airplane": "เครื่องบิน",
    "bus": "รถบัส",
    "train": "รถไฟ",
    "truck": "รถบรรทุก",
    "boat": "เรือ",
    
    # Objects
    "traffic light": "ไฟจราจร",
    "fire hydrant": "หัวดับเพลิง",
    "stop sign": "ป้ายหยุด",
    "bench": "ม้านั่ง",
    "backpack": "กระเป๋าเป้",
    "umbrella": "ร่ม",
    "handbag": "กระเป๋าถือ",
    "tie": "เนคไท",
    "suitcase": "กระเป๋าเดินทาง",
    
    # Sports
    "frisbee": "จานร่อน",
    "skis": "สกี",
    "snowboard": "สโนว์บอร์ด",
    "sports ball": "ลูกบอล",
    "kite": "ว่าว",
    "baseball bat": "ไม้เบสบอล",
    "skateboard": "สเก็ตบอร์ด",
    "surfboard": "กระดานโต้คลื่น",
    "tennis racket": "ไม้เทนนิส",
    
    # Kitchen
    "bottle": "ขวด",
    "wine glass": "แก้วไวน์",
    "cup": "แก้ว",
    "fork": "ส้อม",
    "knife": "มีด",
    "spoon": "ช้อน",
    "bowl": "ชาม",
    
    # Food
    "banana": "กล้วย",
    "apple": "แอปเปิ้ล",
    "sandwich": "แซนวิช",
    "orange": "ส้ม",
    "broccoli": "บร็อคโคลี่",
    "carrot": "แครอท",
    "hot dog": "ฮอทด็อก",
    "pizza": "พิซซ่า",
    "donut": "โดนัท",
    "cake": "เค้ก",
    
    # Furniture
    "chair": "เก้าอี้",
    "couch": "โซฟา",
    "potted plant": "ต้นไม้",
    "bed": "เตียง",
    "dining table": "โต๊ะอาหาร",
    "toilet": "ห้องน้ำ",
    
    # Electronics
    "tv": "ทีวี",
    "laptop": "แล็ปท็อป",
    "mouse": "เมาส์",
    "remote": "รีโมท",
    "keyboard": "คีย์บอร์ด",
    "cell phone": "โทรศัพท์",
    "microwave": "ไมโครเวฟ",
    "oven": "เตาอบ",
    "toaster": "เครื่องปิ้งขนมปัง",
    "sink": "อ่างล้างจาน",
    "refrigerator": "ตู้เย็น",
    
    # Other
    "book": "หนังสือ",
    "clock": "นาฬิกา",
    "vase": "แจกัน",
    "scissors": "กรรไกร",
    "teddy bear": "ตุ๊กตาหมี",
    "hair drier": "ไดร์เป่าผม",
    "toothbrush": "แปรงสีฟัน",
}

# Objects to ignore (don't speak)
IGNORE_OBJECTS = {"person"}


class ThaiTTS:
    """Thai Text-to-Speech with queue to prevent overlapping"""
    
    def __init__(self, cache_dir="/tmp/tts_cache", robot_voice=True, pitch_shift=0.3):
        self.cache_dir = cache_dir
        self.robot_voice = robot_voice  # Enable robot voice effect
        self.pitch_shift = pitch_shift  # Pitch shift amount (0.3-0.5 = robot-like)
        self.speech_queue = queue.Queue()
        self.is_speaking = False
        self._running = True
        self._thread = None
        self._last_spoken = ""
        self._last_spoken_time = 0
        self._cooldown = 3.0  # Seconds before repeating same object
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)

        # Start speech thread (uses aplay, not pygame mixer)
        if GTTS_AVAILABLE:
            try:
                self._thread = threading.Thread(target=self._speech_loop, daemon=True)
                self._thread.start()
                robot_status = "ON" if robot_voice and PYDUB_AVAILABLE else "OFF"
                print(f"ThaiTTS: Initialized (Robot voice: {robot_status})")
            except Exception as e:
                print(f"ThaiTTS: Failed to initialize - {e}")
    
    def speak(self, english_text):
        """Add text to speech queue (non-blocking)"""
        if not GTTS_AVAILABLE:
            return
        
        # Get first object only
        if ":" in english_text:
            english_text = english_text.split(":")[0]
        
        # Skip ignored objects
        if english_text.lower() in IGNORE_OBJECTS:
            return
        
        # Check cooldown for same object
        now = time.time()
        if english_text == self._last_spoken and (now - self._last_spoken_time) < self._cooldown:
            return
        
        # Get Thai translation
        thai_text = ENGLISH_TO_THAI.get(english_text.lower(), english_text)
        
        # Add to queue (don't block)
        try:
            self.speech_queue.put_nowait(thai_text)
            self._last_spoken = english_text
            self._last_spoken_time = now
        except queue.Full:
            pass  # Skip if queue is full
    
    def _speech_loop(self):
        """Background thread to process speech queue"""
        while self._running:
            try:
                # Wait for text in queue
                thai_text = self.speech_queue.get(timeout=0.5)
                
                if thai_text:
                    self._speak_now(thai_text)
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"ThaiTTS: Error - {e}")
    
    def _speak_now(self, thai_text):
        """Generate and play speech using aplay (avoids pygame mixer conflict)"""
        try:
            # Cache file names (original and robotized)
            text_hash = abs(hash(thai_text))
            cache_file = os.path.join(self.cache_dir, f"{text_hash}.mp3")
            robot_cache_file = os.path.join(self.cache_dir, f"{text_hash}_robot.wav")
            wav_file = os.path.join(self.cache_dir, f"{text_hash}.wav")

            # Determine which file to play
            if self.robot_voice and PYDUB_AVAILABLE:
                play_file = robot_cache_file
            else:
                play_file = wav_file

            # Generate if not cached
            if not os.path.exists(play_file):
                # First generate normal speech
                if not os.path.exists(cache_file):
                    print(f"ThaiTTS: Generating speech for '{thai_text}'")
                    tts = gTTS(text=thai_text, lang='th')
                    tts.save(cache_file)

                # Apply robot effect if enabled
                if self.robot_voice and PYDUB_AVAILABLE:
                    self._robotize(cache_file, robot_cache_file)
                else:
                    # Convert mp3 to wav for aplay
                    if PYDUB_AVAILABLE:
                        sound = AudioSegment.from_mp3(cache_file)
                        sound.export(wav_file, format="wav")

            # Play audio using aplay (system command, no pygame conflict)
            self.is_speaking = True
            print(f"ThaiTTS: Playing '{thai_text}'")
            import subprocess
            subprocess.run(['aplay', '-q', play_file], check=False, timeout=10)
            self.is_speaking = False

        except Exception as e:
            print(f"ThaiTTS: Speech error - {e}")
            self.is_speaking = False
    
    def _robotize(self, input_file, output_file):
        """Apply robot voice effect by shifting pitch"""
        try:
            # Load audio
            sound = AudioSegment.from_mp3(input_file)
            
            # Shift pitch up (higher = more robotic)
            # Using sample rate change method
            new_sample_rate = int(sound.frame_rate * (2 ** self.pitch_shift))
            
            # Create pitch-shifted audio
            robot_sound = sound._spawn(sound.raw_data, overrides={
                'frame_rate': new_sample_rate
            })
            
            # Resample back to standard rate
            robot_sound = robot_sound.set_frame_rate(44100)
            
            # Export as WAV for faster loading
            robot_sound.export(output_file, format="wav")
            
        except Exception as e:
            print(f"ThaiTTS: Robotize error - {e}")
    
    def stop(self):
        """Stop TTS"""
        self._running = False
        # Kill any playing aplay process
        import subprocess
        subprocess.run(['pkill', '-9', 'aplay'], check=False, capture_output=True)
        if self._thread:
            self._thread.join(timeout=1.0)
        print("ThaiTTS: Stopped")


# Test
if __name__ == "__main__":
    print("Testing ThaiTTS...")
    
    tts = ThaiTTS()
    
    test_objects = ["cat", "dog", "cup", "person", "chair", "banana"]
    
    for obj in test_objects:
        print(f"Speaking: {obj}")
        tts.speak(obj)
        time.sleep(2)
    
    tts.stop()
    print("Done!")
