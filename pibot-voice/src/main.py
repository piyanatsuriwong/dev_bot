#!/usr/bin/env python3
"""
Pibot Voice - Main Entry Point
Voice interaction system for Pibot on Raspberry Pi
"""

import asyncio
import signal
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import yaml
from dataclasses import dataclass
from typing import Optional

from tts import TTSEngine
from stt import STTEngine
from audio import AudioManager, AudioConfig
from clawdbot_client import ClawdbotClient


@dataclass
class VoiceState:
    """Voice system state"""
    IDLE = "idle"
    LISTENING = "listening"
    PROCESSING = "processing"
    SPEAKING = "speaking"


class PibotVoice:
    """
    Main voice interaction controller
    
    Flow:
    1. Wait for wake word (or continuous mode)
    2. Record user speech
    3. STT: Convert to text
    4. Send to Clawdbot
    5. TTS: Speak response
    6. Repeat
    """
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.state = VoiceState.IDLE
        self._running = False
        
        # Initialize components
        self._init_components()
    
    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from YAML"""
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
        
        if Path(config_path).exists():
            with open(config_path) as f:
                return yaml.safe_load(f)
        else:
            print(f"⚠️ Config not found: {config_path}, using defaults")
            return {}
    
    def _init_components(self):
        """Initialize all voice components"""
        # Audio
        audio_cfg = self.config.get("audio", {})
        self.audio = AudioManager(AudioConfig(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1)
        ))
        
        # TTS
        tts_cfg = self.config.get("tts", {})
        self.tts = TTSEngine(
            voice=tts_cfg.get("voice", "th-TH-PremwadeeNeural"),
            rate=tts_cfg.get("rate", "+0%"),
            volume=tts_cfg.get("volume", "+0%")
        )
        
        # STT
        stt_cfg = self.config.get("stt", {})
        self.stt = STTEngine(
            model=stt_cfg.get("model", "base"),
            language=stt_cfg.get("language", "th"),
            use_api=stt_cfg.get("use_api", False)
        )
        
        # Clawdbot client
        cb_cfg = self.config.get("clawdbot", {})
        self.clawdbot = ClawdbotClient(
            gateway_url=cb_cfg.get("gateway_url", "http://localhost:18789"),
            timeout=cb_cfg.get("timeout", 60)
        )
    
    async def run(self):
        """Main run loop"""
        print("🤖 Pibot Voice System")
        print("=" * 40)
        
        # Check components
        if not await self._check_components():
            return
        
        # Setup signal handlers
        self._setup_signals()
        
        self._running = True
        behavior = self.config.get("behavior", {})
        continuous = behavior.get("continuous_listen", True)
        
        print("\n🎤 Ready! Say 'พีบอท' to start...")
        print("   Press Ctrl+C to exit\n")
        
        while self._running:
            try:
                await self._interaction_loop(continuous)
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(1)
        
        print("\n👋 Shutting down...")
        self._cleanup()
    
    async def _check_components(self) -> bool:
        """Check if all components are available"""
        print("🔍 Checking components...")
        
        # Check Clawdbot
        if not await self.clawdbot.check_health():
            print("❌ Clawdbot Gateway not available")
            print("   Run: clawdbot gateway start")
            return False
        print("  ✅ Clawdbot Gateway")
        
        # Check audio (warning only)
        if not self.audio.has_microphone():
            print("  ⚠️ No microphone found (connect USB mic)")
        else:
            print("  ✅ Microphone")
        
        if not self.audio.has_speaker():
            print("  ⚠️ No speaker found (connect USB speaker)")
        else:
            print("  ✅ Speaker")
        
        print("  ✅ TTS (Edge)")
        print("  ✅ STT (Whisper)")
        
        return True
    
    async def _interaction_loop(self, continuous: bool = True):
        """Single interaction cycle"""
        behavior = self.config.get("behavior", {})
        silence_threshold = behavior.get("silence_threshold", 1500) / 1000.0
        
        # 1. Wait for wake word (simplified: just start recording)
        self.state = VoiceState.IDLE
        
        # For now, just listen directly (wake word TODO)
        # In future: await self.wake_word.listen()
        
        # Play listening sound
        if behavior.get("listen_sound", True):
            await self.tts.speak("ครับ")
        
        # 2. Record user speech
        self.state = VoiceState.LISTENING
        print("\n🎤 Listening...")
        
        audio_path = await self.audio.record(
            silence_threshold=silence_threshold,
            max_duration=30.0
        )
        
        if not audio_path:
            print("❌ Recording failed")
            return
        
        # 3. STT
        self.state = VoiceState.PROCESSING
        print("🔄 Transcribing...")
        
        text = await self.stt.transcribe(audio_path)
        
        if not text:
            print("❌ Could not transcribe audio")
            await self.tts.speak("ขอโทษครับ ไม่ได้ยินชัด")
            return
        
        print(f"👤 You: {text}")
        
        # 4. Send to Clawdbot
        print("🤔 Thinking...")
        
        response = await self.clawdbot.send_message(text)
        
        if not response:
            print("❌ No response from Clawdbot")
            await self.tts.speak("ขอโทษครับ เกิดข้อผิดพลาด")
            return
        
        print(f"🤖 Pibot: {response}")
        
        # 5. TTS
        self.state = VoiceState.SPEAKING
        await self.tts.speak(response)
        
        # 6. Back to idle
        self.state = VoiceState.IDLE
        
        if not continuous:
            self._running = False
    
    def _setup_signals(self):
        """Setup signal handlers for graceful shutdown"""
        def handle_signal(sig, frame):
            print("\n⚡ Signal received, stopping...")
            self._running = False
        
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    
    def _cleanup(self):
        """Cleanup resources"""
        self.audio.cleanup()


async def main():
    """CLI entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pibot Voice System")
    parser.add_argument(
        "-c", "--config",
        help="Path to config file",
        default=None
    )
    parser.add_argument(
        "--test-tts",
        help="Test TTS with given text",
        metavar="TEXT"
    )
    parser.add_argument(
        "--test-stt",
        help="Test STT with given audio file",
        metavar="FILE"
    )
    parser.add_argument(
        "--list-devices",
        help="List audio devices",
        action="store_true"
    )
    
    args = parser.parse_args()
    
    if args.list_devices:
        audio = AudioManager()
        print("🔊 Audio Devices:")
        for d in audio.list_devices():
            dtype = []
            if d["inputs"] > 0:
                dtype.append("🎤")
            if d["outputs"] > 0:
                dtype.append("🔊")
            print(f"  [{d['index']}] {' '.join(dtype)} {d['name']}")
        return
    
    if args.test_tts:
        tts = TTSEngine()
        await tts.speak(args.test_tts)
        return
    
    if args.test_stt:
        stt = STTEngine()
        text = await stt.transcribe(args.test_stt)
        print(f"📝 Result: {text}")
        return
    
    # Run main voice system
    voice = PibotVoice(config_path=args.config)
    await voice.run()


if __name__ == "__main__":
    asyncio.run(main())
