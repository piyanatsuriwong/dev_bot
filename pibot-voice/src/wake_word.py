#!/usr/bin/env python3
"""
Wake Word Detection for Pibot Voice
Detects trigger phrase "พีบอท" to activate listening
"""

import asyncio
import time
from typing import Callable, Optional
from dataclasses import dataclass

# Try importing wake word engines
try:
    import pvporcupine
    PORCUPINE_AVAILABLE = True
except ImportError:
    PORCUPINE_AVAILABLE = False

try:
    from openwakeword.model import Model as OWWModel
    OPENWAKEWORD_AVAILABLE = True
except ImportError:
    OPENWAKEWORD_AVAILABLE = False


@dataclass 
class WakeWordConfig:
    """Wake word configuration"""
    phrase: str = "พีบอท"
    engine: str = "keyword"  # keyword, porcupine, openwakeword
    sensitivity: float = 0.5
    timeout: float = None  # None = listen forever


class WakeWordDetector:
    """
    Wake word detector with multiple engine support
    
    Engines:
    - keyword: Simple STT-based keyword matching (fallback)
    - porcupine: Picovoice Porcupine (requires API key)
    - openwakeword: OpenWakeWord (free, local)
    """
    
    def __init__(self, config: WakeWordConfig = None):
        self.config = config or WakeWordConfig()
        self._running = False
        self._stt = None
        self._audio = None
    
    async def listen(
        self,
        on_wake: Callable = None,
        on_timeout: Callable = None
    ) -> bool:
        """
        Listen for wake word
        
        Args:
            on_wake: Callback when wake word detected
            on_timeout: Callback when timeout reached
        
        Returns:
            True if wake word detected, False if timeout
        """
        engine = self.config.engine.lower()
        
        if engine == "keyword":
            return await self._listen_keyword(on_wake, on_timeout)
        elif engine == "porcupine":
            return await self._listen_porcupine(on_wake, on_timeout)
        elif engine == "openwakeword":
            return await self._listen_openwakeword(on_wake, on_timeout)
        else:
            print(f"⚠️ Unknown engine: {engine}, falling back to keyword")
            return await self._listen_keyword(on_wake, on_timeout)
    
    async def _listen_keyword(
        self,
        on_wake: Callable = None,
        on_timeout: Callable = None
    ) -> bool:
        """
        Simple keyword-based detection using STT
        Records short audio clips and checks for wake word
        """
        # Lazy import to avoid circular dependency
        from audio import AudioManager, AudioConfig
        from stt import STTEngine
        
        if self._audio is None:
            self._audio = AudioManager(AudioConfig(sample_rate=16000))
        if self._stt is None:
            self._stt = STTEngine(model="tiny", language="th", use_api=False)
        
        print(f"🎯 Listening for wake word: '{self.config.phrase}'")
        
        start_time = time.time()
        self._running = True
        
        while self._running:
            # Check timeout
            if self.config.timeout:
                elapsed = time.time() - start_time
                if elapsed >= self.config.timeout:
                    if on_timeout:
                        on_timeout()
                    return False
            
            # Record short clip
            audio_path = await self._audio.record(
                duration=2.0,  # 2 second clips
                max_duration=2.0
            )
            
            if not audio_path:
                await asyncio.sleep(0.5)
                continue
            
            # Transcribe
            text = await self._stt.transcribe(audio_path)
            
            if text:
                text_lower = text.lower().strip()
                wake_lower = self.config.phrase.lower()
                
                # Check for wake word
                if wake_lower in text_lower or self._fuzzy_match(text_lower, wake_lower):
                    print(f"✅ Wake word detected! ({text})")
                    if on_wake:
                        on_wake()
                    return True
            
            await asyncio.sleep(0.1)
        
        return False
    
    async def _listen_porcupine(
        self,
        on_wake: Callable = None,
        on_timeout: Callable = None
    ) -> bool:
        """Porcupine wake word detection (requires API key)"""
        if not PORCUPINE_AVAILABLE:
            print("⚠️ Porcupine not available, falling back to keyword")
            return await self._listen_keyword(on_wake, on_timeout)
        
        # TODO: Implement Porcupine detection
        # Requires custom wake word training for "พีบอท"
        print("⚠️ Porcupine not implemented yet, falling back to keyword")
        return await self._listen_keyword(on_wake, on_timeout)
    
    async def _listen_openwakeword(
        self,
        on_wake: Callable = None,
        on_timeout: Callable = None
    ) -> bool:
        """OpenWakeWord detection (free, local)"""
        if not OPENWAKEWORD_AVAILABLE:
            print("⚠️ OpenWakeWord not available, falling back to keyword")
            return await self._listen_keyword(on_wake, on_timeout)
        
        # TODO: Implement OpenWakeWord detection
        # Requires training custom model for "พีบอท"
        print("⚠️ OpenWakeWord not implemented yet, falling back to keyword")
        return await self._listen_keyword(on_wake, on_timeout)
    
    def _fuzzy_match(self, text: str, phrase: str) -> bool:
        """Simple fuzzy matching for Thai text"""
        # Common variations
        variations = [
            "พีบอท", "pbot", "p bot", "พี บอท",
            "ปี้บอท", "พีบ็อท", "pibot", "pi bot"
        ]
        
        for var in variations:
            if var.lower() in text:
                return True
        
        return False
    
    def stop(self):
        """Stop listening"""
        self._running = False
    
    def cleanup(self):
        """Cleanup resources"""
        self.stop()
        if self._audio:
            self._audio.cleanup()


async def main():
    """CLI test"""
    import sys
    
    print("🎯 Wake Word Detector Test")
    print("-" * 40)
    
    config = WakeWordConfig(
        phrase="พีบอท",
        engine="keyword",
        timeout=30.0  # 30 second timeout for testing
    )
    
    detector = WakeWordDetector(config)
    
    def on_wake():
        print("🔔 WAKE!")
    
    def on_timeout():
        print("⏱️ Timeout")
    
    try:
        detected = await detector.listen(
            on_wake=on_wake,
            on_timeout=on_timeout
        )
        print(f"\nResult: {'Detected' if detected else 'Not detected'}")
    except KeyboardInterrupt:
        print("\n👋 Stopped")
    finally:
        detector.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
