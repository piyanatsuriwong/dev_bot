#!/usr/bin/env python3
"""
Text-to-Speech module for Pibot Voice
Uses Edge TTS (free, good Thai support)
"""

import asyncio
import tempfile
import subprocess
import sys
from pathlib import Path

try:
    import edge_tts
except ImportError:
    edge_tts = None


class TTSEngine:
    """Text-to-Speech engine using Edge TTS"""
    
    def __init__(self, voice: str = "th-TH-PremwadeeNeural", rate: str = "+0%", volume: str = "+0%"):
        self.voice = voice
        self.rate = rate
        self.volume = volume
        
        if edge_tts is None:
            raise ImportError("edge-tts not installed. Run: pip install edge-tts")
    
    async def synthesize(self, text: str, output_path: str = None) -> str:
        """Convert text to speech, return path to audio file"""
        if not output_path:
            output_path = tempfile.mktemp(suffix=".mp3")
        
        communicate = edge_tts.Communicate(
            text=text,
            voice=self.voice,
            rate=self.rate,
            volume=self.volume
        )
        
        await communicate.save(output_path)
        return output_path
    
    async def speak(self, text: str) -> bool:
        """Synthesize and play audio"""
        try:
            audio_path = await self.synthesize(text)
            
            # Try different players
            players = [
                ["mpv", "--no-video", audio_path],
                ["ffplay", "-nodisp", "-autoexit", audio_path],
                ["aplay", audio_path],
                ["paplay", audio_path],
            ]
            
            for player_cmd in players:
                try:
                    result = subprocess.run(
                        player_cmd,
                        capture_output=True,
                        timeout=60
                    )
                    if result.returncode == 0:
                        return True
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    continue
            
            print(f"⚠️ No audio player found. Audio saved to: {audio_path}")
            return False
            
        except Exception as e:
            print(f"❌ TTS error: {e}")
            return False
    
    def speak_sync(self, text: str) -> bool:
        """Synchronous wrapper for speak()"""
        return asyncio.run(self.speak(text))
    
    @staticmethod
    async def list_voices(language: str = "th") -> list:
        """List available voices for a language"""
        voices = await edge_tts.list_voices()
        return [v for v in voices if v["Locale"].startswith(language)]


async def main():
    """CLI test"""
    if len(sys.argv) < 2:
        print("Usage: python tts.py <text>")
        print("\nAvailable Thai voices:")
        voices = await TTSEngine.list_voices("th")
        for v in voices:
            print(f"  - {v['ShortName']}: {v['Gender']}")
        return
    
    text = " ".join(sys.argv[1:])
    print(f"🔊 Speaking: {text}")
    
    tts = TTSEngine()
    success = await tts.speak(text)
    
    if success:
        print("✅ Done")
    else:
        print("❌ Failed to play audio")


if __name__ == "__main__":
    asyncio.run(main())
