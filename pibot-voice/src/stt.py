#!/usr/bin/env python3
"""
Speech-to-Text module for Pibot Voice
Uses OpenAI Whisper (local or API)
"""

import asyncio
import tempfile
import subprocess
import sys
import os
from pathlib import Path
from typing import Optional

# Try to import whisper (local)
try:
    import whisper
    WHISPER_LOCAL = True
except ImportError:
    WHISPER_LOCAL = False

# Try to import openai (API)
try:
    import openai
    WHISPER_API = True
except ImportError:
    WHISPER_API = False


class STTEngine:
    """Speech-to-Text engine using Whisper"""
    
    def __init__(
        self,
        model: str = "base",
        language: str = "th",
        use_api: bool = False,
        api_key: str = None
    ):
        self.model_name = model
        self.language = language
        self.use_api = use_api
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self._model = None
        
        if use_api:
            if not WHISPER_API:
                raise ImportError("openai not installed. Run: pip install openai")
            if not self.api_key:
                raise ValueError("OpenAI API key required for API mode")
        else:
            if not WHISPER_LOCAL:
                print("⚠️ Local whisper not installed. Falling back to API mode.")
                self.use_api = True
                if not self.api_key:
                    raise ImportError(
                        "Neither local whisper nor API available.\n"
                        "Run: pip install openai-whisper  (for local)\n"
                        "Or:  pip install openai  (for API)"
                    )
    
    def _load_model(self):
        """Lazy load the whisper model"""
        if self._model is None and not self.use_api:
            print(f"🔄 Loading Whisper model: {self.model_name}")
            self._model = whisper.load_model(self.model_name)
            print("✅ Model loaded")
        return self._model
    
    async def transcribe(self, audio_path: str) -> Optional[str]:
        """Transcribe audio file to text"""
        if not Path(audio_path).exists():
            print(f"❌ Audio file not found: {audio_path}")
            return None
        
        try:
            if self.use_api:
                return await self._transcribe_api(audio_path)
            else:
                return await self._transcribe_local(audio_path)
        except Exception as e:
            print(f"❌ STT error: {e}")
            return None
    
    async def _transcribe_local(self, audio_path: str) -> Optional[str]:
        """Transcribe using local Whisper model"""
        model = self._load_model()
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: model.transcribe(
                audio_path,
                language=self.language,
                fp16=False  # Use FP32 on CPU
            )
        )
        
        return result.get("text", "").strip()
    
    async def _transcribe_api(self, audio_path: str) -> Optional[str]:
        """Transcribe using OpenAI Whisper API"""
        client = openai.AsyncOpenAI(api_key=self.api_key)
        
        with open(audio_path, "rb") as audio_file:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language=self.language
            )
        
        return response.text.strip()
    
    def transcribe_sync(self, audio_path: str) -> Optional[str]:
        """Synchronous wrapper for transcribe()"""
        return asyncio.run(self.transcribe(audio_path))


async def main():
    """CLI test"""
    if len(sys.argv) < 2:
        print("Usage: python stt.py <audio_file>")
        print("\nSupported formats: wav, mp3, m4a, webm, mp4")
        return
    
    audio_path = sys.argv[1]
    use_api = "--api" in sys.argv
    
    print(f"🎤 Transcribing: {audio_path}")
    print(f"   Mode: {'API' if use_api else 'Local'}")
    
    stt = STTEngine(use_api=use_api)
    text = await stt.transcribe(audio_path)
    
    if text:
        print(f"\n📝 Result: {text}")
    else:
        print("\n❌ Failed to transcribe")


if __name__ == "__main__":
    asyncio.run(main())
