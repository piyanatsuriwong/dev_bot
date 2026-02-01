#!/usr/bin/env python3
"""
Audio I/O Manager for Pibot Voice
Handles microphone input and speaker output
"""

import asyncio
import wave
import tempfile
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Callable
from dataclasses import dataclass

try:
    import pyaudio
    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False


@dataclass
class AudioConfig:
    """Audio configuration"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: int = 8  # paInt16 = 8
    input_device: Optional[int] = None
    output_device: Optional[int] = None


class AudioManager:
    """Manages audio input/output devices"""
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig()
        self._pyaudio = None
        self._check_availability()
    
    def _check_availability(self):
        """Check if audio is available"""
        if not PYAUDIO_AVAILABLE:
            print("⚠️ PyAudio not installed. Run: pip install pyaudio")
            print("   On Debian/Ubuntu: sudo apt install python3-pyaudio")
            return False
        return True
    
    def _get_pyaudio(self):
        """Lazy init PyAudio"""
        if self._pyaudio is None and PYAUDIO_AVAILABLE:
            self._pyaudio = pyaudio.PyAudio()
        return self._pyaudio
    
    def list_devices(self) -> list:
        """List available audio devices"""
        if not PYAUDIO_AVAILABLE:
            return []
        
        pa = self._get_pyaudio()
        devices = []
        
        for i in range(pa.get_device_count()):
            info = pa.get_device_info_by_index(i)
            devices.append({
                "index": i,
                "name": info["name"],
                "inputs": info["maxInputChannels"],
                "outputs": info["maxOutputChannels"],
                "default_rate": info["defaultSampleRate"]
            })
        
        return devices
    
    def has_microphone(self) -> bool:
        """Check if any microphone is available"""
        devices = self.list_devices()
        return any(d["inputs"] > 0 for d in devices)
    
    def has_speaker(self) -> bool:
        """Check if any speaker is available"""
        devices = self.list_devices()
        return any(d["outputs"] > 0 for d in devices)
    
    async def record(
        self,
        duration: float = None,
        silence_threshold: float = 1.5,
        max_duration: float = 30.0,
        output_path: str = None,
        on_start: Callable = None,
        on_sound: Callable = None
    ) -> Optional[str]:
        """
        Record audio from microphone
        
        Args:
            duration: Fixed duration in seconds (None = auto-stop on silence)
            silence_threshold: Seconds of silence before stopping
            max_duration: Maximum recording duration
            output_path: Output file path (None = temp file)
            on_start: Callback when recording starts
            on_sound: Callback when sound is detected
        
        Returns:
            Path to recorded WAV file or None on error
        """
        if not PYAUDIO_AVAILABLE:
            print("❌ PyAudio not available")
            return None
        
        if not self.has_microphone():
            print("❌ No microphone found")
            return None
        
        if not output_path:
            output_path = tempfile.mktemp(suffix=".wav")
        
        pa = self._get_pyaudio()
        
        try:
            stream = pa.open(
                format=self.config.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=self.config.input_device,
                frames_per_buffer=self.config.chunk_size
            )
            
            if on_start:
                on_start()
            
            print("🎤 Recording...")
            frames = []
            start_time = time.time()
            last_sound_time = start_time
            sound_detected = False
            
            while True:
                elapsed = time.time() - start_time
                
                # Check max duration
                if elapsed >= max_duration:
                    print("⏱️ Max duration reached")
                    break
                
                # Fixed duration mode
                if duration and elapsed >= duration:
                    break
                
                # Read audio chunk
                data = stream.read(self.config.chunk_size, exception_on_overflow=False)
                frames.append(data)
                
                # Simple volume detection (RMS)
                rms = self._calculate_rms(data)
                is_sound = rms > 500  # Threshold
                
                if is_sound:
                    if not sound_detected and on_sound:
                        on_sound()
                    sound_detected = True
                    last_sound_time = time.time()
                
                # Auto-stop on silence (only if sound was detected)
                if duration is None and sound_detected:
                    silence_duration = time.time() - last_sound_time
                    if silence_duration >= silence_threshold:
                        print("🔇 Silence detected, stopping")
                        break
                
                await asyncio.sleep(0.01)
            
            stream.stop_stream()
            stream.close()
            
            # Save to WAV file
            self._save_wav(output_path, frames)
            print(f"✅ Saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Recording error: {e}")
            return None
    
    def _calculate_rms(self, data: bytes) -> float:
        """Calculate RMS volume of audio data"""
        import struct
        count = len(data) // 2
        shorts = struct.unpack(f"{count}h", data)
        sum_squares = sum(s * s for s in shorts)
        return (sum_squares / count) ** 0.5
    
    def _save_wav(self, path: str, frames: list):
        """Save frames to WAV file"""
        wf = wave.open(path, "wb")
        wf.setnchannels(self.config.channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(self.config.sample_rate)
        wf.writeframes(b"".join(frames))
        wf.close()
    
    async def play(self, audio_path: str) -> bool:
        """Play audio file through speaker"""
        if not Path(audio_path).exists():
            print(f"❌ File not found: {audio_path}")
            return False
        
        # Try different players
        players = [
            ["mpv", "--no-video", "--really-quiet", audio_path],
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path],
            ["aplay", "-q", audio_path],
            ["paplay", audio_path],
        ]
        
        for cmd in players:
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL
                )
                await process.wait()
                if process.returncode == 0:
                    return True
            except FileNotFoundError:
                continue
        
        print(f"⚠️ No audio player found")
        return False
    
    def cleanup(self):
        """Cleanup resources"""
        if self._pyaudio:
            self._pyaudio.terminate()
            self._pyaudio = None


async def main():
    """CLI test"""
    manager = AudioManager()
    
    print("🔊 Audio Devices:")
    print("-" * 40)
    
    devices = manager.list_devices()
    if not devices:
        print("  No devices found (PyAudio not installed?)")
    else:
        for d in devices:
            dtype = []
            if d["inputs"] > 0:
                dtype.append("🎤 IN")
            if d["outputs"] > 0:
                dtype.append("🔊 OUT")
            print(f"  [{d['index']}] {d['name']} ({', '.join(dtype)})")
    
    print()
    print(f"🎤 Microphone: {'✅' if manager.has_microphone() else '❌'}")
    print(f"🔊 Speaker: {'✅' if manager.has_speaker() else '❌'}")
    
    if "--record" in sys.argv and manager.has_microphone():
        print("\n🎤 Recording for 5 seconds...")
        path = await manager.record(duration=5)
        if path:
            print(f"   Saved to: {path}")
    
    manager.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
