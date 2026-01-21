import subprocess
import os
import shutil
import time

print("="*50)
print("  Microphone & Speech Recognition Diagnostic Tool")
print("="*50)

# 1. Hardware Test using 'arecord'
print("\n[Step 1] Testing Hardware Recording (arecord)...")
filename = "/tmp/test_mic.wav"
# Delete existing file
if os.path.exists(filename):
    os.remove(filename)

print("   Recording 3 seconds... (Please speak now!)")
# Using plughw:2,0 (C920 Webcam)
cmd = "arecord -D plughw:2,0 -d 3 -f cd -t wav -r 16000 -q " + filename
exit_code = subprocess.call(cmd, shell=True)

if exit_code == 0 and os.path.exists(filename):
    size = os.path.getsize(filename)
    print(f"   [PASS] Recording finished. File size: {size} bytes")
    if size < 1000:
        print("   [WARN] File size seems too small. Mic might be muted or not working.")
else:
    print(f"   [FAIL] Recording failed. Exit code: {exit_code}")
    print("   Check if microphone is connected (arecord -l)")
    exit(1)

# 2. Software Test using 'speech_recognition'
print("\n[Step 2] Testing Speech Recognition Library...")

# Monkey Patch for FLAC
print("   Applying Monkey Patch to bypass FLAC check...")
original_which = shutil.which
def mock_which(cmd, mode=os.F_OK | os.X_OK, path=None):
    if cmd == "flac":
        return None
    return original_which(cmd, mode, path)
shutil.which = mock_which

try:
    import speech_recognition as sr
    print("   [PASS] speech_recognition imported successfully")
except ImportError:
    print("   [FAIL] Could not import speech_recognition")
    exit(1)

print("\n[Step 3] Recognizing Audio via Google API...")
r = sr.Recognizer()
try:
    with sr.AudioFile(filename) as source:
        audio = r.record(source)
    
    # Attempt recognition
    start_time = time.time()
    text = r.recognize_google(audio, language="th-TH")
    end_time = time.time()
    
    print(f"   [PASS] Recognized Text: '{text}'")
    print(f"   Time taken: {end_time - start_time:.2f}s")
except sr.UnknownValueError:
    print("   [INFO] Google could not understand audio (Result was empty/unclear)")
except sr.RequestError as e:
    print(f"   [FAIL] Google API request failed: {e}")
except Exception as e:
    print(f"   [FAIL] Unexpected error: {e}")

print("="*50)
