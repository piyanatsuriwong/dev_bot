# Troubleshooting: IMX500 Camera "Device or resource busy" Error

## ปัญหาที่พบ

เมื่อสลับโหมดระหว่าง **TRACK (Hand)** และ **DETECT (YOLO)** เกิด error:
```
IMX500: Unable to set network firmware /home/pi/.modlib/zoo/imx500_network_yolov8n_pp.rpk: [Errno 16] Device or resource busy
```

### สาเหตุ

1. **Device ยังถูก lock อยู่** - แม้ว่าจะ cleanup แล้ว แต่ device ยังไม่ได้ release resources จริงๆ
2. **Process อื่นใช้กล้องอยู่** - มี zombie processes หรือ process อื่นที่ lock กล้อง
3. **Reference cycles** - Python objects มี circular references ทำให้ GC ไม่สามารถ cleanup ได้ทันที
4. **Timing issue** - Device ต้องการเวลามากขึ้นในการ release resources

---

## วิธีแก้ไขที่ทำ

### 1. เพิ่ม Garbage Collection (GC)

**ไฟล์:** `yolo_tracker.py`, `main_roboeyes.py`

```python
import gc

# บังคับล้าง memory และ break reference cycles
collected = gc.collect()
if collected > 0:
    print(f"GC collected {collected} objects")
```

**ผลลัพธ์:**
- บังคับล้าง memory ทันที
- Break reference cycles
- Finalize objects ที่รอ cleanup

---

### 2. ปรับปรุงการ Cleanup Device

**ไฟล์:** `yolo_tracker.py`

**การปรับปรุง:**
- ใช้ context manager pattern (`__exit__()`) สำหรับ cleanup
- เพิ่ม delay 2 วินาทีหลังจาก cleanup device
- Force garbage collection หลัง cleanup
- Clear references เพื่อ break cycles

```python
def _cleanup_device(self):
    """Cleanup AI Camera device resources"""
    if self.device is not None:
        # Step 1: Explicit cleanup (context manager)
        if hasattr(device_ref, '__exit__'):
            device_ref.__exit__(None, None, None)
        
        # Step 2: Clear reference
        self.device = None
        
        # Step 3: Force GC
        collected = gc.collect()
        
        # Step 4: Wait for OS to release
        time.sleep(1.0)
```

---

### 3. เพิ่ม Retry Mechanism

**ไฟล์:** `yolo_tracker.py`

**การปรับปรุง:**
- ลอง initialize camera 0 ถึง 3 ครั้ง
- รอ 2 วินาทีระหว่าง retry
- Cleanup และ GC collect ระหว่าง retry

```python
def _init_camera(self):
    max_retries = 3
    retry_delay = 2.0
    
    for attempt in range(max_retries):
        try:
            self.device = AiCamera(frame_rate=self.frame_rate, num=0)
            break
        except Exception as e:
            if attempt < max_retries - 1:
                # Cleanup and retry
                gc.collect()
                time.sleep(retry_delay)
```

---

### 4. เพิ่มฟังก์ชัน Force Clear Camera

**ไฟล์:** `main_roboeyes.py`

**ฟังก์ชัน:** `force_clear_camera(device_path)`

**หน้าที่:**
1. ตรวจสอบ process ที่ใช้กล้องด้วย `fuser`
2. Kill zombie processes ที่ block กล้อง (ไม่ใช่ process ตัวเอง)
3. ถ้าเป็นตัวเอง -> ทำ GC collect
4. Force release device resources

```python
def force_clear_camera(device_path="/dev/video0"):
    """
    ตรวจสอบว่ามีใครแย่งใช้กล้องอยู่ไหม
    - ถ้าเป็นคนอื่น -> สั่ง Kill ทันที
    - ถ้าเป็นตัวเอง -> สั่ง Garbage Collection เพื่อบังคับคืนค่า
    """
    # 1. ใช้ fuser หา process ที่ใช้กล้อง
    # 2. Kill process อื่น (ไม่ใช่ตัวเอง)
    # 3. Force GC collect
```

**การใช้งาน:**
```python
# เมื่อสลับโหมด (Key 'D')
force_clear_camera("/dev/video0")
force_clear_camera("/dev/video4")
time.sleep(2.0)
```

---

### 5. ปรับปรุงการสลับโหมด

**ไฟล์:** `main_roboeyes.py`

**ขั้นตอนการสลับโหมด (Key 'D'):**

1. **Cleanup tracker เก่า**
   ```python
   if self.yolo_tracker is not None:
       self.yolo_tracker.cleanup()
       self.yolo_tracker = None
   ```

2. **Force clear camera**
   ```python
   force_clear_camera("/dev/video0")
   force_clear_camera("/dev/video4")
   ```

3. **รอ device release**
   ```python
   time.sleep(2.0)
   ```

4. **Initialize tracker ใหม่**
   ```python
   self.yolo_tracker = create_yolo_tracker(...)
   ```

---

### 6. เพิ่ม Weakref Finalizer

**ไฟล์:** `yolo_tracker.py`

**การปรับปรุง:**
- เพิ่ม `weakref.finalize()` เป็น fallback cleanup
- ทำงานเมื่อ object ถูก garbage collected
- เป็น safety net เมื่อลืมเรียก `cleanup()`

```python
import weakref

def __init__(self, ...):
    # Register finalizer as fallback
    self._finalizer = weakref.finalize(
        self, 
        self._finalize_cleanup, 
        weakref.ref(self)
    )
```

---

## Best Practices ที่ใช้

### 1. Context Manager Pattern
- ใช้ `__exit__()` สำหรับ deterministic cleanup
- ทำงานได้ดีกว่า `__del__()` ที่ไม่ guarantee

### 2. Explicit Resource Cleanup
- Cleanup ก่อน garbage collection
- ไม่พึ่งพา GC เพียงอย่างเดียว

### 3. Force Garbage Collection
- ใช้ `gc.collect()` เมื่อจำเป็น
- Break reference cycles
- Finalize objects

### 4. Delay สำหรับ Hardware
- รอให้ OS release device resources
- IMX500 ต้องการเวลาในการ release

### 5. Retry Mechanism
- ลองหลายครั้งก่อนล้มเหลว
- Cleanup ระหว่าง retry

---

## การตรวจสอบปัญหา

### 1. ตรวจสอบ Process ที่ใช้กล้อง

```bash
# ดู process ที่ใช้ video devices
fuser /dev/video0
lsof | grep video

# ดู process Python ที่รันอยู่
ps aux | grep python | grep -E 'main_roboeyes|yolo'
```

### 2. Kill Zombie Processes

```bash
# Kill process ที่ block กล้อง
pkill -f main_roboeyes
pkill -f yolo

# หรือ kill process เฉพาะ
kill -9 <PID>
```

### 3. ตรวจสอบ Camera State

```bash
# List cameras
rpicam-hello --list-cameras

# Check video devices
ls -la /dev/video*
```

---

## สรุปการแก้ไข

| ปัญหา | วิธีแก้ไข | ไฟล์ |
|-------|----------|------|
| Device busy | Force clear camera + retry | `main_roboeyes.py` |
| Memory leaks | GC collect + weakref finalizer | `yolo_tracker.py` |
| Reference cycles | Clear references + GC | `yolo_tracker.py` |
| Timing issues | เพิ่ม delay + retry | `yolo_tracker.py` |
| Zombie processes | Kill process อื่น | `main_roboeyes.py` |

---

## คำแนะนำ

1. **เมื่อสลับโหมด:**
   - ระบบจะ cleanup และ force clear อัตโนมัติ
   - รอ 2-3 วินาทีระหว่างสลับโหมด

2. **ถ้ายังมีปัญหา:**
   - Kill process เก่าก่อนรันใหม่
   - ตรวจสอบว่าไม่มี process อื่นใช้กล้อง
   - เพิ่ม delay ถ้าจำเป็น

3. **Monitoring:**
   - ดู log messages เพื่อ debug
   - ตรวจสอบ GC collected count
   - ตรวจสอบ process ที่ถูก kill

---

## ไฟล์ที่แก้ไข

1. `yolo_tracker.py`
   - เพิ่ม `import gc`, `import weakref`
   - ปรับปรุง `_cleanup_device()`
   - เพิ่ม retry mechanism ใน `_init_camera()`
   - เพิ่ม `weakref.finalize()` fallback

2. `main_roboeyes.py`
   - เพิ่ม `import signal`
   - สร้างฟังก์ชัน `force_clear_camera()`
   - ปรับปรุงการสลับโหมด (Key 'D' และ 'T')
   - เพิ่ม `gc.collect()` ในส่วนที่จำเป็น

---

## References

- [Python GC Documentation](https://docs.python.org/3/library/gc.html)
- [Context Managers](https://docs.python.org/3/library/stdtypes.html#context-manager-types)
- [Weakref Finalizers](https://docs.python.org/3/library/weakref.html#weakref.finalize)

---

**Last Updated:** 2025-01-22
**Version:** 1.0
