# IMX500 Metadata Usage Analysis

## การตรวจสอบ: โค้ดใช้แค่ Metadata หรือดึงภาพด้วย?

### ✅ ผลการตรวจสอบ

#### 1. **Picamera2 Backend** (`yolo_tracker_v2.py`)

**สถานะ: ✅ ใช้แค่ Metadata**

```python
# Line 438: _detection_loop_picamera2()
metadata = self.picam2.capture_metadata()  # ✅ Metadata only
```

- ใช้ `capture_metadata()` เท่านั้น
- **ไม่ดึงภาพ** (ประหยัด bandwidth 99.99%)
- ได้ผลลัพธ์ AI จาก IMX500 โดยตรง

---

#### 2. **modlib Backend** (`yolo_tracker_v2.py`)

**สถานะเดิม: ❌ ดึงทั้งภาพและ Metadata**

```python
# Line 718: _detection_loop_modlib()
frame_image = frame.image  # ❌ ดึงภาพด้วย
self._latest_frame = frame_image  # เก็บภาพไว้
```

**ปัญหา:**
- modlib stream ให้ทั้งภาพและ detections มา
- ดึงภาพทุก frame แม้ไม่ต้องการแสดงผล
- สิ้นเปลือง bandwidth (1.2 MB/frame)

**แก้ไขแล้ว: ✅ เพิ่ม option `capture_image`**

```python
# ใหม่: สามารถเลือกได้ว่าจะดึงภาพหรือไม่
def __init__(self, ..., capture_image: bool = True):
    self.capture_image = capture_image

# ใน detection loop
if self.capture_image:
    frame_image = frame.image  # ดึงภาพถ้าต้องการ
else:
    frame_image = None  # Metadata-only mode
```

**ผลลัพธ์:**
- ถ้า `capture_image=False`: ใช้แค่ Metadata ✅
- ถ้า `capture_image=True`: ดึงทั้งภาพและ Metadata (สำหรับแสดงผล)

---

#### 3. **main_roboeyes.py**

**สถานะ: ✅ ปรับให้ใช้ Metadata-only เมื่อไม่ต้องการแสดงผล**

```python
# Line 316-324: init_yolo_tracker()
capture_image = self.show_hdmi  # ดึงภาพเฉพาะเมื่อแสดง HDMI

self.yolo_tracker = create_yolo_tracker(
    confidence_threshold=self.yolo_confidence,
    frame_rate=getattr(config, 'YOLO_FRAME_RATE', 5),
    capture_image=capture_image  # ✅ Metadata-only if no HDMI
)
```

**ผลลัพธ์:**
- ถ้า `--no-hdmi`: ใช้ Metadata-only ✅ (ประหยัด bandwidth)
- ถ้าแสดง HDMI: ดึงภาพเพื่อแสดงผล (จำเป็น)

---

## สรุปการใช้งาน Metadata

### ✅ Metadata-Only Mode (ประหยัด Bandwidth)

**เมื่อใช้:**
- ไม่ต้องการแสดงภาพบน HDMI (`--no-hdmi`)
- Headless mode (IoT, Edge Computing)
- Real-time detection โดยไม่ต้องบันทึกภาพ

**ข้อดี:**
- ประหยัด bandwidth 99.99% (88 bytes vs 1.2 MB)
- ไม่ใช้ CPU ในการประมวลผลภาพ
- เร็วและมีประสิทธิภาพ

**การตั้งค่า:**
```python
# Option 1: ผ่าน create_yolo_tracker()
tracker = create_yolo_tracker(capture_image=False)

# Option 2: ผ่าน config.py
YOLO_CAPTURE_IMAGE = False

# Option 3: ผ่าน main_roboeyes.py
python3 main_roboeyes.py --no-hdmi  # Auto metadata-only
```

---

### 📊 Image + Metadata Mode (สำหรับแสดงผล)

**เมื่อใช้:**
- ต้องการแสดงภาพบน HDMI
- ต้องการวาด bounding boxes บนภาพ
- Debugging หรือ development

**การตั้งค่า:**
```python
# Default: capture_image=True
tracker = create_yolo_tracker(capture_image=True)

# หรือแสดง HDMI
python3 main_roboeyes.py  # Auto capture image
```

---

## Backend Comparison

| Backend | Metadata Only | Image + Metadata | หมายเหตุ |
|---------|--------------|------------------|----------|
| **Picamera2** | ✅ `capture_metadata()` | ❌ ไม่รองรับ | ใช้ Metadata เท่านั้น |
| **modlib** | ✅ `capture_image=False` | ✅ `capture_image=True` | รองรับทั้งสองแบบ |

---

## การปรับปรุงที่ทำ

### 1. เพิ่ม `capture_image` Parameter

```python
# yolo_tracker_v2.py
def __init__(self, ..., capture_image: bool = True):
    self.capture_image = capture_image
```

### 2. ปรับ modlib Detection Loop

```python
# ดึงภาพเฉพาะเมื่อต้องการ
if self.capture_image:
    frame_image = frame.image
else:
    frame_image = None  # Metadata-only
```

### 3. Auto-detect ใน main_roboeyes.py

```python
# ดึงภาพเฉพาะเมื่อแสดง HDMI
capture_image = self.show_hdmi
```

---

## ตัวอย่างการใช้งาน

### Metadata-Only (Headless)

```python
# ไม่ต้องการแสดงภาพ
tracker = create_yolo_tracker(
    confidence_threshold=0.5,
    capture_image=False  # ✅ Metadata-only
)

tracker.start()

while True:
    detections = tracker.detections
    for det in detections:
        print(f"{det['label']}: {det['confidence']:.2f}")
    
    time.sleep(0.1)
```

### Image + Metadata (Display)

```python
# ต้องการแสดงภาพ
tracker = create_yolo_tracker(
    confidence_threshold=0.5,
    capture_image=True  # ✅ ดึงภาพด้วย
)

tracker.start()

while True:
    frame = tracker.latest_frame
    detections = tracker.detections
    
    # วาด bounding boxes
    for det in detections:
        x, y, w, h = det['bbox']
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imshow('Detection', frame)
    cv2.waitKey(1)
```

---

## สรุป

### ✅ ยืนยัน: โค้ดสามารถใช้แค่ Metadata ได้!

1. **Picamera2 backend**: ใช้ Metadata เท่านั้น ✅
2. **modlib backend**: รองรับทั้ง Metadata-only และ Image+Metadata ✅
3. **main_roboeyes.py**: Auto-detect ตาม `show_hdmi` ✅

### 💡 ข้อแนะนำ

- **Headless/IoT**: ใช้ `capture_image=False` เพื่อประหยัด bandwidth
- **Development/Debug**: ใช้ `capture_image=True` เพื่อดูภาพ
- **Production**: ใช้ `--no-hdmi` เพื่อ auto metadata-only mode

---

## References

- `yolo_tracker_v2.py` - YOLO Tracker implementation
- `main_roboeyes.py` - Main application
- `IMX500_METADATA_TEST_RESULTS.md` - Test results showing metadata size
