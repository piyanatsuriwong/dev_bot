#!/usr/bin/env python3
"""
Face Recognition System - ระบบทำความรู้จักและจดจำคน
ใช้ InsightFace (ONNX-based) สำหรับ Detection + Recognition
Modern approach: เร็วกว่า dlib 5-7x, CPU 20-30%
"""

import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
import time

# Local imports
from database import (
    init_database, add_person, get_person, get_person_by_name,
    list_people, add_encoding, get_all_encodings, log_sighting,
    get_recent_sightings, delete_person
)

# Paths
BASE_DIR = Path(__file__).parent
FACES_DIR = BASE_DIR / "captured_faces"
FACES_DIR.mkdir(exist_ok=True)

class FaceRecognitionSystem:
    """
    Modern Face Recognition System using InsightFace (ONNX)
    - Detection: RetinaFace / SCRFD
    - Recognition: ArcFace (512-dim embeddings)
    - Performance: 10-15 FPS on RPi 5
    """
    
    def __init__(self, camera_id: int = 0, threshold: float = 0.5):
        """
        Args:
            camera_id: Pi Camera ID (0 for IMX500, 1 for IMX708)
            threshold: Cosine similarity threshold (0.4-0.6 recommended)
        """
        self.camera_id = camera_id
        self.threshold = threshold
        self.picam = None
        self.face_analyzer = None
        self._encodings_cache = None
        self._cache_dirty = True
        
        # Initialize database
        init_database()
        
        # Initialize InsightFace
        self._init_insightface()
    
    def _init_insightface(self):
        """Initialize InsightFace analyzer"""
        try:
            from insightface.app import FaceAnalysis
            
            # Use buffalo_l for best accuracy, buffalo_s for speed
            self.face_analyzer = FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 480))
            print("✅ InsightFace initialized (buffalo_l model)")
            print("   Detection: SCRFD | Recognition: ArcFace")
        except Exception as e:
            print(f"❌ InsightFace init failed: {e}")
            self.face_analyzer = None
    
    def start_camera(self):
        """Start Pi Camera"""
        if self.picam is None:
            try:
                from picamera2 import Picamera2
                self.picam = Picamera2(self.camera_id)
                config = self.picam.create_preview_configuration(
                    main={"size": (640, 480), "format": "RGB888"},
                    controls={"FrameRate": 30}
                )
                self.picam.configure(config)
                self.picam.start()
                time.sleep(1)  # Warm up
                print(f"✅ Camera {self.camera_id} started")
            except Exception as e:
                print(f"❌ Camera error: {e}")
    
    def stop_camera(self):
        """Stop camera"""
        if self.picam:
            self.picam.stop()
            self.picam = None
    
    def capture_frame(self) -> Optional[np.ndarray]:
        """Capture a single frame (RGB format)"""
        if self.picam is None:
            self.start_camera()
        
        if self.picam:
            return self.picam.capture_array()
        return None
    
    def _get_encodings_cache(self) -> List[Tuple[int, str, np.ndarray]]:
        """Get cached encodings from database"""
        if self._cache_dirty or self._encodings_cache is None:
            self._encodings_cache = get_all_encodings()
            self._cache_dirty = False
        return self._encodings_cache
    
    def _invalidate_cache(self):
        """Mark cache as dirty"""
        self._cache_dirty = True
    
    def detect_faces(self, image: np.ndarray) -> list:
        """
        Detect faces in image using InsightFace
        
        Returns:
            List of face objects with bbox, embedding, landmarks
        """
        if self.face_analyzer is None:
            return []
        
        # InsightFace expects RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Check if BGR (from OpenCV)
            pass  # InsightFace handles both
        
        faces = self.face_analyzer.get(image)
        return faces
    
    def learn_face(self, name: str, image: np.ndarray = None, notes: str = "") -> dict:
        """
        Learn a new face
        
        Args:
            name: Person's name
            image: RGB image (optional, will capture if not provided)
            notes: Optional notes
        
        Returns:
            dict with success status and info
        """
        if self.face_analyzer is None:
            return {"success": False, "error": "InsightFace not initialized"}
        
        # Capture if no image provided
        if image is None:
            image = self.capture_frame()
            if image is None:
                return {"success": False, "error": "Failed to capture image"}
        
        # Detect faces
        faces = self.detect_faces(image)
        
        if len(faces) == 0:
            return {"success": False, "error": "ไม่พบใบหน้าในภาพ"}
        
        if len(faces) > 1:
            return {"success": False, "error": f"พบ {len(faces)} ใบหน้า กรุณาใช้ภาพที่มีคนเดียว"}
        
        face = faces[0]
        embedding = face.embedding  # 512-dim vector
        bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
        
        # Check if person exists
        person = get_person_by_name(name)
        if person:
            person_id = person['id']
        else:
            person_id = add_person(name, notes)
        
        # Save face image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        face_filename = f"{name.lower().replace(' ', '_')}_{timestamp}.jpg"
        face_path = FACES_DIR / face_filename
        
        # Crop face with margin
        h, w = image.shape[:2]
        margin = 30
        x1 = max(0, bbox[0] - margin)
        y1 = max(0, bbox[1] - margin)
        x2 = min(w, bbox[2] + margin)
        y2 = min(h, bbox[3] + margin)
        face_crop = image[y1:y2, x1:x2]
        
        # Convert RGB to BGR for saving
        cv2.imwrite(str(face_path), cv2.cvtColor(face_crop, cv2.COLOR_RGB2BGR))
        
        # Add encoding to database (normalize embedding)
        embedding_norm = embedding / np.linalg.norm(embedding)
        encoding_id = add_encoding(person_id, embedding_norm, str(face_path), float(face.det_score))
        self._invalidate_cache()
        
        return {
            "success": True,
            "person_id": person_id,
            "encoding_id": encoding_id,
            "name": name,
            "face_image": str(face_path),
            "confidence": float(face.det_score),
            "message": f"✅ เรียนรู้ใบหน้า '{name}' สำเร็จ!"
        }
    
    def recognize(self, image: np.ndarray = None) -> List[dict]:
        """
        Recognize faces in image
        
        Returns:
            List of recognized faces with person_id, name, confidence
        """
        if self.face_analyzer is None:
            return []
        
        # Capture if no image provided
        if image is None:
            image = self.capture_frame()
            if image is None:
                return []
        
        # Detect faces
        faces = self.detect_faces(image)
        if not faces:
            return []
        
        # Get known encodings
        known = self._get_encodings_cache()
        if not known:
            # No known faces
            return [
                {
                    "person_id": None,
                    "name": "unknown",
                    "confidence": 0.0,
                    "bbox": face.bbox.astype(int).tolist(),
                    "det_score": float(face.det_score)
                }
                for face in faces
            ]
        
        known_ids, known_names, known_encodings = zip(*known)
        known_encodings = np.array(known_encodings)
        
        results = []
        for face in faces:
            embedding = face.embedding
            embedding_norm = embedding / np.linalg.norm(embedding)
            
            # Cosine similarity
            similarities = np.dot(known_encodings, embedding_norm)
            best_idx = np.argmax(similarities)
            best_sim = similarities[best_idx]
            
            if best_sim >= self.threshold:
                person_id = known_ids[best_idx]
                name = known_names[best_idx]
                
                # Log sighting
                log_sighting(person_id, float(best_sim))
                
                results.append({
                    "person_id": person_id,
                    "name": name,
                    "confidence": round(float(best_sim), 3),
                    "bbox": face.bbox.astype(int).tolist(),
                    "det_score": float(face.det_score)
                })
            else:
                results.append({
                    "person_id": None,
                    "name": "unknown",
                    "confidence": round(float(best_sim), 3),
                    "bbox": face.bbox.astype(int).tolist(),
                    "det_score": float(face.det_score)
                })
        
        return results
    
    def recognize_continuous(self, callback=None, show_display: bool = False):
        """
        Continuous recognition loop
        
        Args:
            callback: Function to call with results (optional)
            show_display: Show OpenCV window
        """
        print("🎥 Starting continuous recognition (Press Ctrl+C to stop)")
        
        self.start_camera()
        fps_counter = 0
        fps_start = time.time()
        
        try:
            while True:
                frame = self.capture_frame()
                if frame is None:
                    continue
                
                results = self.recognize(frame)
                
                # FPS calculation
                fps_counter += 1
                if fps_counter % 30 == 0:
                    fps = 30 / (time.time() - fps_start)
                    print(f"FPS: {fps:.1f}")
                    fps_start = time.time()
                
                # Callback
                if callback and results:
                    callback(results)
                
                # Display
                if show_display:
                    display = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    for r in results:
                        bbox = r['bbox']
                        color = (0, 255, 0) if r['person_id'] else (0, 0, 255)
                        cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                        label = f"{r['name']} ({r['confidence']:.0%})"
                        cv2.putText(display, label, (bbox[0], bbox[1]-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    cv2.imshow("Face Recognition", display)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                time.sleep(0.03)  # ~30 FPS
                
        except KeyboardInterrupt:
            print("\n🛑 Stopping...")
        finally:
            self.stop_camera()
            if show_display:
                cv2.destroyAllWindows()
    
    def list_known_people(self) -> List[dict]:
        """List all known people"""
        return list_people()
    
    def get_recent_sightings(self, limit: int = 20) -> List[dict]:
        """Get recent sightings"""
        return get_recent_sightings(limit)
    
    def forget_person(self, name: str) -> bool:
        """Remove a person"""
        person = get_person_by_name(name)
        if person:
            result = delete_person(person['id'])
            self._invalidate_cache()
            return result
        return False


def main():
    """CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Face Recognition System (InsightFace)")
    parser.add_argument("action", choices=["learn", "recognize", "live", "list", "sightings", "test"])
    parser.add_argument("--name", "-n", help="Person name")
    parser.add_argument("--image", "-i", help="Image file path")
    parser.add_argument("--camera", "-c", type=int, default=0, help="Camera ID")
    parser.add_argument("--display", "-d", action="store_true", help="Show display")
    parser.add_argument("--notes", default="", help="Notes")
    
    args = parser.parse_args()
    
    system = FaceRecognitionSystem(camera_id=args.camera)
    
    if args.action == "list":
        people = system.list_known_people()
        if people:
            print("\n👥 คนที่รู้จัก:")
            for p in people:
                print(f"  • {p['name']} (ID: {p['id']}) - {p['encoding_count']} encodings")
        else:
            print("📭 ยังไม่รู้จักใครเลย")
    
    elif args.action == "sightings":
        sightings = system.get_recent_sightings()
        if sightings:
            print("\n👁️ การพบเห็นล่าสุด:")
            for s in sightings:
                name = s['name'] or 'unknown'
                conf = f"{s['confidence']*100:.0f}%" if s['confidence'] else 'N/A'
                print(f"  • {name} ({conf}) - {s['seen_at']}")
        else:
            print("📭 ยังไม่มีบันทึก")
    
    elif args.action == "learn":
        if not args.name:
            print("❌ กรุณาระบุชื่อด้วย --name")
            return
        
        image = None
        if args.image:
            image = cv2.imread(args.image)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                print(f"❌ ไม่สามารถโหลดรูป {args.image}")
                return
        
        print(f"📸 กำลังเรียนรู้ใบหน้า '{args.name}'...")
        result = system.learn_face(args.name, image, args.notes)
        print(result.get("message") or result.get("error"))
        if result.get("face_image"):
            print(f"   รูป: {result['face_image']}")
    
    elif args.action == "recognize":
        image = None
        if args.image:
            image = cv2.imread(args.image)
            if image is not None:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print("🔍 กำลังระบุใบหน้า...")
        results = system.recognize(image)
        
        if results:
            print(f"\n👤 พบ {len(results)} ใบหน้า:")
            for r in results:
                if r['person_id']:
                    print(f"  ✅ {r['name']} (confidence: {r['confidence']*100:.0f}%)")
                else:
                    print(f"  ❓ ไม่รู้จัก (similarity: {r['confidence']*100:.0f}%)")
        else:
            print("😶 ไม่พบใบหน้า")
    
    elif args.action == "live":
        system.recognize_continuous(show_display=args.display)
    
    elif args.action == "test":
        print("🧪 ทดสอบระบบ...")
        print(f"   Camera: {args.camera}")
        print(f"   InsightFace: {'✅' if system.face_analyzer else '❌'}")
        
        system.start_camera()
        time.sleep(1)
        
        frame = system.capture_frame()
        if frame is not None:
            print(f"   Capture: ✅ ({frame.shape[1]}x{frame.shape[0]})")
            
            faces = system.detect_faces(frame)
            print(f"   Detection: {len(faces)} face(s)")
            
            for i, face in enumerate(faces):
                print(f"     Face {i+1}: score={face.det_score:.2f}, embedding={face.embedding.shape}")
        else:
            print("   Capture: ❌")
        
        system.stop_camera()
        print("\n✅ ทดสอบเสร็จสิ้น")


if __name__ == "__main__":
    main()
