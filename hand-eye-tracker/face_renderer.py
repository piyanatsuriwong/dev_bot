import pygame
import config
import random
import time
import math
import json
import os

# --- YARDIMCI FONKSİYONLAR ---
def lerp_smooth(start, end, amount):
    """Yumuşak geçiş matematiği"""
    t = amount * (2 - amount)
    return start + (end - start) * t

def catmull_rom_spline(p0, p1, p2, p3, t):
    """5 Noktalı kaşı pürüzsüz eğriye çeviren fonksiyon"""
    return 0.5 * ((2 * p1) + (-p0 + p2) * t + 
                  (2 * p0 - 5 * p1 + 4 * p2 - p3) * t**2 + 
                  (-p0 + 3 * p1 - 3 * p2 + p3) * t**3)

class FaceRenderer:
    def __init__(self, screen):
        self.screen = screen
        self.center_x = config.SCREEN_WIDTH // 2
        self.center_y = config.SCREEN_HEIGHT // 2
        
        # --- ESKİ DEĞİŞKENLER (Uyumluluk İçin) ---
        self.current_tilt = 0; self.target_tilt = 0
        self.current_lid_y = -60; self.target_lid_y = -60
        self.current_bend = 0; self.target_bend = 0
        self.current_tip_thick = 8; self.target_tip_thick = 8
        self.current_center_thick = 15; self.target_center_thick = 15
        self.current_happy_squint = 0.0; self.target_happy_squint = 0.0
        self.current_glow_intensity = 1.0; self.target_glow_intensity = 1.0
        
        # --- YENİ: V16+ SİSTEM DEĞİŞKENLERİ ---
        # 5 Segmentli Kaş Yapısı
        self.default_segments = [{"off_y": 0, "thick": 8} for _ in range(5)]
        self.current_segments = [s.copy() for s in self.default_segments]
        self.target_segments = [s.copy() for s in self.default_segments]
        
        # Kaş Genişliği
        self.current_brow_width = 120
        self.target_brow_width = 120
        
        # Ekstra Çizimler (Kalem/Kesme)
        self.current_extras = []
        
        # Göz Görünürlüğü
        self.eye_visible = True
        self.is_hollow = False
        
        # --- EFEKT & FİZİK ---
        self.shake_intensity = 1.0 
        self.glow_enabled = True    
        self.smooth_speed = 0.2
        
        # Bakış
        self.look_x = 0; self.look_y = 0
        self.target_look_x = 0; self.target_look_y = 0
        
        # AI / Davranış
        self.is_aggro = False; self.is_tracking = False
        self.neutral_brow_visible = True
        self.last_mouse_pos = (0, 0)
        self.last_mouse_move_time = time.time()
        self.idle_timeout = 7.0 
        self.next_random_track_time = time.time() + 2
        
        # External control (for hand tracking)
        self.external_control = False

        # Blink
        self.last_blink_time = time.time()
        self.next_blink_wait = random.uniform(2, 6)
        self.blink_state = "idle"
        self.blink_start_time = 0
        self.current_lid_pos = -60

        # Veri Yükleme
        self.emotions_data = {}
        self.load_emotions_from_disk()

    def load_emotions_from_disk(self):
        data_path = "assets/data"
        if not os.path.exists(data_path): return
        for file_name in os.listdir(data_path):
            if file_name.endswith(".json"):
                name = file_name.replace(".json", "")
                try:
                    with open(os.path.join(data_path, file_name), "r") as f:
                        self.emotions_data[name] = json.load(f)
                except: pass

    def set_emotion(self, emotion_name):
        """Editörden gelen JSON verisini okur ve hedefleri günceller"""
        if emotion_name in self.emotions_data:
            data = self.emotions_data[emotion_name]
            
            # Temel Ayarlar
            self.target_lid_y = data.get("lid_y", -60)
            self.target_happy_squint = data.get("squint", 0.0)
            self.target_brow_width = data.get("brow_width", 120)
            self.is_hollow = data.get("is_hollow", False)
            self.eye_visible = data.get("eye_visible", True) 
            
            # Segmentleri Yükle (5 Nokta)
            if "segments" in data:
                self.target_segments = [s.copy() for s in data["segments"]]
            
            # Ekstraları Yükle
            # Dict dönüşümünü burada garantiye alalım
            raw_extras = data.get("extras", [])
            processed_extras = []
            for item in raw_extras:
                if isinstance(item, list):
                    processed_extras.append({'points': item, 'type': 'draw', 'thick': 3})
                elif isinstance(item, dict):
                    processed_extras.append(item)
            self.current_extras = processed_extras
            
            self.neutral_brow_visible = True
        else:
            print(f"Emotion {emotion_name} not found!")

    def update(self):
        current_time = time.time()
        
        # --- MOUSE & AI (skip if external_control is enabled) ---
        if not self.external_control:
            mx, my = pygame.mouse.get_pos()
            
            if (mx, my) != self.last_mouse_pos:
                self.last_mouse_move_time = current_time
                self.last_mouse_pos = (mx, my)
            
            if not self.is_aggro:
                if current_time > self.next_random_track_time:
                    if random.random() < 0.4: self.is_tracking = True
                    self.next_random_track_time = current_time + random.uniform(2, 5)
            
            if self.is_tracking or (current_time - self.last_mouse_move_time < self.idle_timeout):
                norm_x = (mx - config.SCREEN_WIDTH/2) / (config.SCREEN_WIDTH/2)
                norm_y = (my - config.SCREEN_HEIGHT/2) / (config.SCREEN_HEIGHT/2)
                self.target_look_x = max(-0.7, min(0.7, norm_x))
                self.target_look_y = max(-0.5, min(0.5, norm_y))
            else:
                self.target_look_x = 0; self.target_look_y = 0

        self.look_x = lerp_smooth(self.look_x, self.target_look_x, 0.1)
        self.look_y = lerp_smooth(self.look_y, self.target_look_y, 0.1)

        # --- GÖZ KIRPMA ---
        if self.eye_visible:
            if self.blink_state == "idle":
                if current_time - self.last_blink_time > self.next_blink_wait:
                    self.blink_state = "closing"; self.blink_start_time = current_time
                self.current_lid_pos = lerp_smooth(self.current_lid_pos, self.target_lid_y, self.smooth_speed)
                
            elif self.blink_state == "closing":
                prog = (current_time - self.blink_start_time) * 15
                self.current_lid_pos = lerp_smooth(self.current_lid_pos, 150, prog)
                if prog >= 1: self.blink_state = "opening"; self.blink_start_time = current_time
                
            elif self.blink_state == "opening":
                prog = (current_time - self.blink_start_time) * 10
                self.current_lid_pos = lerp_smooth(150, self.target_lid_y, prog)
                if prog >= 1: 
                    self.blink_state = "idle"; self.last_blink_time = current_time; self.next_blink_wait = random.uniform(2, 6)
        else:
            self.current_lid_pos = self.target_lid_y

        # --- SEGMENT İNTERPOLASYONU ---
        self.current_brow_width = lerp_smooth(self.current_brow_width, self.target_brow_width, self.smooth_speed)
        
        for i in range(len(self.current_segments)):
            curr = self.current_segments[i]
            targ = self.target_segments[i]
            curr["off_y"] = lerp_smooth(curr["off_y"], targ["off_y"], self.smooth_speed)
            curr["thick"] = lerp_smooth(curr["thick"], targ["thick"], self.smooth_speed)
            self.current_happy_squint = lerp_smooth(self.current_happy_squint, self.target_happy_squint, self.smooth_speed)
            self.current_glow_intensity = lerp_smooth(self.current_glow_intensity, self.target_glow_intensity, self.smooth_speed * 0.5)

    # --- GELİŞMİŞ ÇİZİM FONKSİYONLARI ---

    def draw_advanced_brow(self, surface, cx, cy_lid, width, is_left):
        """5 noktalı kaş ve SİYAH PERDE çizer"""
        segments = self.current_segments
        nodes = []
        
        start_x = cx - (width / 2)
        step_x = width / 4
        
        for i, seg in enumerate(segments):
            nx = start_x + (i * step_x)
            ny = cy_lid + seg["off_y"]
            
            if not is_left:
                dist = nx - cx
                nx = cx - dist # X'i tersle
            nodes.append((nx, ny))

        calc_nodes = [nodes[0]] + nodes + [nodes[-1]]
        points_to_draw = []
        thicks_to_draw = []
        
        for i in range(len(nodes) - 1):
            p0, p1, p2, p3 = calc_nodes[i], calc_nodes[i+1], calc_nodes[i+2], calc_nodes[i+3]
            th0 = segments[max(0, i-1)]["thick"]; th1 = segments[i]["thick"]
            th2 = segments[i+1]["thick"]; th3 = segments[min(4, i+2)]["thick"]
            
            for s in range(10):
                t = s / 10
                x = catmull_rom_spline(p0[0], p1[0], p2[0], p3[0], t)
                y = catmull_rom_spline(p0[1], p1[1], p2[1], p3[1], t)
                th = catmull_rom_spline(th0, th1, th2, th3, t)
                
                # Titreme Ekle
                off_x = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
                off_y = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
                points_to_draw.append((x + off_x, y + off_y))
                thicks_to_draw.append(max(2, th))
        
        if len(points_to_draw) > 2:
            # 1. SİYAH PERDE (THE CURTAIN)
            mask_poly = list(points_to_draw)
            top_y = -500 # Çok yukarı
            mask_poly.append((points_to_draw[-1][0] + 100, top_y))
            mask_poly.append((points_to_draw[0][0] - 100, top_y))
            pygame.draw.polygon(surface, config.COLOR_BLACK, mask_poly)

            # 2. KAŞ ÇİZGİSİ
            for i in range(len(points_to_draw) - 1):
                start = points_to_draw[i]
                end = points_to_draw[i+1]
                w = int(thicks_to_draw[i])
                
                if self.glow_enabled:
                    pygame.draw.line(surface, (*config.COLOR_UZI_PURPLE[:3], 100), start, end, w + 4)
                pygame.draw.line(surface, config.COLOR_UZI_PURPLE, start, end, w)

    def draw_extras(self, surface, cx, cy, is_left, gaze_x, gaze_y):
        """Kalemle çizilenleri göze yapıştırır (KESME DAHİL)"""
        for item in self.current_extras:
            if 'points' not in item or len(item['points']) < 2: continue
            
            points = item['points']
            mode = item.get('type', 'draw') 
            thick = item.get('thick', 3)
            
            transformed = []
            for p in points:
                px, py = p[0], p[1]
                if not is_left: px = -px # Sağ göz için ayna
                
                # Parenting
                final_x = cx + px + gaze_x
                final_y = cy + py + gaze_y
                
                # Titreme
                off_x = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
                off_y = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
                transformed.append((final_x + off_x, final_y + off_y))
            
            if len(transformed) > 1:
                if mode == 'cut':
                    # --- KESME MODU (DÜZELTİLDİ) ---
                    # Sadece çizgiyi değil, çizginin ÜSTÜNÜ kapatan bir alan çiziyoruz.
                    mask_poly = list(transformed)
                    # Çok yukarıda iki nokta ekle (perde oluşturmak için)
                    mask_poly.append((transformed[-1][0] + 500, -500)) 
                    mask_poly.append((transformed[0][0] - 500, -500))
                    
                    # Bu alanı siyahla doldur (Maskeleme)
                    pygame.draw.polygon(surface, config.COLOR_BLACK, mask_poly)
                else:
                    # NORMAL ÇİZİM MODU
                    pygame.draw.lines(surface, config.COLOR_UZI_PURPLE, False, transformed, thick)

    def draw_eye_system(self, x, y, is_left):
        off_x = random.randint(-int(self.shake_intensity), int(self.shake_intensity))
        off_y = random.randint(-int(self.shake_intensity), int(self.shake_intensity))

        # Adjust gaze range based on display mode
        single_eye_mode = getattr(config, 'SINGLE_EYE_MODE', False)
        is_round_display = config.SCREEN_WIDTH == 240 and config.SCREEN_HEIGHT == 240

        if single_eye_mode:
            # Single eye on round display
            gaze_x = self.look_x * 30
            gaze_y = self.look_y * 25
            surf_w, surf_h = config.EYE_WIDTH + 120, config.EYE_HEIGHT + 150
        elif is_round_display:
            # Two eyes on round display - smaller surface
            gaze_x = self.look_x * 15
            gaze_y = self.look_y * 12
            surf_w, surf_h = config.EYE_WIDTH + 60, config.EYE_HEIGHT + 80
        else:
            # Normal display
            gaze_x = self.look_x * 60
            gaze_y = self.look_y * 45
            surf_w, surf_h = config.EYE_WIDTH + 200, config.EYE_HEIGHT + 300

        cx, cy = surf_w // 2, surf_h // 2
        eye_layer = pygame.Surface((surf_w, surf_h), pygame.SRCALPHA)

        # 1. GÖZ YUVARLAĞI
        if self.eye_visible:
            rect_main = pygame.Rect(0, 0, config.EYE_WIDTH, config.EYE_HEIGHT)
            rect_main.center = (cx + gaze_x, cy + gaze_y)
            
            glow_mul = self.current_glow_intensity
            if self.glow_enabled:
                 pygame.draw.ellipse(eye_layer, (*config.COLOR_UZI_PURPLE, int(30*glow_mul)), rect_main.inflate(30*glow_mul, 30*glow_mul))
                 pygame.draw.ellipse(eye_layer, (*config.COLOR_UZI_PURPLE, int(60*glow_mul)), rect_main.inflate(10*glow_mul, 10*glow_mul))
            
            thickness = 12 if self.is_hollow else 0
            pygame.draw.ellipse(eye_layer, config.COLOR_UZI_PURPLE, rect_main, thickness)

        # 2. EKSTRA ÇİZİMLER (Kesme burada işlenir)
        self.draw_extras(eye_layer, cx, cy, is_left, gaze_x, gaze_y)

        # 3. YANAK KISMASI (Squint)
        if self.current_happy_squint > 0.01 and self.eye_visible:
            squint_h = config.EYE_HEIGHT * self.current_happy_squint
            cheek_rect = pygame.Rect(0, 0, config.EYE_WIDTH + 60, squint_h + 50)
            cheek_rect.midtop = (cx + gaze_x, (cy + gaze_y) + (config.EYE_HEIGHT/2) - squint_h + 8)
            pygame.draw.ellipse(eye_layer, config.COLOR_BLACK, cheek_rect)

        # 4. KAŞ
        if self.neutral_brow_visible:
            brow_y = cy + self.current_lid_pos + (gaze_y * 0.7)
            brow_cx = cx + (gaze_x * 0.7)
            self.draw_advanced_brow(eye_layer, brow_cx, brow_y, self.current_brow_width, is_left)

        self.screen.blit(eye_layer, (x + off_x, y + off_y))

    def draw(self):
        self.update()
        self.screen.fill(config.COLOR_BLACK)

        # Check for single eye mode (GC9A01A round display)
        single_eye_mode = getattr(config, 'SINGLE_EYE_MODE', False)
        is_round_display = config.SCREEN_WIDTH == 240 and config.SCREEN_HEIGHT == 240

        if single_eye_mode:
            # Single eye mode for round display (240x240)
            surf_w = config.EYE_WIDTH + 120
            surf_h = config.EYE_HEIGHT + 150
            eye_x = (config.SCREEN_WIDTH - surf_w) // 2
            eye_y = (config.SCREEN_HEIGHT - surf_h) // 2
            self.draw_eye_system(eye_x, eye_y, is_left=True)
        elif is_round_display:
            # Dual eye mode for round display (240x240)
            # ตาเล็กลง วางชิดกัน
            eye_spacing = 8
            surf_w = config.EYE_WIDTH + 60
            surf_h = config.EYE_HEIGHT + 80
            total_width = (surf_w * 2) + eye_spacing
            start_x = (config.SCREEN_WIDTH - total_width) // 2
            eye_y = (config.SCREEN_HEIGHT - surf_h) // 2 - 10

            self.draw_eye_system(start_x, eye_y, is_left=True)
            self.draw_eye_system(start_x + surf_w + eye_spacing, eye_y, is_left=False)
        else:
            # Dual eye mode for normal display
            eye_spacing = 20
            offset_y = (config.EYE_HEIGHT + 300) // 2
            final_y = self.center_y - offset_y
            center_x_adjusted = self.center_x - 100

            self.draw_eye_system(center_x_adjusted - config.EYE_WIDTH - eye_spacing, final_y, is_left=True)
            self.draw_eye_system(center_x_adjusted + config.EYE_WIDTH + eye_spacing, final_y, is_left=False)