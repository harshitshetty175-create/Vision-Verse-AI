"""AIR INVADERS: AETHER PROTOCOL
Workshop-ready Pygame + MediaPipe Tasks + trained Random Forest.

Run from the project folder after installing dependencies:
    python air_invaders_aether_competition.py

Required beside this file: config.py, features.py, vision_utils.py,
models/hand_landmarker.task, gesture_model.pkl

CV focus: hand landmarks -> 42 normalised features -> classifier -> stable gesture events.
Keyboard 1-4 is a testing fallback; it does not replace the trained model.
"""
from __future__ import annotations
import math, random, time, sys, json, urllib.request, urllib.error, hashlib
from collections import deque, Counter
from pathlib import Path

import cv2
import numpy as np
import pygame
import joblib
import config as cfg
from features import extract_features
from vision_utils import create_hand_landmarker, detect_hands

WIDTH, HEIGHT = 1280, 720
FPS = 60
ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / getattr(cfg, "MODEL_PATH", "gesture_model.pkl")
HAND_PATH = ROOT / getattr(cfg, "HAND_MODEL_PATH", "models/hand_landmarker.task")
if not HAND_PATH.exists():
    fallback_hand_path = ROOT / "hand_landmarker.task"
    if fallback_hand_path.exists():
        HAND_PATH = fallback_hand_path
GESTURES = ("fist", "palm", "peace", "point")
KEYS = {pygame.K_1: "fist", pygame.K_2: "palm", pygame.K_3: "peace", pygame.K_4: "point"}
BG = (5, 8, 24); WHITE = (235, 247, 255); MUTED = (132, 155, 188)
CYAN = (63, 235, 255); BLUE = (65, 123, 255); PURPLE = (184, 95, 255)
PINK = (255, 78, 173); RED = (255, 83, 103); GOLD = (255, 205, 91)
GREEN = (101, 255, 177); PANEL = (12, 20, 43)


# ---------------- WORKSHOP COMPETITION SETTINGS ----------------
# Paste your deployed Google Apps Script Web App URL into this setting.
# Until configured, the game saves completed runs to a local CSV file.
LEADERBOARD_URL = "https://script.google.com/macros/s/AKfycbzXTzJDwj0XbrXI5LhDDs2o-Nbhl_28c7hH2Jgn4BQttnezC6GDg7RDiJ9aDUQfmNYD/exec"  # Example: "https://script.google.com/macros/s/YOUR_DEPLOYMENT_ID/exec"
WORKSHOP_ID = "AETHER-WORKSHOP-2026"
LEVEL_SCORE_TARGETS = [0, 180, 420, 720]  # kept for compatibility with existing UI
LEVEL_NAMES = ["TARGET LOCK", "SHIELDBREAK RUN", "FIST OVERDRIVE", "PHANTOM STRIKE"]
LEVEL_GOALS = {1: 12, 2: 10, 3: 12}  # beginner-friendly kill targets
LEVEL_TIMERS = {3: 90, 4: 100}  # generous time limits for workshop players
LOCAL_RESULTS_PATH = ROOT / "aether_results.csv"

def submit_score_online(payload):
    """Send one completed run to the Apps Script endpoint; return (ok, message)."""
    if not LEADERBOARD_URL.strip():
        return False, "Google Sheets URL not configured"
    try:
        body = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            LEADERBOARD_URL,
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=8) as response:
            result = json.loads(response.read().decode("utf-8"))
        return bool(result.get("ok")), str(result.get("message", "No response message"))
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"

def _leaderboard_sort_key(row):
    """Rank winners first, then fewer missed targets, faster time, and higher score."""
    won = str(row.get("boss_defeated", "")).strip().lower() in ("true", "1", "yes")
    try:
        missed = max(0, int(float(row.get("missed_targets", 0))))
    except (TypeError, ValueError):
        missed = 999  # Unknown miss count should not outrank a verified lower count.
    try:
        duration = float(row.get("duration_seconds", float("inf")))
    except (TypeError, ValueError):
        duration = float("inf")
    try:
        score = int(float(row.get("score", 0)))
    except (TypeError, ValueError):
        score = 0
    return (0 if won else 1, missed, duration, -score)


def fetch_leaderboard():
    """Fetch online scores; fall back to local results and rank winners by time."""
    rows = []
    if LEADERBOARD_URL.strip():
        try:
            with urllib.request.urlopen(LEADERBOARD_URL, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
            if data.get("ok"):
                rows = data.get("scores", [])
        except Exception:
            rows = []
    if not rows and LOCAL_RESULTS_PATH.exists():
        import csv
        try:
            with LOCAL_RESULTS_PATH.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        except Exception:
            rows = []
    return sorted(rows, key=_leaderboard_sort_key)[:20]


def save_local_score(payload):
    """Append an offline result, migrating older CSV headers safely if needed."""
    import csv
    path = LOCAL_RESULTS_PATH
    fields = ["workshop_id", "participant_code", "player_name", "score",
              "level_reached", "boss_defeated", "duration_seconds", "missed_targets",
              "accuracy", "submitted_at", "run_id"]
    existing_rows = []
    if path.exists() and path.stat().st_size:
        with path.open("r", newline="", encoding="utf-8") as f:
            existing_rows = list(csv.DictReader(f))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in existing_rows:
            writer.writerow({k: row.get(k, "") for k in fields})
        writer.writerow({k: payload.get(k, "") for k in fields})

pygame.init()
# A small original ambient soundtrack is synthesised locally, so no external
# music file or internet connection is needed. If audio is unavailable, the game
# continues silently.
def start_background_music():
    try:
        if not pygame.mixer.get_init():
            pygame.mixer.init(frequency=22050, size=-16, channels=2, buffer=512)
        sample_rate = 22050
        seconds = 16
        count = sample_rate * seconds
        t = np.arange(count, dtype=np.float32) / sample_rate
        # Slow, minor-key space pad with a quiet pulsing bass and high shimmer.
        progression = [(110.00, 164.81, 220.00), (98.00, 146.83, 196.00),
                       (130.81, 196.00, 261.63), (87.31, 130.81, 174.61)]
        music = np.zeros(count, dtype=np.float32)
        segment = seconds / len(progression)
        for i, chord in enumerate(progression):
            mask = (t >= i * segment) & (t < (i + 1) * segment)
            local = t[mask] - i * segment
            fade = np.minimum(1.0, local / 0.45) * np.minimum(1.0, (segment - local) / 0.45)
            layer = np.zeros(local.shape, dtype=np.float32)
            for j, freq in enumerate(chord):
                layer += np.sin(2 * np.pi * freq * local + j * 0.7) * (0.12 / (j + 1))
                layer += np.sin(2 * np.pi * freq * 2.01 * local) * (0.025 / (j + 1))
            # Quiet repeating pulse adds a sense of movement without overpowering speech.
            pulse = (0.5 + 0.5 * np.sin(2 * np.pi * 1.5 * local))
            music[mask] += layer * fade + (np.sin(2 * np.pi * (chord[0] / 2) * local) * pulse * fade * 0.045)
        # Smooth loop boundaries and keep volume workshop-friendly.
        edge = int(sample_rate * 0.35)
        music[:edge] *= np.linspace(0, 1, edge, dtype=np.float32)
        music[-edge:] *= np.linspace(1, 0, edge, dtype=np.float32)
        shimmer = np.sin(2 * np.pi * 440 * t) * 0.012 + np.sin(2 * np.pi * 554.37 * t) * 0.008
        music += shimmer * np.sin(np.pi * np.minimum(t, seconds - t) / 0.7).clip(0, 1)
        music = np.clip(music, -0.85, 0.85)
        stereo = np.column_stack((music, music * 0.94))
        audio = (stereo * 32767).astype(np.int16)
        sound = pygame.sndarray.make_sound(audio)
        sound.set_volume(0.28)
        sound.play(loops=-1)
        return sound
    except Exception as exc:
        print(f"Background music unavailable; continuing without audio ({type(exc).__name__}).")
        return None

pygame.display.set_caption("AETHER PROTOCOL // Gesture-Controlled Space Defense")
SCREEN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
CLOCK = pygame.time.Clock()
FONT = pygame.font.SysFont("segoeui", 20)
SMALL = pygame.font.SysFont("segoeui", 15)
MED = pygame.font.SysFont("segoeui", 28, bold=True)
BIG = pygame.font.SysFont("segoeui", 58, bold=True)
HUGE = pygame.font.SysFont("segoeui", 76, bold=True)


def text(surface, value, pos, font=FONT, color=WHITE, center=False):
    img = font.render(str(value), True, color)
    rect = img.get_rect()
    rect.center = pos if center else rect.center
    if not center: rect.topleft = pos
    surface.blit(img, rect)
    return rect


def glow(surface, pos, radius, color, alpha=48):
    x, y = int(pos[0]), int(pos[1])
    layer = pygame.Surface((radius*4+4, radius*4+4), pygame.SRCALPHA)
    c = (color[0], color[1], color[2], alpha)
    pygame.draw.circle(layer, c, (layer.get_width()//2, layer.get_height()//2), radius*2)
    surface.blit(layer, (x-layer.get_width()//2, y-layer.get_height()//2), special_flags=pygame.BLEND_PREMULTIPLIED)
    pygame.draw.circle(surface, color, (x,y), radius, 1)


def panel(surface, rect, fill=(12,20,43,220), border=(42,78,115)):
    layer = pygame.Surface((rect[2], rect[3]), pygame.SRCALPHA)
    pygame.draw.rect(layer, fill, layer.get_rect(), border_radius=15)
    surface.blit(layer, rect[:2])
    pygame.draw.rect(surface, border, rect, 1, border_radius=15)


class Particle:
    def __init__(self, x, y, color, speed=180):
        a = random.random()*math.tau; v=random.uniform(speed*.25,speed)
        self.x=x; self.y=y; self.vx=math.cos(a)*v; self.vy=math.sin(a)*v
        self.life=self.maxlife=random.uniform(.25,.7); self.color=color; self.r=random.randint(2,5)
    def update(self, dt):
        self.x+=self.vx*dt; self.y+=self.vy*dt; self.vx*=.98; self.vy*=.98; self.life-=dt
    def draw(self, s):
        if self.life>0:
            r=max(1,int(self.r*self.life/self.maxlife)); pygame.draw.circle(s,self.color,(int(self.x),int(self.y)),r)


class Projectile:
    def __init__(self, x, y, color=CYAN, target=None, power=1):
        self.x=float(x); self.y=float(y); self.prev_x=float(x); self.prev_y=float(y)
        self.color=color; self.target=target; self.power=power; self.speed=780
    def update(self, dt):
        self.prev_x, self.prev_y = self.x, self.y
        self.y -= self.speed * dt
    def draw(self,s):
        glow(s,(self.x,self.y),5,self.color,38)
        pygame.draw.line(s,self.color,(self.x,self.y+13),(self.x,self.y-15),4)
        pygame.draw.line(s,WHITE,(self.x,self.y+7),(self.x,self.y-10),1)


class Enemy:
    SPECS = {
        "drone": (CYAN, 1, 22, "fist"),
        "flare": (GOLD, 1, 24, "palm"),
        "wraith": (PURPLE, 1, 24, "peace"),
        "scout": (GREEN, 1, 19, "point"),
        "brute": (PINK, 2, 32, "fist"),
    }
    def __init__(self, kind, level, x=None, y=-45):
        self.kind=kind; self.color,self.hp,self.radius,self.counter= self.SPECS[kind]
        self.maxhp=self.hp; self.x=float(x if x is not None else random.randint(90,WIDTH-90)); self.y=float(y)
        self.speed=random.uniform(68,92)+(level-1)*5; self.phase=random.random()*6.28
        self.vx=random.choice([-1,1])*random.uniform(12,34); self.flash=0
    def update(self,dt,t):
        self.prev_x, self.prev_y = self.x, self.y
        self.y+=self.speed*dt; self.x+=math.sin(t*1.7+self.phase)*self.vx*dt
        self.x=max(55,min(WIDTH-55,self.x)); self.flash=max(0,self.flash-dt)
    def draw(self,s,t):
        x,y=int(self.x),int(self.y); r=self.radius
        glow(s,(x,y),r+7,self.color,24)
        if self.kind in ("drone","brute"):
            pts=[(x-r,y-r//3),(x-r//2,y-r),(x+r//2,y-r),(x+r,y-r//3),(x+r//2,y+r),(x-r//2,y+r)]
            pygame.draw.polygon(s,(18,28,53),pts); pygame.draw.polygon(s,WHITE if self.flash else self.color,pts,2)
            pygame.draw.line(s,self.color,(x-r//2,y),(x+r//2,y),2)
            pygame.draw.circle(s,self.color,(x-7,y+4),3); pygame.draw.circle(s,self.color,(x+7,y+4),3)
        elif self.kind=="flare":
            pts=[(x,y-r-7),(x+r,y-r//3),(x+r//2,y+r),(x,y+r//2),(x-r//2,y+r),(x-r,y-r//3)]
            pygame.draw.polygon(s,(45,26,40),pts); pygame.draw.polygon(s,self.color,pts,3)
            pygame.draw.circle(s,GOLD,(x,y),6)
        elif self.kind=="wraith":
            pygame.draw.circle(s,(27,21,60),(x,y-3),r); pygame.draw.circle(s,self.color,(x,y-3),r,2)
            pygame.draw.arc(s,self.color,(x-r,y-r//3,r*2,r*2),0,math.pi,3)
            pygame.draw.circle(s,WHITE,(x-7,y-4),4); pygame.draw.circle(s,WHITE,(x+7,y-4),4)
        else:
            pts=[(x,y-r),(x+r,y),(x,y+r),(x-r,y)]
            pygame.draw.polygon(s,(16,40,42),pts); pygame.draw.polygon(s,self.color,pts,3)
            pygame.draw.circle(s,WHITE,(x,y),r//3,2)
        if self.hp>1:
            pygame.draw.rect(s,(45,54,76),(x-r,y-r-10,r*2,4),border_radius=2)
            pygame.draw.rect(s,GREEN,(x-r,y-r-10,int(2*r*self.hp/self.maxhp),4),border_radius=2)


def swept_circle_hit(ax0, ay0, ax1, ay1, bx0, by0, bx1, by1, radius):
    """Return True if two moving circular objects overlap at any point this frame."""
    # Work in relative coordinates so both objects' movement is accounted for.
    rx0, ry0 = ax0 - bx0, ay0 - by0
    rvx = (ax1 - ax0) - (bx1 - bx0)
    rvy = (ay1 - ay0) - (by1 - by0)
    denom = rvx * rvx + rvy * rvy
    t = 0.0 if denom == 0 else max(0.0, min(1.0, -(rx0 * rvx + ry0 * rvy) / denom))
    dx, dy = rx0 + t * rvx, ry0 + t * rvy
    return dx * dx + dy * dy <= radius * radius


class AetherGame:
    def __init__(self, model, detector, camera):
        self.model=model; self.detector=detector; self.camera=camera
        self.state="TITLE"; self.story_page=0; self.t0=time.monotonic(); self.t=0; self.score=0; self.lives=7; self.missed_targets=0; self.combo=0
        self.player_name=""; self.participant_code=""; self.name_edit="name"; self.registration_error=""; self.run_id=hashlib.sha256(f"{time.time()}-{random.random()}".encode()).hexdigest()[:12]
        self.run_started_at=0; self.run_duration_seconds=0.0; self.missed_targets=0; self.run_submitted=False; self.result_message=""; self.result_color=GOLD; self.max_level_reached=1; self.boss_defeated=False; self.shots_fired=0; self.shots_hit=0; self.leaderboard=[]
        self.player_x=WIDTH/2; self.player_y=HEIGHT-115; self.energy=100; self.shield_until=0
        self.enemies=[]; self.shots=[]; self.particles=[]; self.enemy_shots=[]; self.stars=[]
        for _ in range(180): self.stars.append([random.randrange(WIDTH),random.randrange(HEIGHT),random.uniform(12,90),random.choice([1,1,1,2])])
        self.last_spawn=0; self.last_fire=-999; self.base_fire_interval=0.24; self.boost_fire_interval=0.085; self.last_shield=-999; self.last_arc=-999; self.last_hit=0; self.level=1
        self.history=deque(maxlen=getattr(cfg,"SMOOTHING_WINDOW",3)); self.candidate="none"; self.candidate_since=0
        self.gesture="none"; self.raw="none"; self.confidence=0; self.fingertip=(WIDTH/2,HEIGHT/2); self.hand_found=False
        self.last_valid_gesture_at=0.0
        self.simulated=None; self.debug=False; self.camera_frame=None; self.camera_rgb=None; self.last_camera_ok=False
        self.boss=None; self.boss_hp=0; self.boss_last_shot=0; self.toast=""; self.toast_until=0
        self.wave_label="FIRST CONTACT"; self.wave_until=0; self.game_over_reason=""
        self.level_kills=0; self.level_started_at=0; self.level_transition_at=0; self.level_transition_pending=False
        self.shield_uses=0; self.level2_shield_used=False; self.level_time_limit=0; self.level4_started=False

    def reset(self):
        self.state="PLAY"; self.score=0; self.lives=7; self.missed_targets=0; self.run_duration_seconds=0.0; self.combo=0; self.player_x=WIDTH/2; self.energy=100
        self.shield_until=0; self.enemies.clear(); self.shots.clear(); self.particles.clear(); self.enemy_shots.clear()
        self.last_spawn=self.t; self.last_fire=-999; self.last_shield=-999; self.last_arc=-999; self.last_hit=0; self.level=1
        self.last_gesture="none"
        self.level_kills=0; self.level_started_at=self.t; self.level_transition_at=0; self.level_transition_pending=False
        self.shield_uses=0; self.level2_shield_used=False; self.level_time_limit=0; self.level4_started=False
        self.boss=None; self.boss_hp=0; self.wave_label="FIRST CONTACT"; self.wave_until=self.t+2.4
        self.run_id=hashlib.sha256(f"{time.time()}-{random.random()}".encode()).hexdigest()[:12]
        self.run_started_at=time.monotonic(); self.run_submitted=False; self.result_message=""; self.max_level_reached=1; self.boss_defeated=False; self.shots_fired=0; self.shots_hit=0
        self.toast="MISSION START"; self.toast_until=self.t+1.6

    def read_camera(self, dt):
        if self.camera is None: return
        ok, frame=self.camera.read()
        if not ok: self.hand_found=False; return
        frame=cv2.flip(frame,1); self.camera_frame=cv2.resize(frame,(320,240))
        rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB); self.camera_rgb=rgb
        self.hand_found=False; self.raw="none"; self.confidence=0
        timestamp=max(1,int(self.t*1000))
        try:
            result=detect_hands(self.detector,rgb,timestamp)
            if result.hand_landmarks:
                hand=result.hand_landmarks[0]; self.hand_found=True
                self.fingertip=(float(hand[8].x)*WIDTH,float(hand[8].y)*HEIGHT)
                feats=extract_features(hand)
                if feats is not None and self.model is not None:
                    probs=self.model.predict_proba(feats.reshape(1,-1))[0]
                    ix=int(np.argmax(probs)); self.raw=str(self.model.classes_[ix]); self.confidence=float(probs[ix])
                    if self.confidence < getattr(cfg,"CONFIDENCE_THRESHOLD",.60): self.raw="none"
                elif feats is not None and self.model is None:
                    # No trained gesture model is available yet. Use the tracked
                    # index fingertip as POINT so the ship always responds to hand
                    # movement and keyboard gesture keys remain optional fallbacks.
                    self.raw="point"
                    self.confidence=1.0
        except Exception as e:
            self.toast=f"VISION: {type(e).__name__}"; self.toast_until=self.t+2
        if self.simulated: self.raw=self.simulated
        # Hysteresis: preserve POINT/FIST through brief classifier flicker,
        # but require a short stable confirmation before changing between them.
        previous = self.gesture
        candidate = self.raw
        if candidate in GESTURES:
            if candidate != self.candidate:
                self.candidate = candidate
                self.candidate_since = self.t
            transition_hold = getattr(cfg, "POINT_FIST_TRANSITION_HOLD", 0.10)
            normal_hold = getattr(cfg, "GESTURE_HOLD_SECONDS", 0.08)
            required_hold = (transition_hold if previous in ("point", "fist")
                             and candidate in ("point", "fist") and candidate != previous
                             else normal_hold)
            if candidate == previous or self.t - self.candidate_since >= required_hold:
                self.gesture = candidate
                self.last_valid_gesture_at = self.t
            # Until a new gesture is confirmed, keep the previous valid action.
        else:
            self.candidate = "none"
            self.candidate_since = self.t
            # Ignore very brief low-confidence frames while a hand is still tracked.
            grace = getattr(cfg, "GESTURE_LOSS_GRACE_SECONDS", 0.14)
            if not self.hand_found or self.t - self.last_valid_gesture_at > grace:
                self.gesture = "none"

    def event(self, gesture):
        if self.state!="PLAY" or gesture not in GESTURES: return
        if gesture=="point":
            # Point is for steering only; shooting is deliberately a separate gesture.
            self.player_x=max(70,min(WIDTH-70,self.fingertip[0]))
        elif gesture=="fist":
            self.fire(CYAN)
        elif gesture=="palm" and self.t-self.last_shield>1.8 and self.energy>=24:
            self.energy-=24; self.shield_until=self.t+1.25; self.last_shield=self.t
            self.shield_uses+=1; self.level2_shield_used=True
            self.toast="AEGIS SHIELD // ACTIVE"; self.toast_until=self.t+1.1
            for _ in range(24): self.particles.append(Particle(self.player_x,self.player_y,CYAN,100))
        elif gesture=="peace" and self.t-self.last_arc>2.2 and self.energy>=35:
            self.energy-=35; self.last_arc=self.t
            targets=sorted(self.enemies,key=lambda e:math.hypot(e.x-self.player_x,e.y-self.player_y))[:3]
            for e in targets:
                e.hp-=2; e.flash=.18
                self.particles.extend(Particle(e.x,e.y,PURPLE,260) for _ in range(10))
                if e.hp<=0: self.destroy(e)
            self.toast="ARC LIGHTNING // CHAIN"; self.toast_until=self.t+1.0

    def fire(self,color=CYAN,targeted=False):
        interval = getattr(self, "fire_interval", self.base_fire_interval)
        if self.t - self.last_fire < interval:
            return
        self.last_fire=self.t
        target=min(self.enemies,key=lambda e:math.hypot(e.x-self.player_x,e.y-self.player_y)) if targeted and self.enemies else None
        self.shots.append(Projectile(self.player_x,self.player_y-30,color,target)); self.shots_fired += 1
        self.particles.extend(Particle(self.player_x+random.randint(-8,8),self.player_y-27,color,60) for _ in range(3))

    def destroy(self,e):
        if e not in self.enemies: return
        self.enemies.remove(e); self.score+=10+min(30,self.combo*2); self.combo+=1; self.level_kills+=1
        self.energy=min(100,self.energy+5)
        self.particles.extend(Particle(e.x,e.y,e.color,270) for _ in range(25))
        if self.combo and self.combo%5==0:
            self.toast=f"{self.combo}X COMBO // ENERGY RESTORED"; self.toast_until=self.t+1.3

    def update(self, dt):
        self.t = time.monotonic() - self.t0
        for star in self.stars:
            star[1] = (star[1] + star[2] * dt) % HEIGHT
        if self.state != "PLAY":
            for particle in self.particles: particle.update(dt)
            self.particles = [p for p in self.particles if p.life > 0]
            return

        # Levels advance only when their objective is completed, never by score alone.
        if self.level_transition_pending:
            if self.t >= self.level_transition_at:
                self.level_transition_pending = False
                self.level += 1
                self.level_kills = 0
                self.level_started_at = self.t
                self.last_spawn = self.t
                self.enemies.clear(); self.shots.clear(); self.enemy_shots.clear()
                self.boss = None; self.boss_hp = 0
                self.wave_label = LEVEL_NAMES[self.level - 1]
                self.wave_until = self.t + 2.2
                if self.level == 3: self.level_time_limit = self.t + LEVEL_TIMERS[3]
                if self.level == 4:
                    self.level_time_limit = self.t + LEVEL_TIMERS[4]
                    self.boss = {"x": WIDTH / 2, "y": 145, "dir": 1, "phase": 0}
                    self.boss_hp = 40
                    self.level4_started = True
            else:
                for p in self.particles: p.update(dt)
                self.particles = [p for p in self.particles if p.life > 0]
                return

        self.max_level_reached = max(self.max_level_reached, self.level)
        # Unified controls: POINT steers and automatically fires; FIST boosts fire rate.
        # Shooting only runs while a confidently recognised hand gesture is active.
        if self.gesture == "point" and self.hand_found:
            target_x = max(70, min(WIDTH - 70, self.fingertip[0]))
            self.player_x += (target_x - self.player_x) * min(1.0, dt * 14)

        self.player_x = max(55, min(WIDTH - 55, self.player_x))
        self.energy = min(100, self.energy + 8 * dt)

        if self.hand_found and self.gesture in ("point", "fist"):
            # Point = steady auto-fire. Fist = optional rapid-fire boost.
            self.fire_interval = (self.boost_fire_interval if self.gesture == "fist"
                                  else self.base_fire_interval)
            self.fire(CYAN)

        # Shield and arc attack trigger once per gesture entry.
        if self.gesture != getattr(self, "last_gesture", "none"):
            if self.gesture in ("palm", "peace"):
                self.event(self.gesture)
        self.last_gesture = self.gesture

        # Slower later waves plus an active-enemy cap prevent overwhelming players.
        spawn_gap = {1: 1.15, 2: 1.80, 3: 2.20, 4: 2.60}.get(self.level, 2.20)
        active_enemy_cap = {1: 3, 2: 3, 3: 4, 4: 3}.get(self.level, 3)
        if (self.t - self.last_spawn > spawn_gap
                and len(self.enemies) < active_enemy_cap
                and (self.level < 4 or self.boss is not None)):
            self.last_spawn = self.t
            if self.level == 1: kinds = ["drone", "scout"]
            elif self.level == 2: kinds = ["flare", "wraith", "scout"]
            elif self.level == 3: kinds = ["drone", "brute", "brute", "scout"]
            else: kinds = ["wraith", "scout", "brute"]
            self.enemies.append(Enemy(random.choice(kinds), self.level))

        for e in self.enemies[:]:
            e.update(dt, self.t)
            if e.y > HEIGHT - 95:
                self.enemies.remove(e); self.lives -= 1; self.missed_targets += 1; self.combo = 0
                self.particles.extend(Particle(e.x, HEIGHT-105, RED, 220) for _ in range(12))
                if self.lives <= 0:
                    self.state = "GAMEOVER"; self.game_over_reason = "DEFENCE LINE BREACHED"; self.submit_result(); return

        # Swept collision: test the complete bullet path against each enemy's
        # complete movement path. This prevents fast bullets tunnelling through targets.
        for shot in self.shots[:]:
            shot.update(dt)
            hit = None
            candidates = [shot.target] if shot.target in self.enemies else self.enemies
            for enemy in candidates:
                ex0 = getattr(enemy, "prev_x", enemy.x)
                ey0 = getattr(enemy, "prev_y", enemy.y)
                if swept_circle_hit(shot.prev_x, shot.prev_y, shot.x, shot.y,
                                    ex0, ey0, enemy.x, enemy.y, enemy.radius + 9):
                    hit = enemy
                    break

            if hit is not None:
                hit.hp -= shot.power
                hit.flash = 0.24
                self.shots_hit += 1
                self.score += 2  # visible reward for every registered hit
                self.toast = "DIRECT HIT  +2"
                self.toast_until = self.t + 0.55
                # Bright impact burst makes even non-lethal hits obvious.
                self.particles.extend(
                    Particle(hit.x + random.randint(-hit.radius // 2, hit.radius // 2),
                             hit.y + random.randint(-hit.radius // 2, hit.radius // 2),
                             random.choice([WHITE, hit.color, CYAN]), 190)
                    for _ in range(9)
                )
                if hit.hp <= 0:
                    self.destroy(hit)
                if shot in self.shots:
                    self.shots.remove(shot)
            elif shot.y < -25 and shot in self.shots:
                self.shots.remove(shot)

        # Level 4 boss: moving target with timed aimed attacks and a visible health bar.
        if self.boss:
            b = self.boss; b["x"] += b["dir"] * (75 if self.level == 4 else 70) * dt
            if b["x"] < 170 or b["x"] > WIDTH-170: b["dir"] *= -1
            for shot in self.shots[:]:
                if abs(shot.x-b["x"]) < 125 and abs(shot.y-b["y"]) < 75:
                    self.boss_hp -= 1; self.shots.remove(shot)
                    self.particles.extend(Particle(b["x"]+random.randint(-70,70), b["y"]+random.randint(-30,30), PINK, 160) for _ in range(3))
            if self.t-self.boss_last_shot > 1.8:
                self.boss_last_shot = self.t
                self.enemy_shots.append([b["x"], b["y"]+45, random.uniform(-90,90), 210])
            if self.boss_hp <= 0:
                self.boss = None; self.score += 250; self.boss_defeated = True
                self.state = "VICTORY"; self.wave_label = "VOID CORE NEUTRALISED"
                self.submit_result()
                self.particles.extend(Particle(WIDTH/2+random.randint(-100,100), 145+random.randint(-50,50), random.choice([CYAN,PURPLE,PINK]), 400) for _ in range(100))
                return

        for shot in self.enemy_shots[:]:
            shot[0] += shot[2]*dt; shot[1] += shot[3]*dt
            if math.hypot(shot[0]-self.player_x, shot[1]-self.player_y) < 27 and self.t > self.shield_until and self.t-self.last_hit > 0.85:
                self.last_hit = self.t; self.lives -= 1; self.combo = 0
                self.particles.extend(Particle(self.player_x, self.player_y, RED, 230) for _ in range(15))
                if self.lives <= 0:
                    self.state = "GAMEOVER"; self.game_over_reason = "PILOT SIGNAL LOST"; self.submit_result(); return
                self.enemy_shots.remove(shot)
            elif shot[1] > HEIGHT:
                self.enemy_shots.remove(shot)

        # Timers for levels 3 and 4.
        if self.level in (3, 4) and self.t >= self.level_time_limit:
            self.state = "GAMEOVER"
            self.game_over_reason = "TIME EXPIRED // MISSION FAILED"
            self.submit_result(); return

        # Level completion conditions.
        if self.level == 1 and self.level_kills >= LEVEL_GOALS[1]:
            self.level_transition_pending = True; self.level_transition_at = self.t + 2.0
            self.wave_label = "TARGET LOCK COMPLETE // NEXT: SHIELDBREAK RUN"; self.wave_until = self.level_transition_at
        elif self.level == 2 and self.level_kills >= LEVEL_GOALS[2]:
            self.level_transition_pending = True; self.level_transition_at = self.t + 2.0
            self.wave_label = "SHIELDBREAK CLEARED // NEXT: FIST OVERDRIVE"; self.wave_until = self.level_transition_at
        elif self.level == 3 and self.level_kills >= LEVEL_GOALS[3]:
            self.level_transition_pending = True; self.level_transition_at = self.t + 2.0
            self.wave_label = "OVERDRIVE CLEARED // FINAL BOSS INCOMING"; self.wave_until = self.level_transition_at

        for p in self.particles: p.update(dt)
        self.particles = [p for p in self.particles if p.life > 0]

    def submit_result(self):
        """Submit a single official result at the end of the run."""
        if self.run_submitted or not self.player_name.strip() or not self.participant_code.strip():
            return
        self.run_submitted = True
        elapsed = round(max(0.0, time.monotonic() - self.run_started_at), 2) if self.run_started_at else 0.0
        self.run_duration_seconds = elapsed
        accuracy = round(100 * self.shots_hit / max(1, self.shots_fired), 2)
        payload = {
            "workshop_id": WORKSHOP_ID,
            "participant_code": self.participant_code.strip(),
            "player_name": self.player_name.strip()[:40],
            "score": int(self.score),
            "level_reached": int(self.max_level_reached),
            "boss_defeated": bool(self.boss_defeated),
            "duration_seconds": elapsed,
            "missed_targets": int(self.missed_targets),
            "accuracy": accuracy,
            "submitted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_id": self.run_id,
        }
        ok, message = submit_score_online(payload)
        if not ok:
            save_local_score(payload)
            self.result_message = "Saved locally; online submission failed"
            self.result_color = GOLD
            print("Score submission:", message)
        else:
            self.result_message = "Score submitted to workshop leaderboard"
            self.result_color = GREEN

    def draw_background(self):
        SCREEN.fill(BG)
        # subtle nebula gradients and parallax starfield
        if not hasattr(self, "_nebula_surface"):
            neb = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            pygame.draw.ellipse(neb, (27, 35, 104, 30), (90, 70, 560, 350))
            pygame.draw.ellipse(neb, (92, 22, 107, 24), (650, 160, 520, 420))
            self._nebula_surface = neb
        SCREEN.blit(self._nebula_surface, (0, 0))
        for x,y,speed,r in self.stars:
            col=(int(70+80*(r/2)),int(95+80*(r/2)),int(140+90*(r/2)))
            pygame.draw.circle(SCREEN,col,(int(x),int(y)),r)
        # distant orbital rings
        pygame.draw.ellipse(SCREEN,(20,45,79),(160,125,960,360),1)
        pygame.draw.ellipse(SCREEN,(17,34,68),(220,165,840,280),1)
        for y in range(0,HEIGHT,80): pygame.draw.line(SCREEN,(11,24,49),(0,y),(WIDTH,y),1)

    def draw_ship(self):
        x,y=int(self.player_x),int(self.player_y)
        glow(SCREEN,(x,y+12),28,CYAN,22)
        # engine plume
        pygame.draw.polygon(SCREEN,(49,157,255),[(x-9,y+25),(x,y+62+int(5*math.sin(self.t*18))),(x+9,y+25)])
        pygame.draw.polygon(SCREEN,(190,250,255),[(x-4,y+25),(x,y+48+int(4*math.sin(self.t*18))),(x+4,y+25)])
        pts=[(x,y-39),(x-20,y+16),(x-42,y+31),(x-13,y+27),(x,y+17),(x+13,y+27),(x+42,y+31),(x+20,y+16)]
        pygame.draw.polygon(SCREEN,(17,40,70),pts); pygame.draw.polygon(SCREEN,CYAN,pts,2)
        pygame.draw.polygon(SCREEN,(173,249,255),[(x,y-25),(x-7,y+9),(x,y+4),(x+7,y+9)])
        pygame.draw.line(SCREEN,(255,255,255),(x,y-20),(x,y+1),2)
        if self.t<self.shield_until:
            glow(SCREEN,(x,y),61,CYAN,25); pygame.draw.circle(SCREEN,(98,228,255),(x,y),52,2)
            pygame.draw.arc(SCREEN,WHITE,(x-55,y-55,110,110),self.t*2,self.t*2+2.7,2)

    def draw_boss(self):
        if not self.boss:return
        x,y=int(self.boss["x"]),int(self.boss["y"])
        glow(SCREEN,(x,y),100,PINK,35)
        pts=[(x-140,y-20),(x-85,y-55),(x-28,y-42),(x,y-62),(x+28,y-42),(x+85,y-55),(x+140,y-20),(x+92,y+36),(x+40,y+48),(x,y+32),(x-40,y+48),(x-92,y+36)]
        pygame.draw.polygon(SCREEN,(22,17,49),pts); pygame.draw.polygon(SCREEN,PINK,pts,3)
        pygame.draw.circle(SCREEN,(58,25,79),(x,y),27); pygame.draw.circle(SCREEN,PINK,(x,y),18,3); pygame.draw.circle(SCREEN,WHITE,(x,y),6)
        pygame.draw.rect(SCREEN,(44,29,63),(WIDTH//2-180,58,360,10),border_radius=5)
        pygame.draw.rect(SCREEN,PINK,(WIDTH//2-180,58,int(360*self.boss_hp/60),10),border_radius=5)
        text(SCREEN,"VOID CORE",(WIDTH//2,35),SMALL,PINK,True)

    def draw_camera_panel(self):
        rect=(WIDTH-270,HEIGHT-205,245,180); panel(SCREEN,rect,(8,15,31,235),(38,75,105))
        text(SCREEN,"LIVE VISION",(rect[0]+13,rect[1]+10),SMALL,CYAN)
        if self.camera_frame is not None:
            rgb=cv2.cvtColor(self.camera_frame,cv2.COLOR_BGR2RGB)
            surf=pygame.surfarray.make_surface(np.transpose(rgb,(1,0,2)))
            surf=pygame.transform.smoothscale(surf,(215,130)); SCREEN.blit(surf,(rect[0]+15,rect[1]+35))
            border=GREEN if self.hand_found else (70,90,120)
            pygame.draw.rect(SCREEN,border,(rect[0]+15,rect[1]+35,215,130),2,border_radius=5)
        else: text(SCREEN,"Camera not available",(rect[0]+20,rect[1]+70),SMALL,MUTED)
        status=self.gesture.upper() if self.gesture!="none" else "WAITING FOR GESTURE"
        text(SCREEN,status,(rect[0]+14,rect[1]+169),SMALL,GREEN if self.gesture!="none" else MUTED)

    def draw_hud(self):
        panel(SCREEN,(22,20,250,100)); text(SCREEN,"MISSION SCORE",(40,34),SMALL,MUTED); text(SCREEN,f"{self.score:06d}",(39,55),MED,WHITE)
        panel(SCREEN,(292,20,300,74)); text(SCREEN,"DEFENCE INTEGRITY",(310,31),SMALL,MUTED)
        for i in range(7):
            x = 310 + i * 38
            col = GREEN if i < self.lives else (48, 59, 79)
            pygame.draw.polygon(SCREEN, col, [(x+10, 49), (x+20, 59), (x+10, 69), (x, 59)])
        text(SCREEN, f"MISSED {self.missed_targets}/7", (442, 78), SMALL, GOLD, True)
        panel(SCREEN,(WIDTH-278,20,255,74)); text(SCREEN,"ENERGY CORE",(WIDTH-260,31),SMALL,MUTED)
        pygame.draw.rect(SCREEN,(38,50,76),(WIDTH-260,54,218,12),border_radius=6)
        pygame.draw.rect(SCREEN,CYAN,(WIDTH-260,54,int(218*self.energy/100),12),border_radius=6)
        text(SCREEN,f"{int(self.energy)}%",(WIDTH-42,51),SMALL,WHITE,True)
        text(SCREEN,f"SECTOR {self.level:02d}",(WIDTH//2,35),SMALL,CYAN,True)
        text(SCREEN,f"COMBO  x{1+self.combo//5}",(WIDTH//2,62),SMALL,GOLD,True)
        objectives = {1: f"POINT: MOVE + AUTO-FIRE  |  FIST: RAPID FIRE  |  TARGETS {self.level_kills}/{LEVEL_GOALS[1]}",
                      2: f"POINT: MOVE + AUTO-FIRE  |  FIST: RAPID FIRE  |  TARGETS {self.level_kills}/{LEVEL_GOALS[2]}",
                      3: f"POINT: AUTO-FIRE  |  FIST: RAPID FIRE  |  TARGETS {self.level_kills}/{LEVEL_GOALS[3]}  |  TIME {max(0,int(self.level_time_limit-self.t))}s",
                      4: f"POINT: AUTO-FIRE  |  FIST: RAPID FIRE  |  BOSS CORE  |  TIME {max(0,int(self.level_time_limit-self.t))}s"}
        if self.level in objectives:
            text(SCREEN, objectives[self.level], (WIDTH//2, 92), SMALL, GOLD if self.level >= 3 else CYAN, True)
        elapsed_now = (time.monotonic() - self.run_started_at) if self.run_started_at else self.run_duration_seconds
        timer_label = f"PLAY TIME  {int(elapsed_now // 60):02d}:{int(elapsed_now % 60):02d}"
        text(SCREEN, timer_label, (WIDTH//2, 119), SMALL, WHITE, True)
        # Gesture legend
        panel(SCREEN,(22,HEIGHT-72,690,48),(8,15,31,220),(32,58,88))
        labels=[("POINT","MOVE + FIRE",GOLD),("FIST","RAPID FIRE",CYAN),("PALM","SHIELD",GREEN),("PEACE","ARC",PURPLE)]
        xx=39
        for g,act,col in labels:
            text(SCREEN,g,(xx,HEIGHT-61),SMALL,col); text(SCREEN,act,(xx,HEIGHT-41),SMALL,WHITE); xx+=165
        if self.debug:
            panel(SCREEN,(22,132,270,88),(8,15,31,235),(70,80,120))
            text(SCREEN,f"raw prediction: {self.raw}",(36,143),SMALL,WHITE)
            text(SCREEN,f"active gesture: {self.gesture}",(36,164),SMALL,GOLD if self.gesture == "point" else CYAN)
            text(SCREEN,f"confidence: {self.confidence:.2f}  hand: {self.hand_found}",(36,185),SMALL,WHITE)
            text(SCREEN,"POINT = AUTO-FIRE  |  FIST = RAPID-FIRE BOOST",(36,204),SMALL,GREEN)

    def draw_title(self):
        # animated orbital logo
        cx,cy=WIDTH//2,230
        for r in (80,105,130): pygame.draw.ellipse(SCREEN,(25,67,105),(cx-r,cy-r//2,cx and r*2,r),1)
        glow(SCREEN,(cx,cy),58,CYAN,25)
        pygame.draw.polygon(SCREEN,(16,43,71),[(cx,cy-60),(cx-35,cy+25),(cx,cy+10),(cx+35,cy+25)])
        pygame.draw.polygon(SCREEN,CYAN,[(cx,cy-60),(cx-35,cy+25),(cx,cy+10),(cx+35,cy+25)],2)
        text(SCREEN,"AETHER",(cx,360),HUGE,WHITE,True)
        text(SCREEN,"PROTOCOL",(cx,419),MED,CYAN,True)
        text(SCREEN,"GESTURE-CONTROLLED ORBITAL DEFENCE",(cx,466),FONT,MUTED,True)
        panel(SCREEN,(cx-260,510,520,74),(9,18,39,235),(43,87,123))
        text(SCREEN,"DEFEND THE ORBIT. TRUST YOUR HANDS.",(cx,533),FONT,WHITE,True)
        text(SCREEN,"PRESS ENTER TO BEGIN BRIEFING  •  Q TO EXIT",(cx,558),SMALL,GREEN,True)
        text(SCREEN,"CV PIPELINE: LANDMARKS  →  FEATURES  →  CLASSIFIER  →  GAME ACTION",(cx,625),SMALL,MUTED,True)

    def draw_registration(self):
        self.draw_background()
        cx = WIDTH // 2
        text(SCREEN, "AETHER // PILOT REGISTRATION", (cx, 105), MED, CYAN, True)
        text(SCREEN, "Enter your name and the unique participant code given by the organiser.", (cx, 151), FONT, MUTED, True)
        panel(SCREEN, (cx-310, 205, 620, 265), (5, 12, 30, 235), (35, 83, 125))
        text(SCREEN, "PLAYER NAME", (cx-260, 235), SMALL, CYAN)
        pygame.draw.rect(SCREEN, (13, 25, 49), (cx-260, 260, 520, 48), border_radius=8)
        pygame.draw.rect(SCREEN, CYAN if self.name_edit=="name" else (50, 75, 103), (cx-260,260,520,48), 2, border_radius=8)
        text(SCREEN, self.player_name + ("|" if self.name_edit=="name" and int(self.t*2)%2==0 else ""), (cx-245,273), FONT, WHITE)
        text(SCREEN, "UNIQUE PARTICIPANT CODE", (cx-260, 330), SMALL, CYAN)
        pygame.draw.rect(SCREEN, (13, 25, 49), (cx-260, 355, 520, 48), border_radius=8)
        pygame.draw.rect(SCREEN, CYAN if self.name_edit=="code" else (50, 75, 103), (cx-260,355,520,48), 2, border_radius=8)
        text(SCREEN, self.participant_code + ("|" if self.name_edit=="code" and int(self.t*2)%2==0 else ""), (cx-245,368), FONT, WHITE)
        text(SCREEN, "Click a field or press TAB to switch. Enter to continue.", (cx, 505), FONT, WHITE, True)
        text(SCREEN, "One official attempt per participant code.", (cx, 540), SMALL, GOLD, True)
        if self.registration_error:
            text(SCREEN, self.registration_error, (cx, 575), SMALL, RED, True)
        text(SCREEN, "ENTER // CONTINUE     •     Q // QUIT", (cx, 635), SMALL, GREEN, True)

    def draw_story(self):
        """Short cinematic briefing before gameplay; no graphics knowledge required."""
        self.draw_background()
        cx = WIDTH // 2
        # A simple orbital command display behind the narrative text.
        for r, alpha_col in ((175, (18, 51, 91)), (135, (23, 75, 119)), (90, (27, 91, 132))):
            pygame.draw.ellipse(SCREEN, alpha_col, (cx-r, 210-r//2, r*2, r), 1)
        glow(SCREEN, (cx, 215), 45, CYAN, 20)
        pygame.draw.polygon(SCREEN, (14, 37, 65), [(cx, 160), (cx-27, 225), (cx, 212), (cx+27, 225)])
        pygame.draw.polygon(SCREEN, CYAN, [(cx, 160), (cx-27, 225), (cx, 212), (cx+27, 225)], 2)
        panel(SCREEN, (145, 330, 990, 250), (5, 12, 30, 232), (35, 83, 125))
        pages = [
            ("YEAR 2189 // THE SILENCE", "Earth's orbital defence network, AETHER, kept the planet safe for decades.",
             "Then every satellite went dark at the same time.", "The first signal came from beyond the Moon."),
            ("THE VOID HAS ARRIVED", "Unknown ships are moving toward Earth's orbital stations.",
             "A hostile intelligence called the VOID has seized the defence grid.", "Without the grid, Earth has no second line of defence."),
            ("YOU ARE THE LAST PILOT", "The emergency interface can read hand movement through a camera.",
             "Your gestures will command plasma fire, shields, chain lightning and targeting.", "Hold the line. Protect the station. Stop the VOID CORE."),
        ]
        title, line1, line2, line3 = pages[self.story_page]
        text(SCREEN, "AETHER // MISSION BRIEFING", (cx, 70), SMALL, CYAN, True)
        text(SCREEN, title, (cx, 370), MED, WHITE, True)
        text(SCREEN, line1, (cx, 424), FONT, MUTED, True)
        text(SCREEN, line2, (cx, 458), FONT, WHITE, True)
        text(SCREEN, line3, (cx, 492), FONT, CYAN, True)
        # Story progress indicator.
        for i in range(3):
            pygame.draw.circle(SCREEN, CYAN if i == self.story_page else (50, 69, 94), (cx-25+i*25, 535), 5)
        prompt = "PRESS ENTER / SPACE TO CONTINUE" if self.story_page < 2 else "PRESS ENTER / SPACE TO DEPLOY"
        text(SCREEN, prompt, (cx, 615), FONT, GREEN, True)
        text(SCREEN, "Q / ESC  //  ABORT BRIEFING", (cx, 650), SMALL, MUTED, True)

    def draw_leaderboard(self):
        self.draw_background()
        cx = WIDTH // 2
        text(SCREEN, "AETHER // FASTEST PILOTS", (cx, 58), BIG, CYAN, True)
        text(SCREEN, "RANK: WIN STATUS  >  FEWER MISSES  >  FASTER TIME  >  SCORE", (cx, 105), SMALL, MUTED, True)
        panel(SCREEN, (170, 135, 940, 490), (5, 12, 30, 235), (35, 83, 125))
        if not self.leaderboard:
            text(SCREEN, "Leaderboard unavailable or not configured yet.", (cx, 230), FONT, GOLD, True)
            text(SCREEN, "Check the Google Apps Script URL and internet connection.", (cx, 265), SMALL, MUTED, True)
        else:
            text(SCREEN, "RANK", (210, 158), SMALL, CYAN)
            text(SCREEN, "PLAYER", (300, 158), SMALL, CYAN)
            text(SCREEN, "TIME", (690, 158), SMALL, CYAN)
            text(SCREEN, "MISSED", (810, 158), SMALL, CYAN)
            text(SCREEN, "SCORE", (920, 158), SMALL, CYAN)
            text(SCREEN, "RESULT", (1050, 158), SMALL, CYAN)
            for i, row in enumerate(self.leaderboard[:20]):
                y = 185 + i*20
                col = GOLD if i < 3 else WHITE
                text(SCREEN, f"{i+1:02d}", (210, y), SMALL, col)
                text(SCREEN, str(row.get("player_name", "Pilot"))[:28], (300, y), SMALL, col)
                try:
                    secs = float(row.get("duration_seconds", 0))
                    time_text = f"{int(secs // 60):02d}:{secs % 60:05.2f}"
                except (TypeError, ValueError):
                    time_text = "—"
                text(SCREEN, time_text, (690, y), SMALL, col)
                text(SCREEN, str(row.get("missed_targets", "—")), (810, y), SMALL, col)
                text(SCREEN, str(row.get("score", 0)), (920, y), SMALL, col)
                won = str(row.get("boss_defeated", "")).lower() in ("true", "yes", "1")
                text(SCREEN, "WIN" if won else f"LVL {row.get('level_reached', 1)}", (1050, y), SMALL, GREEN if won else MUTED)
        text(SCREEN, "L // REFRESH     •     ENTER // RETURN TO TITLE     •     Q // QUIT", (cx, 665), SMALL, GREEN, True)

    def draw_overlay(self, title, subtitle, accent):
        veil=pygame.Surface((WIDTH,HEIGHT),pygame.SRCALPHA); veil.fill((3,5,18,175)); SCREEN.blit(veil,(0,0))
        glow(SCREEN,(WIDTH//2,HEIGHT//2-70),75,accent,20)
        text(SCREEN,title,(WIDTH//2,HEIGHT//2-75),BIG,accent,True)
        text(SCREEN,subtitle,(WIDTH//2,HEIGHT//2-22),FONT,WHITE,True)
        text(SCREEN,"PRESS ENTER TO PLAY AGAIN  •  Q TO QUIT",(WIDTH//2,HEIGHT//2+35),SMALL,MUTED,True)

    def draw(self):
        self.draw_background()
        if self.state=="TITLE": self.draw_title()
        elif self.state=="REGISTER": self.draw_registration()
        elif self.state=="STORY": self.draw_story()
        elif self.state=="LEADERBOARD": self.draw_leaderboard()
        else:
            for e in self.enemies: e.draw(SCREEN,self.t)
            for s in self.shots: s.draw(SCREEN)
            for s in self.enemy_shots:
                glow(SCREEN,(s[0],s[1]),8,PINK,30); pygame.draw.circle(SCREEN,PINK,(int(s[0]),int(s[1])),5)
            self.draw_ship(); self.draw_boss()
            for p in self.particles: p.draw(SCREEN)
            pygame.draw.line(SCREEN,(35,63,93),(0,HEIGHT-75),(WIDTH,HEIGHT-75),1)
            self.draw_hud(); self.draw_camera_panel()
            if self.wave_until>self.t:
                text(SCREEN,self.wave_label,(WIDTH//2,HEIGHT//2-80),MED,CYAN,True)
            if self.toast_until>self.t:
                panel(SCREEN,(WIDTH//2-190,HEIGHT-130,380,40),(8,16,36,220),(36,91,128))
                text(SCREEN,self.toast,(WIDTH//2,HEIGHT-110),SMALL,WHITE,True)
            if self.state=="GAMEOVER":
                self.draw_overlay("SIGNAL LOST",self.game_over_reason,RED)
                text(SCREEN, f"SURVIVAL TIME  {self.run_duration_seconds:.2f}s  //  MISSED {self.missed_targets}/7", (WIDTH//2,HEIGHT//2+48), SMALL, WHITE, True)
                if self.result_message: text(SCREEN,self.result_message,(WIDTH//2,HEIGHT//2+78),SMALL,self.result_color,True)
            elif self.state=="VICTORY":
                self.draw_overlay("SECTOR SECURED","VOID CORE NEUTRALISED  //  +250 BONUS",GREEN)
                text(SCREEN, f"COMPLETION TIME  {self.run_duration_seconds:.2f}s", (WIDTH//2,HEIGHT//2+48), SMALL, WHITE, True)
                if self.result_message: text(SCREEN,self.result_message,(WIDTH//2,HEIGHT//2+78),SMALL,self.result_color,True)
        pygame.display.flip()

    def close(self):
        if self.camera is not None: self.camera.release()
        if self.detector is not None: self.detector.close()
        pygame.quit()


def main():
    if not HAND_PATH.exists(): raise SystemExit(f"Missing hand landmark model: {HAND_PATH}")
    model = joblib.load(MODEL_PATH) if MODEL_PATH.exists() else None
    if model is None:
        print("Gesture model not found; running with keyboard controls (1=fist, 2=palm, 3=peace, 4=point).")
    detector=create_hand_landmarker(str(HAND_PATH))
    music_sound = start_background_music()
    camera_index = getattr(cfg, "CAMERA_INDEX", 0)
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    camera_ok = False
    for _ in range(10):
        camera_ok, test_frame = camera.read()
        if camera_ok and test_frame is not None:
            break
    if not camera_ok:
        camera.release()
        detector.close()
        pygame.quit()
        raise SystemExit("Webcam opened but no frames received. Check camera access.")
    game=AetherGame(model,detector,camera); running=True; last=time.monotonic()
    try:
        while running:
            now=time.monotonic(); dt=min(.04,now-last); last=now
            for event in pygame.event.get():
                if event.type==pygame.QUIT: running=False
                elif event.type == pygame.TEXTINPUT and game.state == "REGISTER":
                    if game.name_edit == "name" and len(game.player_name) < 40 and event.text.isprintable():
                        game.player_name += event.text
                    elif game.name_edit == "code" and len(game.participant_code) < 32 and event.text.isprintable():
                        game.participant_code += event.text
                elif event.type == pygame.MOUSEBUTTONDOWN and game.state == "REGISTER":
                    mx, my = event.pos
                    cx = WIDTH // 2
                    if cx-260 <= mx <= cx+260 and 260 <= my <= 308: game.name_edit = "name"
                    elif cx-260 <= mx <= cx+260 and 355 <= my <= 403: game.name_edit = "code"
                elif event.type==pygame.KEYDOWN:
                    if event.key in (pygame.K_q,pygame.K_ESCAPE): running=False
                    elif event.key in (pygame.K_RETURN, pygame.K_SPACE):
                        if game.state == "TITLE":
                            game.state = "REGISTER"
                        elif game.state == "REGISTER":
                            if len(game.player_name.strip()) < 2 or len(game.participant_code.strip()) < 3:
                                game.registration_error = "Enter a name (2+ chars) and participant code (3+ chars)."
                            else:
                                game.registration_error = ""; game.state = "STORY"; game.story_page = 0
                        elif game.state == "STORY":
                            if game.story_page < 2: game.story_page += 1
                            else: game.reset()
                        elif game.state in ("GAMEOVER", "VICTORY"):
                            game.leaderboard = fetch_leaderboard()
                            game.state = "LEADERBOARD"
                        elif game.state == "LEADERBOARD":
                            game.state = "TITLE"
                    elif event.key == pygame.K_TAB and game.state == "REGISTER":
                        game.name_edit = "code" if game.name_edit == "name" else "name"
                    elif event.key == pygame.K_BACKSPACE and game.state == "REGISTER":
                        if game.name_edit == "name": game.player_name = game.player_name[:-1]
                        else: game.participant_code = game.participant_code[:-1]
                    elif event.key == pygame.K_l:
                        if game.state == "GAMEOVER" or game.state == "VICTORY":
                            game.leaderboard = fetch_leaderboard(); game.state = "LEADERBOARD"
                        elif game.state == "LEADERBOARD":
                            game.leaderboard = fetch_leaderboard()
                    elif event.key==pygame.K_d: game.debug=not game.debug
                    elif event.key in KEYS:
                        game.simulated=KEYS[event.key]; game.history.clear(); game.history.append(game.simulated); game.candidate=game.simulated; game.candidate_since=game.t-1
                    elif event.key==pygame.K_0: game.simulated=None
            game.read_camera(dt)
            game.update(dt)
            game.draw()
            CLOCK.tick(FPS)
    finally: game.close()

if __name__=="__main__": main()
