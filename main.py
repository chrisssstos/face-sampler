"""
Gesture Controller — Hand & face gesture tracking → Ableton via OSC + MIDI.
Camera window + control panel. All audio handled in Ableton.
"""
import os
import sys
import time
import threading

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    HandLandmarker, HandLandmarkerOptions,
    FaceLandmarker, FaceLandmarkerOptions,
    RunningMode,
)

from PyQt6.QtWidgets import QApplication
from panel import SamplerPanel

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
HAND_MODEL = os.path.join(SCRIPT_DIR, "hand_landmarker.task")
FACE_MODEL = os.path.join(SCRIPT_DIR, "face_landmarker.task")

HAND_GESTURES = ["open_palm", "fist", "peace", "thumb_up", "pointing"]
FACE_GESTURES = ["mouth_open", "left_eyebrow", "right_eyebrow"]
ALL_GESTURES = HAND_GESTURES + FACE_GESTURES

LABELS = {
    "open_palm": "Open Palm", "fist": "Fist", "peace": "Peace",
    "thumb_up": "Thumbs Up", "pointing": "Point",
    "mouth_open": "Mouth Open", "left_eyebrow": "L Eyebrow", "right_eyebrow": "R Eyebrow",
}
COLORS = {
    "open_palm": (0, 220, 80), "fist": (60, 80, 255), "peace": (0, 220, 220),
    "thumb_up": (220, 80, 255), "pointing": (255, 200, 0),
    "mouth_open": (0, 140, 255), "left_eyebrow": (100, 255, 100),
    "right_eyebrow": (100, 100, 255), "unknown": (100, 100, 100),
}

HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17),
]
FACE_OVAL = [10,338,297,332,284,251,389,356,454,323,361,288,397,365,379,378,400,377,152,148,176,149,150,136,172,58,132,93,234,127,162,21,54,103,67,109,10]
L_EYE = [362,382,381,380,374,373,390,249,263,466,388,387,386,385,384,398,362]
R_EYE = [33,7,163,144,145,153,154,155,133,173,157,158,159,160,161,246,33]
L_BROW = [276,283,282,295,285,300,293,334,296,336]
R_BROW = [46,53,52,65,55,70,63,105,66,107]
LIPS_O = [61,146,91,181,84,17,314,405,321,375,291,409,270,269,267,0,37,39,40,185,61]
LIPS_I = [78,95,88,178,87,14,317,402,318,324,308,415,310,311,312,13,82,81,80,191,78]
NOSE = [168,6,197,195,5,4,1,19,94,2]

MT, MB = 13, 14
LB, LET = 70, 159
RB, RET = 300, 386


class Cal:
    def __init__(self):
        self.neutral_mouth = self.neutral_lb = self.neutral_rb = None
        self.calibrated = False
        self.hand_calibrated = False
        self.finger_margin = 0.0

    def do_face(self, lm):
        self.neutral_mouth = abs(lm[MB].y - lm[MT].y)
        self.neutral_lb = abs(lm[LET].y - lm[LB].y)
        self.neutral_rb = abs(lm[RET].y - lm[RB].y)
        self.calibrated = True

    def do_hand(self, lm):
        diffs = [(lm[p].y - lm[t].y) for t, p in [(8,6),(12,10),(16,14),(20,18)]]
        self.finger_margin = sorted(diffs)[1]
        self.hand_calibrated = True


def run_calibration(cap, hlm, flm, cal, start_ts=9000):
    CAM_WIN = "Gesture Controller"
    t0 = time.time()
    cd, dur = 2.0, 3.0
    ts = [start_ts]

    while True:
        ret, f = cap.read()
        if not ret: break
        f = cv2.flip(f, 1)
        h, w, _ = f.shape
        el = time.time() - t0
        f = (f * 0.3).astype(np.uint8)
        if el < cd:
            sec = int(cd - el) + 1
            cv2.putText(f, "CALIBRATION", (w//2 - 200, h//2 - 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 180, 0), 3)
            cv2.putText(f, "Neutral face + relaxed hand in view",
                        (w//2 - 280, h//2 - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 225), 2)
            cv2.putText(f, str(sec), (w//2 - 30, h//2 + 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 4.0, (0, 210, 90), 6)
        elif el < cd + dur:
            p = (el - cd) / dur
            bw, bx = 500, w//2 - 250
            cv2.putText(f, "Calibrating...", (w//2 - 150, h//2 - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 210, 90), 3)
            cv2.rectangle(f, (bx, h//2+30), (bx+bw, h//2+60), (50, 50, 60), 2)
            cv2.rectangle(f, (bx+2, h//2+32), (bx+int(bw*p), h//2+58), (0, 210, 90), -1)
            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts[0] += 33
            hlm.detect_async(mi, ts[0])
            flm.detect_async(mi, ts[0])
        else:
            break
        cv2.imshow(CAM_WIN, f)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            return False, ts[0]

    ret, f = cap.read()
    if ret:
        f = cv2.flip(f, 1)
        rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        hc = HandLandmarker.create_from_options(HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=RunningMode.IMAGE, num_hands=1,
            min_hand_detection_confidence=0.5))
        fc = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=FACE_MODEL),
            running_mode=RunningMode.IMAGE, num_faces=1,
            min_face_detection_confidence=0.5))
        mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        hr, fr = hc.detect(mi), fc.detect(mi)
        if fr.face_landmarks: cal.do_face(fr.face_landmarks[0])
        if hr.hand_landmarks: cal.do_hand(hr.hand_landmarks[0])
        hc.close(); fc.close()
        print(f"  Cal: face={'OK' if cal.calibrated else 'miss'} "
              f"hand={'OK' if cal.hand_calibrated else 'miss'}")

    for _ in range(15):
        ret, f = cap.read()
        if not ret: break
        f = cv2.flip(f, 1)
        f = (f * 0.3).astype(np.uint8)
        h, w, _ = f.shape
        st = "DONE" if cal.calibrated else "USING DEFAULTS"
        cv2.putText(f, f"Calibration: {st}", (w//2 - 240, h//2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 210, 90), 3)
        cv2.imshow(CAM_WIN, f)
        cv2.waitKey(33)

    return True, ts[0]


def _lm_to_array(lm):
    return np.array([(l.x, l.y, l.z) for l in lm], dtype=np.float32)


def _normalize_landmarks(arr):
    wrist = arr[0].copy()
    norm = arr - wrist
    scale = np.linalg.norm(norm[9]) if np.linalg.norm(norm[9]) > 0.001 else 1.0
    return norm / scale


def match_custom_gesture(lm, captured_gestures, threshold=0.35):
    if not captured_gestures:
        return None
    current = _normalize_landmarks(_lm_to_array(lm))
    best_name, best_dist = None, threshold
    for name, ref_arr in captured_gestures.items():
        dist = np.mean(np.linalg.norm(current - ref_arr, axis=1))
        if dist < best_dist:
            best_dist = dist
            best_name = name
    return best_name


def classify_hand(lm, label, cal, ui=None):
    if ui and ui.captured_gestures:
        custom = match_custom_gesture(lm, ui.captured_gestures)
        if custom:
            return custom

    m = ui.hand_margin if ui else (cal.finger_margin if cal.hand_calibrated else 0.0)
    def up(t, p): return (lm[p].y - lm[t].y) > m
    tt, ti, tm_ = lm[4], lm[3], lm[2]
    th = (tt.x < ti.x and tt.x < tm_.x) if label == "Right" else (tt.x > ti.x and tt.x > tm_.x)
    i, mi, r, p = up(8,6), up(12,10), up(16,14), up(20,18)
    c = sum([i, mi, r, p])
    if th and c == 0: return "thumb_up"
    if not th and c == 0: return "fist"
    if i and c == 1 and not th: return "pointing"
    if i and mi and not r and not p and c == 2: return "peace"
    if th and c >= 3: return "open_palm"
    return "unknown"


def classify_face(lm, cal, ui):
    active = []
    mouth = abs(lm[MB].y - lm[MT].y)
    lb = abs(lm[LET].y - lm[LB].y)
    rb = abs(lm[RET].y - lm[RB].y)
    ui.live_face = {"mouth": mouth, "left_brow": lb, "right_brow": rb}
    if mouth > ui.get_threshold("mouth_open"): active.append("mouth_open")
    if lb > ui.get_threshold("left_eyebrow"): active.append("left_eyebrow")
    if rb > ui.get_threshold("right_eyebrow"): active.append("right_eyebrow")
    return active


def draw_hand(f, lm, col, h, w):
    pts = [(int(l.x*w), int(l.y*h)) for l in lm]
    for p in pts: cv2.circle(f, p, 3, col, -1)
    for a, b in HAND_CONN: cv2.line(f, pts[a], pts[b], col, 2)

def _cont(f, lm, idx, col, h, w, t=1):
    pts = [(int(lm[i].x*w), int(lm[i].y*h)) for i in idx if i < len(lm)]
    for i in range(len(pts)-1): cv2.line(f, pts[i], pts[i+1], col, t)

def draw_face(f, lm, active, h, w):
    b = (70, 70, 80)
    _cont(f, lm, FACE_OVAL, b, h, w)
    _cont(f, lm, L_EYE, (150,150,160), h, w)
    _cont(f, lm, R_EYE, (150,150,160), h, w)
    _cont(f, lm, NOSE, b, h, w)
    lbc = COLORS["left_eyebrow"] if "left_eyebrow" in active else (90,90,100)
    rbc = COLORS["right_eyebrow"] if "right_eyebrow" in active else (90,90,100)
    _cont(f, lm, L_BROW, lbc, h, w, 3 if "left_eyebrow" in active else 1)
    _cont(f, lm, R_BROW, rbc, h, w, 3 if "right_eyebrow" in active else 1)
    mc = COLORS["mouth_open"] if "mouth_open" in active else (90,90,100)
    mt = 3 if "mouth_open" in active else 1
    _cont(f, lm, LIPS_O, mc, h, w, mt)
    _cont(f, lm, LIPS_I, mc, h, w, mt)
    if active:
        fh_ = lm[10]
        fx, fy = int(fh_.x*w), int(fh_.y*h)
        yo = fy - 30
        for g in active:
            cv2.putText(f, LABELS.get(g, g).upper(), (fx-60, yo),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        COLORS.get(g, (255,255,255)), 2)
            yo -= 25


def main():
    app = QApplication(sys.argv)

    modes = {g: "shot" for g in ALL_GESTURES}
    cal = Cal()

    ui = SamplerPanel(ALL_GESTURES, LABELS, COLORS, modes, cal, FACE_GESTURES)

    # MediaPipe
    hs, fs = {"d": None}, {"d": None}
    hl, fl = threading.Lock(), threading.Lock()
    def oh(r, i, t):
        with hl: hs["d"] = r
    def of(r, i, t):
        with fl: fs["d"] = r

    hlm = HandLandmarker.create_from_options(HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=HAND_MODEL),
        running_mode=RunningMode.LIVE_STREAM, num_hands=2,
        min_hand_detection_confidence=0.7, min_hand_presence_confidence=0.6,
        min_tracking_confidence=0.6, result_callback=oh))
    flm = FaceLandmarker.create_from_options(FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=FACE_MODEL),
        running_mode=RunningMode.LIVE_STREAM, num_faces=1,
        min_face_detection_confidence=0.6, min_face_presence_confidence=0.6,
        min_tracking_confidence=0.5, result_callback=of))

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open camera."); return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    CAM_WIN = "Gesture Controller"
    cv2.namedWindow(CAM_WIN, cv2.WINDOW_AUTOSIZE)

    fts = 9000
    t0_ms = int(time.time() * 1000)

    prev_hand, prev_face = set(), set()
    gate_release_timers = {}  # gk → time when gesture left, for debouncing
    GATE_DEBOUNCE = 0.15  # seconds — ignore flickers shorter than this
    last_ui_draw = 0.0
    UI_INTERVAL = 1.0 / 15

    print("Running! Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        frame = cv2.flip(frame, 1)
        fh, fw, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mi = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        fts = int(time.time() * 1000) - t0_ms + 10000
        hlm.detect_async(mi, fts)
        if ui.face_enabled:
            flm.detect_async(mi, fts)

        now = time.time()

        # --- Gesture Capture ---
        if ui.capture_mode:
            elapsed = now - ui.capture_countdown
            with hl: hr_cap = hs["d"]
            if elapsed >= 3.0 and hr_cap and hr_cap.hand_landmarks:
                cap_lm = hr_cap.hand_landmarks[0]
                arr = _normalize_landmarks(_lm_to_array(cap_lm))
                slot_idx = ui.capture_slot
                g = ui.gestures[slot_idx]
                ui.captured_gestures[g] = arr
                ui.capture_mode = False
                print(f"Captured gesture for slot {slot_idx + 1} ({g})")

        # --- Hands ---
        cur_h = set()
        with hl: hr = hs["d"]
        if hr and hr.hand_landmarks:
            for i, hls in enumerate(hr.hand_landmarks):
                lab = "Right"
                if hr.handedness and i < len(hr.handedness):
                    lab = hr.handedness[i][0].category_name
                g = classify_hand(hls, lab, cal, ui)
                col = COLORS.get(g, (100,100,100))
                draw_hand(frame, hls, col, fh, fw)

                wx, wy = int(hls[0].x*fw), int(hls[0].y*fh)
                mode = modes.get(g, "shot")
                cv2.putText(frame, LABELS.get(g, g).upper(), (wx-60, wy-30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)

                gk = f"hand_{lab}_{g}"
                if g != "unknown":
                    cur_h.add(gk)
                    onset = gk not in prev_hand

                    if mode == "shot":
                        if onset:
                            ui.osc_fire(g)
                            ui.midi_fire(g)
                            ui.trigger_times[g] = now
                            cv2.circle(frame, (wx, wy-50), 12, col, -1)
                    elif mode == "gate":
                        if onset:
                            ui.osc_fire(g)
                            ui.midi_fire(g)
                            ui.trigger_times[g] = now
                            cv2.circle(frame, (wx, wy-50), 12, col, -1)
                    elif mode == "toggle":
                        if onset:
                            playing = ui.toggle_states.get(gk, False)
                            if playing:
                                ui.osc_stop(g)
                                ui.midi_stop(g)
                                ui.toggle_states[gk] = False
                            else:
                                ui.osc_fire(g)
                                ui.midi_fire(g)
                                ui.toggle_states[gk] = True
                                ui.trigger_times[g] = now
                                cv2.circle(frame, (wx, wy-50), 12, col, -1)
                    elif mode == "loop":
                        if onset:
                            ui.osc_fire(g)
                            ui.midi_fire(g)
                            ui.trigger_times[g] = now

        # Gate: debounced stop on release
        for gk in prev_hand - cur_h:
            g_name = gk.split("_", 2)[-1]
            if modes.get(g_name, "shot") == "gate" and gk not in gate_release_timers:
                gate_release_timers[gk] = now
        # Cancel timers for gestures that came back
        for gk in cur_h:
            gate_release_timers.pop(gk, None)
        # Fire stops for gestures that stayed gone past debounce
        expired = [gk for gk, t in gate_release_timers.items() if now - t >= GATE_DEBOUNCE]
        for gk in expired:
            g_name = gk.split("_", 2)[-1]
            ui.osc_stop(g_name)
            ui.midi_stop(g_name)
            del gate_release_timers[gk]
        prev_hand = cur_h

        # --- Face ---
        cur_f = set()
        if ui.face_enabled:
            with fl: fr = fs["d"]
            if fr and fr.face_landmarks:
                for fls in fr.face_landmarks:
                    af = classify_face(fls, cal, ui)
                    draw_face(frame, fls, af, fh, fw)
                    for g in af:
                        mode = modes.get(g, "shot")
                        gk = f"face_{g}"
                        cur_f.add(gk)
                        onset = gk not in prev_face

                        if mode == "shot":
                            if onset:
                                ui.osc_fire(g)
                                ui.midi_fire(g)
                                ui.trigger_times[g] = now
                        elif mode == "gate":
                            if onset:
                                ui.osc_fire(g)
                                ui.midi_fire(g)
                                ui.trigger_times[g] = now
                        elif mode == "toggle":
                            if onset:
                                playing = ui.toggle_states.get(gk, False)
                                if playing:
                                    ui.osc_stop(g)
                                    ui.midi_stop(g)
                                    ui.toggle_states[gk] = False
                                else:
                                    ui.osc_fire(g)
                                    ui.midi_fire(g)
                                    ui.toggle_states[gk] = True
                                    ui.trigger_times[g] = now
                        elif mode == "loop":
                            if onset:
                                ui.osc_fire(g)
                                ui.midi_fire(g)
                                ui.trigger_times[g] = now

        # Face gate: debounced stop on release
        for gk in prev_face - cur_f:
            g_name = gk.split("_", 1)[-1]
            if modes.get(g_name, "shot") == "gate" and gk not in gate_release_timers:
                gate_release_timers[gk] = now
        for gk in cur_f:
            gate_release_timers.pop(gk, None)
        expired_f = [gk for gk, t in gate_release_timers.items()
                     if gk.startswith("face_") and now - t >= GATE_DEBOUNCE]
        for gk in expired_f:
            g_name = gk.split("_", 1)[-1]
            ui.osc_stop(g_name)
            ui.midi_stop(g_name)
            del gate_release_timers[gk]
        prev_face = cur_f

        # --- Capture overlay ---
        if ui.capture_mode:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (fw, 50), (0, 0, 80), -1)
            elapsed_cap = now - ui.capture_countdown
            slot_label = ui.labels.get(ui.gestures[ui.capture_slot], "?")
            if elapsed_cap < 3.0:
                sec = 3 - int(elapsed_cap)
                cv2.putText(overlay, f"CAPTURE [{slot_label}] in {sec}...",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 100, 255), 2)
            else:
                cv2.putText(overlay, f"SHOW GESTURE for [{slot_label}]",
                            (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                t = int(abs(now * 4 % 2 - 1) * 255)
                cv2.rectangle(overlay, (2, 2), (fw-2, fh-2), (0, 0, t), 3)
            frame = overlay

        cv2.imshow(CAM_WIN, frame)

        if now - last_ui_draw >= UI_INTERVAL:
            ui.update()
            last_ui_draw = now

        # Auto-cal request
        if ui.auto_cal_requested:
            ui.auto_cal_requested = False
            prev_hand, prev_face = set(), set()
            ok, fts = run_calibration(cap, hlm, flm, cal, fts + 1000)
            if not ok: break

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    hlm.close(); flm.close()
    ui.cleanup()


if __name__ == "__main__":
    main()
