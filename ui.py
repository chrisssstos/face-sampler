"""
Interactive sampler UI — own OpenCV window.
Scrollable, per-gesture threshold tuning, sample trim editor.
"""
import cv2
import numpy as np
import time

# Theme
BG        = (22, 22, 28)
PANEL     = (30, 30, 38)
BORDER    = (50, 50, 60)
TEXT      = (220, 220, 225)
DIM       = (100, 100, 110)
ACCENT    = (0, 180, 255)
GREEN     = (0, 210, 90)
GREEN_HI  = (0, 255, 120)
ORANGE    = (0, 140, 255)
SLOT_BG   = (35, 35, 45)
SLOT_HOVER= (45, 45, 58)
SLOT_ACT  = (30, 70, 30)
SEL_COL   = (0, 180, 255)
LOOP_COL  = (0, 140, 255)
SHOT_COL  = (180, 180, 190)
SL_BG     = (45, 45, 55)
SL_FILL   = (0, 200, 100)
SL_KNOB   = (220, 220, 230)
BTN_BG    = (50, 50, 62)
BTN_HV    = (65, 65, 80)
HDR_BG    = (26, 26, 34)
TRIM_COL  = (0, 200, 255)
SECTION_TX = (80, 80, 90)
SCROLL_COL = (120, 120, 140)
SCROLL_BD  = (160, 160, 175)

PW = 560

# Consistent button height used across the UI
BTN_H = 26


class R:
    __slots__ = ("x", "y", "w", "h")
    def __init__(self, x, y, w, h):
        self.x, self.y, self.w, self.h = x, y, w, h
    def hit(self, mx, my):
        return self.x <= mx <= self.x + self.w and self.y <= my <= self.y + self.h
    def lerp(self, mx):
        return max(0.0, min(1.0, (mx - self.x) / max(1, self.w)))


class SamplerUI:
    WIN = "Sampler"

    def __init__(self, gestures, labels, colors, mapping, modes,
                 sampler, clock, cal, face_gestures):
        self.gestures = gestures
        self.labels = labels
        self.colors = colors
        self.mapping = mapping
        self.modes = modes
        self.sampler = sampler
        self.clock = clock
        self.cal = cal
        self.face_gestures = face_gestures

        self.face_enabled = True
        self.selected = 0
        self.available = sampler.get_sample_names() + ["none"]

        self.thresholds = {
            "mouth_open": 0.06,
            "left_eyebrow": 0.04,
            "right_eyebrow": 0.04,
        }
        self.hand_margin = 0.0
        self.cal_mode = False
        self.auto_cal_requested = False
        self.live_face = {"mouth": 0.0, "left_brow": 0.0, "right_brow": 0.0}
        self.trigger_times = {}

        self._mx = self._my = 0
        self._down = False
        self._drag = None
        self._hover = -1
        self._reg = {}
        self._scroll_y = 0.0
        self._max_scroll = 0
        self._wf_x = 12
        self._wf_w = 100

        cv2.namedWindow(self.WIN, cv2.WINDOW_AUTOSIZE)
        cv2.setMouseCallback(self.WIN, self._mouse_cb)

    def _mouse_cb(self, ev, x, y, flags, _):
        if ev == cv2.EVENT_MOUSEWHEEL:
            # On macOS trackpad, flags encodes scroll amount directly
            # Positive = scroll up (decrease scroll_y), Negative = scroll down
            if flags > 0:
                self._scroll_y = max(0, self._scroll_y - 40)
            else:
                self._scroll_y = min(self._max_scroll, self._scroll_y + 40)
            return

        yc = y + int(self._scroll_y)
        self._mx, self._my = x, yc

        if ev == cv2.EVENT_LBUTTONDOWN:
            self._down = True
            self._click(x, yc)
        elif ev == cv2.EVENT_LBUTTONUP:
            self._down = False
            self._drag = None
        elif ev == cv2.EVENT_MOUSEMOVE:
            if self._down and self._drag:
                self._do_drag(x, yc)
            self._hover = -1
            for i in range(len(self.gestures)):
                r = self._reg.get(f"slot_{i}")
                if r and r.hit(x, yc):
                    self._hover = i
                    break

    def _click(self, x, y):
        for name, rect in self._reg.items():
            if not rect.hit(x, y):
                continue
            parts = name.split("_", 1)
            kind = parts[0]
            idx = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else -1

            if kind in ("sample", "samplefwd") and idx >= 0:
                g = self.gestures[idx]
                cur = self.mapping.get(g, "none")
                ci = self.available.index(cur) if cur in self.available else len(self.available) - 1
                self.mapping[g] = self.available[(ci + 1) % len(self.available)]
                self.selected = idx
            elif kind == "samplerev" and idx >= 0:
                g = self.gestures[idx]
                cur = self.mapping.get(g, "none")
                ci = self.available.index(cur) if cur in self.available else 0
                self.mapping[g] = self.available[(ci - 1) % len(self.available)]
                self.selected = idx
            elif kind == "mode" and idx >= 0:
                g = self.gestures[idx]
                self.modes[g] = "loop" if self.modes[g] == "shot" else "shot"
                if self.modes[g] == "shot":
                    for gk in list(self.clock.get_active_loops()):
                        # Match keys like "hand_Right_open_palm" or "face_mouth_open"
                        # Use endswith to avoid false substring matches
                        if gk.endswith("_" + g) or gk == g:
                            self.clock.stop_loop(gk)
            elif kind == "vol" and idx >= 0:
                g = self.gestures[idx]
                sn = self.mapping.get(g, "none")
                if sn != "none" and sn in self.sampler.samples:
                    self.sampler.samples[sn].volume = rect.lerp(x)
                    self._drag = (idx, "vol")
                self.selected = idx
            elif kind == "sens" and idx >= 0:
                g = self.gestures[idx]
                if g in self.thresholds:
                    self.thresholds[g] = 0.005 + rect.lerp(x) * 0.095
                self._drag = (idx, "sens")
                self.selected = idx
            elif kind == "handmargin":
                self.hand_margin = rect.lerp(x) * 0.08
                self._drag = (-1, "handmargin")
            elif kind == "slot" and idx >= 0:
                self.selected = idx
            elif name == "bpm_up":
                self.clock.bpm = min(300, self.clock.bpm + 5)
            elif name == "bpm_down":
                self.clock.bpm = max(40, self.clock.bpm - 5)
            elif name == "quantize":
                self.clock.enabled = not self.clock.enabled
            elif name == "subdiv":
                subs = [1, 2, 4]
                si = subs.index(self.clock.subdivision) if self.clock.subdivision in subs else 0
                self.clock.subdivision = subs[(si + 1) % len(subs)]
            elif name == "face":
                self.face_enabled = not self.face_enabled
            elif name == "tune":
                self.cal_mode = not self.cal_mode
            elif name == "autocal":
                self.auto_cal_requested = True
            elif name == "reload":
                import os
                samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "samples")
                self.sampler.load_directory(samples_dir)
                self.available = self.sampler.get_sample_names() + ["none"]
                # Fix up mappings that reference deleted samples
                for gk in self.mapping:
                    if self.mapping[gk] not in self.available:
                        self.mapping[gk] = "none"
            elif name == "midi":
                self.sampler.midi_enabled = not self.sampler.midi_enabled
            elif kind == "preview" and idx >= 0:
                g = self.gestures[idx]
                sn = self.mapping.get(g, "none")
                if sn != "none" and sn in self.sampler.samples:
                    vol = self.sampler.samples[sn].volume
                    self.sampler.play(sn, vol)
            elif name == "trimstart_0":
                self._drag = (-1, "trimstart")
            elif name == "trimend_0":
                self._drag = (-1, "trimend")

    def _do_drag(self, x, y):
        if not self._drag:
            return
        idx, kind = self._drag
        if kind == "vol":
            r = self._reg.get(f"vol_{idx}")
            if r:
                sn = self.mapping.get(self.gestures[idx], "none")
                if sn != "none" and sn in self.sampler.samples:
                    self.sampler.samples[sn].volume = r.lerp(x)
        elif kind == "sens":
            r = self._reg.get(f"sens_{idx}")
            if r:
                g = self.gestures[idx]
                if g in self.thresholds:
                    self.thresholds[g] = 0.005 + r.lerp(x) * 0.095
        elif kind == "handmargin":
            r = self._reg.get("handmargin_0")
            if r:
                self.hand_margin = r.lerp(x) * 0.08
        elif kind == "trimstart":
            g = self.gestures[self.selected]
            sn = self.mapping.get(g, "none")
            if sn != "none" and sn in self.sampler.samples:
                s = self.sampler.samples[sn]
                frac = max(0.0, min(s.trim_end - 0.01, (x - self._wf_x) / max(1, self._wf_w)))
                s.trim_start = frac
        elif kind == "trimend":
            g = self.gestures[self.selected]
            sn = self.mapping.get(g, "none")
            if sn != "none" and sn in self.sampler.samples:
                s = self.sampler.samples[sn]
                frac = max(s.trim_start + 0.01, min(1.0, (x - self._wf_x) / max(1, self._wf_w)))
                s.trim_end = frac

    def get_threshold(self, gesture):
        return self.thresholds.get(gesture, 0.03)

    def _btn(self, c, x, y, w, h, text, name, tc=None):
        hv = R(x, y, w, h).hit(self._mx, self._my)
        bg = BTN_HV if hv else BTN_BG
        cv2.rectangle(c, (x, y), (x + w, y + h), bg, -1)
        cv2.rectangle(c, (x, y), (x + w, y + h), BORDER, 1)
        ts = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)[0]
        cv2.putText(c, text, (x + (w - ts[0]) // 2, y + (h + ts[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, tc or TEXT, 1)
        self._reg[name] = R(x, y, w, h)

    def _slider(self, c, name, x, y, w, val, color=SL_FILL):
        self._reg[name] = R(x, y, w, 14)
        cv2.rectangle(c, (x, y + 1), (x + w, y + 13), SL_BG, -1)
        fw = int(w * max(0.0, min(1.0, val)))
        if fw > 0:
            cv2.rectangle(c, (x, y + 1), (x + fw, y + 13), color, -1)
        cv2.circle(c, (x + fw, y + 7), 5, SL_KNOB, -1)

    def _section_label(self, c, y, text):
        """Draw a subtle section divider with label."""
        cv2.rectangle(c, (0, y), (PW, y + 22), (25, 25, 32), -1)
        cv2.putText(c, text, (16, y + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, SECTION_TX, 1)
        tw = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.32, 1)[0][0]
        cv2.line(c, (22 + tw, y + 11), (PW - 14, y + 11), SECTION_TX, 1)

    def show(self):
        """Render and display the panel in its own window."""
        now = time.time()
        active = {g for g, t in self.trigger_times.items() if now - t < 0.25}
        loops = self.clock.get_active_loops()
        slot_h = 96 if self.cal_mode else 76

        # Content height
        content_h = 114 + 26 + len(self.gestures) * slot_h + 8
        if self.cal_mode:
            content_h += 30
            g = self.gestures[self.selected]
            sn = self.mapping.get(g, "none")
            if sn != "none" and sn in self.sampler.samples:
                content_h += 180
        content_h += 70

        H = content_h
        c = np.full((H, PW, 3), BG, dtype=np.uint8)
        self._reg = {}

        # === HEADER ===
        cv2.rectangle(c, (0, 0), (PW, 58), HDR_BG, -1)
        cv2.putText(c, "GESTURE SAMPLER", (14, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, ACCENT, 2)
        cal_c = GREEN if self.cal.calibrated else ACCENT
        cv2.putText(c, "CALIBRATED" if self.cal.calibrated else "MANUAL",
                    (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.35, cal_c, 1)

        bx = PW - 190
        self._btn(c, bx, 8, 58, BTN_H, "FACE " + ("ON" if self.face_enabled else "OFF"),
                  "face", GREEN if self.face_enabled else DIM)
        self._btn(c, bx + 64, 8, 58, BTN_H, "TUNE " + ("ON" if self.cal_mode else "OFF"),
                  "tune", ACCENT if self.cal_mode else DIM)
        self._btn(c, bx + 128, 8, 58, BTN_H, "MIDI " + ("ON" if self.sampler.midi_enabled else "OFF"),
                  "midi", GREEN if self.sampler.midi_enabled else DIM)
        self._btn(c, bx + 64, 38, 122, BTN_H - 4, "AUTO CALIBRATE", "autocal", ACCENT)

        # Strong accent line separating header from content
        cv2.line(c, (0, 58), (PW, 58), ACCENT, 2)

        y = 62

        # === BPM BAR ===
        cv2.rectangle(c, (0, y), (PW, y + 46), PANEL, -1)
        cv2.line(c, (0, y + 46), (PW, y + 46), BORDER, 1)
        bpm_c = GREEN if self.clock.enabled else DIM
        cv2.putText(c, str(self.clock.bpm), (14, y + 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, bpm_c, 2)
        cv2.putText(c, "BPM", (75, y + 32), cv2.FONT_HERSHEY_SIMPLEX, 0.4, DIM, 1)
        self._btn(c, 110, y + 8, 32, BTN_H, "+", "bpm_up")
        self._btn(c, 148, y + 8, 32, BTN_H, "-", "bpm_down")
        subdiv_t = {1: "1/4", 2: "1/8", 4: "1/16"}.get(self.clock.subdivision, "?")
        self._btn(c, 192, y + 8, 52, BTN_H, subdiv_t, "subdiv")
        q_t = "SYNC" if self.clock.enabled else "FREE"
        self._btn(c, 252, y + 8, 58, BTN_H, q_t, "quantize",
                  GREEN if self.clock.enabled else ORANGE)

        for i in range(4):
            dc = GREEN if (self.clock.is_on_beat and self.clock._beat_count % 4 == i) else (40, 40, 50)
            cv2.circle(c, (336 + i * 26, y + 24), 8, dc, -1)
            cv2.circle(c, (336 + i * 26, y + 24), 8, BORDER, 1)

        midi_c = GREEN if self.sampler.midi_enabled else DIM
        cv2.putText(c, "MIDI: " + ("ON" if self.sampler.midi_enabled else "OFF"),
                    (450, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.38, midi_c, 1)

        y += 50

        # === SECTION LABEL ===
        self._section_label(c, y, "INSTRUMENTS")
        y += 26

        # === SLOTS ===
        for i, g in enumerate(self.gestures):
            sy = y + i * slot_h
            is_sel = (i == self.selected)
            is_hov = (i == self._hover)
            is_act = g in active
            is_loop = any(g in gk for gk in loops)
            sn = self.mapping.get(g, "none")
            mode = self.modes.get(g, "shot")
            col = self.colors.get(g, DIM)

            slot_top = sy + 2
            slot_bot = sy + slot_h - 4
            bg = SLOT_ACT if is_act else (SLOT_HOVER if is_hov else SLOT_BG)
            cv2.rectangle(c, (8, slot_top), (PW - 8, slot_bot), bg, -1)
            self._reg[f"slot_{i}"] = R(8, slot_top, PW - 16, slot_bot - slot_top)

            # Selected slot: bright left accent bar + subtle border
            if is_sel:
                cv2.rectangle(c, (8, slot_top), (13, slot_bot), ACCENT, -1)
                cv2.rectangle(c, (8, slot_top), (PW - 8, slot_bot), SEL_COL, 1)

            # Active slot: strong green left glow
            if is_act:
                cv2.rectangle(c, (8, slot_top), (14, slot_bot), GREEN_HI, -1)

            rx = 20
            cv2.putText(c, str(i + 1), (rx, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1)
            rx += 22
            cv2.circle(c, (rx + 4, sy + 16), 6, col, -1)
            rx += 18
            cv2.putText(c, self.labels.get(g, g), (rx, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, TEXT, 1)

            sx = 174
            sn_t = sn if sn != "none" else "---"
            self._reg[f"samplerev_{i}"] = R(sx, sy + 4, 18, BTN_H)
            cv2.putText(c, "<", (sx + 3, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1)
            self._reg[f"sample_{i}"] = R(sx + 20, sy + 4, 114, BTN_H)
            nbg = BTN_HV if R(sx + 20, sy + 4, 114, BTN_H).hit(self._mx, self._my) else BTN_BG
            cv2.rectangle(c, (sx + 20, sy + 4), (sx + 134, sy + 4 + BTN_H), nbg, -1)
            cv2.rectangle(c, (sx + 20, sy + 4), (sx + 134, sy + 4 + BTN_H), BORDER, 1)
            cv2.putText(c, sn_t, (sx + 26, sy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1)
            self._reg[f"samplefwd_{i}"] = R(sx + 136, sy + 4, 18, BTN_H)
            cv2.putText(c, ">", (sx + 138, sy + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, DIM, 1)

            mx_ = 326
            self._reg[f"mode_{i}"] = R(mx_, sy + 4, 54, BTN_H)
            mc = LOOP_COL if mode == "loop" else SHOT_COL
            mt = "LOOP" if mode == "loop" else "SHOT"
            mbg = BTN_HV if R(mx_, sy + 4, 54, BTN_H).hit(self._mx, self._my) else BTN_BG
            cv2.rectangle(c, (mx_, sy + 4), (mx_ + 54, sy + 4 + BTN_H), mbg, -1)
            cv2.rectangle(c, (mx_, sy + 4), (mx_ + 54, sy + 4 + BTN_H), BORDER, 1)
            cv2.putText(c, mt, (mx_ + 8, sy + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, mc, 1)

            # Preview button — consistent size matching other buttons
            self._btn(c, 388, sy + 4, 38, BTN_H, "PLAY", f"preview_{i}", GREEN)

            if is_loop:
                cv2.circle(c, (436, sy + 17), 6, LOOP_COL, -1)
                cv2.circle(c, (436, sy + 17), 6, BORDER, 1)

            # Trigger flash — prominent pulsing green dot
            if is_act:
                pulse = int(abs(now * 8 % 2 - 1) * 3)
                cv2.circle(c, (PW - 24, sy + 17), 8 + pulse, GREEN_HI, -1)
                cv2.circle(c, (PW - 24, sy + 17), 8 + pulse, (255, 255, 255), 1)
            elif is_loop:
                cv2.circle(c, (PW - 24, sy + 17), 5, LOOP_COL, -1)

            # VOL slider + waveform
            r2y = sy + 36
            cv2.putText(c, "VOL", (20, r2y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)
            vol = self.sampler.samples[sn].volume if sn != "none" and sn in self.sampler.samples else 0
            self._slider(c, f"vol_{i}", 48, r2y, 150, vol)
            cv2.putText(c, f"{int(vol * 100)}%", (204, r2y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)

            wfx, wfw = 244, max(1, PW - 260)
            if sn != "none" and sn in self.sampler.samples:
                wf = self.sampler.get_waveform(sn, wfw)
                cv2.rectangle(c, (wfx, r2y), (wfx + wfw, r2y + 16), (30, 30, 38), -1)
                mid = r2y + 8
                wc = col if is_act else (50, 55, 65)
                for j, a in enumerate(wf):
                    bh = int(a * 7)
                    if bh > 0:
                        cv2.line(c, (wfx + j, mid - bh), (wfx + j, mid + bh), wc, 1)

            # Threshold slider (tune mode, face gestures)
            if self.cal_mode and g in self.face_gestures:
                r3y = sy + 58
                cv2.putText(c, "THR", (20, r3y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)
                thresh = self.thresholds.get(g, 0.03)
                sn_ = (thresh - 0.005) / 0.095
                self._slider(c, f"sens_{i}", 48, r3y, 150, sn_, ACCENT)
                cv2.putText(c, f"{thresh:.3f}", (204, r3y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)

                fk = {"mouth_open": "mouth", "left_eyebrow": "left_brow",
                       "right_eyebrow": "right_brow"}.get(g)
                if fk:
                    live = self.live_face.get(fk, 0)
                    mx2, mw = 244, max(1, PW - 260)
                    cv2.rectangle(c, (mx2, r3y), (mx2 + mw, r3y + 16), (30, 30, 38), -1)
                    scale = max(0.001, thresh * 2.5)
                    bw = int(min(1.0, live / scale) * mw)
                    bc = GREEN if live > thresh else (50, 50, 60)
                    if bw > 0:
                        cv2.rectangle(c, (mx2, r3y + 2), (mx2 + bw, r3y + 14), bc, -1)
                    thx = mx2 + int(min(1.0, thresh / scale) * mw)
                    cv2.line(c, (thx, r3y), (thx, r3y + 16), (0, 0, 220), 2)
                    cv2.putText(c, "TH", (thx - 8, r3y - 2),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.25, (0, 0, 220), 1)

        y += len(self.gestures) * slot_h + 6

        # Hand margin slider (tune mode)
        if self.cal_mode:
            cv2.line(c, (0, y), (PW, y), BORDER, 1)
            y += 6
            cv2.putText(c, "HAND MARGIN", (14, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, DIM, 1)
            self._slider(c, "handmargin_0", 130, y, 150, self.hand_margin / 0.08, ACCENT)
            cv2.putText(c, f"{self.hand_margin:.3f}", (286, y + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, DIM, 1)
            y += 24

        # === WAVEFORM EDITOR (tune mode) ===
        if self.cal_mode:
            g = self.gestures[self.selected]
            sn = self.mapping.get(g, "none")
            if sn != "none" and sn in self.sampler.samples:
                sample = self.sampler.samples[sn]
                cv2.line(c, (0, y), (PW, y), BORDER, 1)
                y += 6

                # Label: gesture name AND sample name
                gesture_label = self.labels.get(g, g)
                cv2.putText(c, "TRIM EDITOR", (14, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, ACCENT, 1)
                cv2.putText(c, f"{gesture_label}  ->  {sn}", (130, y + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, TEXT, 1)
                y += 22

                # Instruction text
                cv2.putText(c, "Drag S/E markers to trim", (14, y + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1)
                y += 18

                wf_x, wf_w, wf_h = 14, max(1, PW - 28), 90
                self._wf_x = wf_x
                self._wf_w = wf_w

                wf = self.sampler.get_waveform(sn, wf_w)
                cv2.rectangle(c, (wf_x, y), (wf_x + wf_w, y + wf_h), (30, 30, 38), -1)
                cv2.rectangle(c, (wf_x, y), (wf_x + wf_w, y + wf_h), BORDER, 1)
                mid_y = y + wf_h // 2
                for j, a in enumerate(wf):
                    bh = int(a * (wf_h // 2 - 2))
                    if bh > 0:
                        cv2.line(c, (wf_x + j, mid_y - bh), (wf_x + j, mid_y + bh), ACCENT, 1)

                ts_px = int(sample.trim_start * wf_w)
                te_px = int(sample.trim_end * wf_w)
                if ts_px > 0:
                    roi = c[y:y + wf_h, wf_x:wf_x + ts_px]
                    c[y:y + wf_h, wf_x:wf_x + ts_px] = (roi * 0.25).astype(np.uint8)
                if te_px < wf_w:
                    roi = c[y:y + wf_h, wf_x + te_px:wf_x + wf_w]
                    c[y:y + wf_h, wf_x + te_px:wf_x + wf_w] = (roi * 0.25).astype(np.uint8)

                # Trim start marker — tab handle, wider hit area (24px)
                sx_m = wf_x + ts_px
                cv2.line(c, (sx_m, y), (sx_m, y + wf_h), TRIM_COL, 2)
                cv2.rectangle(c, (sx_m - 6, y), (sx_m + 6, y + 14), TRIM_COL, -1)
                cv2.putText(c, "S", (sx_m - 4, y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                self._reg["trimstart_0"] = R(sx_m - 12, y, 24, wf_h)

                # Trim end marker — tab handle, wider hit area (24px)
                ex_m = wf_x + te_px
                cv2.line(c, (ex_m, y), (ex_m, y + wf_h), TRIM_COL, 2)
                cv2.rectangle(c, (ex_m - 6, y), (ex_m + 6, y + 14), TRIM_COL, -1)
                cv2.putText(c, "E", (ex_m - 4, y + 11),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                self._reg["trimend_0"] = R(ex_m - 12, y, 24, wf_h)

                cv2.putText(c, f"Start: {sample.trim_start:.2f}  End: {sample.trim_end:.2f}",
                            (wf_x, y + wf_h + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, DIM, 1)
                y += wf_h + 26

        # === FOOTER ===
        cv2.line(c, (0, y), (PW, y), BORDER, 1)
        y += 10
        self._btn(c, 14, y, 80, BTN_H, "RELOAD", "reload")
        cv2.putText(c, "Drop .wav/.flac into samples/, click RELOAD",
                    (106, y + 17), cv2.FONT_HERSHEY_SIMPLEX, 0.32, DIM, 1)
        if self.sampler.midi_enabled:
            cv2.putText(c, "MIDI out: 'Gesture Sampler' -> connect in DAW",
                        (14, y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.32, GREEN, 1)
        y += 56

        # Scroll + crop
        self._max_scroll = max(0, H - 750)
        self._scroll_y = min(self._scroll_y, self._max_scroll)
        sy_off = int(self._scroll_y)
        visible_h = min(750, H - sy_off)
        if visible_h <= 0:
            visible_h = min(750, H)
            sy_off = 0
        result = c[sy_off:sy_off + visible_h]

        # Scroll indicator — wider bar, brighter color, outline
        if self._max_scroll > 0:
            bar_h = max(24, int(visible_h * visible_h / H))
            bar_y = int(sy_off * (visible_h - bar_h) / max(1, self._max_scroll))
            cv2.rectangle(result, (PW - 10, bar_y), (PW - 2, bar_y + bar_h),
                          SCROLL_COL, -1)
            cv2.rectangle(result, (PW - 10, bar_y), (PW - 2, bar_y + bar_h),
                          SCROLL_BD, 1)

        cv2.imshow(self.WIN, result)
