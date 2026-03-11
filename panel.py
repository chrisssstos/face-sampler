"""
Gesture Control Panel — PyQt6 UI for gesture → Ableton clip mapping.
All sounds handled in Ableton via OSC (AbletonOSC) + MIDI.
"""
import os
import time
import threading

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QGridLayout, QLabel, QPushButton, QSlider, QSpinBox, QGroupBox,
    QFrame, QScrollArea, QLineEdit,
)
from PyQt6.QtCore import Qt, QTimer

import socket
from pythonosc.udp_client import SimpleUDPClient
from pythonosc.osc_message_builder import OscMessageBuilder
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import ThreadingOSCUDPServer
import mido


DARK_SS = """
QWidget {
    background: #16161c;
    color: #dcdce1;
    font-family: 'Menlo', 'Monaco', monospace;
    font-size: 12px;
}
QGroupBox {
    border: 1px solid #32323c;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
    font-weight: bold;
    color: #00b4ff;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QPushButton {
    background: #32323e;
    border: 1px solid #3c3c4a;
    border-radius: 3px;
    padding: 4px 10px;
    color: #dcdce1;
    min-height: 22px;
}
QPushButton:hover { background: #414152; }
QPushButton:pressed { background: #00b4ff; color: #000; }
QPushButton:checked { background: #00d25a; color: #000; border-color: #00d25a; }
QSlider::groove:horizontal {
    height: 6px;
    background: #2d2d38;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    width: 14px;
    height: 14px;
    margin: -4px 0;
    background: #dcdce1;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #00c864;
    border-radius: 3px;
}
QLabel { background: transparent; }
QFrame[frameShape="4"] { color: #32323c; }
QScrollArea { border: none; }
QLineEdit {
    background: #32323e;
    border: 1px solid #3c3c4a;
    border-radius: 3px;
    padding: 3px 6px;
    color: #dcdce1;
}
"""


class GestureSlot(QFrame):
    """One gesture → OSC clip / MIDI note mapping row."""

    def __init__(self, index, gesture, label, color, parent_panel):
        super().__init__()
        self.index = index
        self.gesture = gesture
        self.panel = parent_panel
        self.color = color

        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setStyleSheet(f"QFrame {{ border-left: 3px solid {color}; border-radius: 2px; }}")
        self.setFixedHeight(48)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 2, 6, 2)
        layout.setSpacing(8)

        num = QLabel(f"{index + 1}")
        num.setFixedWidth(16)
        num.setStyleSheet("color: #646470;")
        layout.addWidget(num)

        self.name_label = QLabel(label)
        self.name_label.setFixedWidth(80)
        self.name_label.setStyleSheet(f"color: {color}; font-weight: bold;")
        layout.addWidget(self.name_label)

        # Mode cycle
        self._mode_idx = 0
        self._mode_list = ["shot", "gate", "toggle", "loop"]
        self._mode_labels = {"shot": "SHOT", "gate": "GATE", "toggle": "TGGL", "loop": "LOOP"}
        self._mode_colors = {
            "shot": "#dcdce1", "gate": "#00d25a", "toggle": "#ff8c00", "loop": "#00b4ff",
        }
        self.mode_btn = QPushButton("SHOT")
        self.mode_btn.setFixedWidth(50)
        self.mode_btn.clicked.connect(self._on_mode)
        layout.addWidget(self.mode_btn)

        # OSC track + clip
        layout.addWidget(self._lbl("TRK", "#e040ff"))
        self.osc_track = QSpinBox()
        self.osc_track.setRange(0, 99)
        self.osc_track.setValue(index)
        self.osc_track.setFixedWidth(44)
        self.osc_track.valueChanged.connect(self._on_osc_map)
        layout.addWidget(self.osc_track)

        layout.addWidget(self._lbl("CLIP", "#e040ff"))
        self.osc_clip = QSpinBox()
        self.osc_clip.setRange(0, 99)
        self.osc_clip.setValue(0)
        self.osc_clip.setFixedWidth(44)
        self.osc_clip.valueChanged.connect(self._on_osc_map)
        layout.addWidget(self.osc_clip)

        # MIDI note
        layout.addWidget(self._lbl("NOTE", "#646470"))
        self.midi_note = QSpinBox()
        self.midi_note.setRange(0, 127)
        self.midi_note.setValue(36 + index)
        self.midi_note.setFixedWidth(44)
        self.midi_note.valueChanged.connect(self._on_midi_note)
        layout.addWidget(self.midi_note)

        # Track name label (populated from Ableton)
        self.track_name = QLabel("")
        self.track_name.setFixedWidth(90)
        self.track_name.setStyleSheet("color: #646470; font-size: 10px;")
        layout.addWidget(self.track_name)

        # Trigger indicator
        self.trigger_dot = QLabel("●")
        self.trigger_dot.setFixedWidth(16)
        self.trigger_dot.setStyleSheet("color: #1e1e26; font-size: 16px;")
        layout.addWidget(self.trigger_dot)

        layout.addStretch()

    def _lbl(self, text, color):
        l = QLabel(text)
        l.setStyleSheet(f"color: {color}; font-size: 10px;")
        l.setFixedWidth(28)
        return l

    def set_active(self, active):
        self.trigger_dot.setStyleSheet(
            "color: #00ff78; font-size: 16px;" if active else "color: #1e1e26; font-size: 16px;")

    def _on_mode(self):
        self._mode_idx = (self._mode_idx + 1) % len(self._mode_list)
        mode = self._mode_list[self._mode_idx]
        self.mode_btn.setText(self._mode_labels[mode])
        col = self._mode_colors[mode]
        self.mode_btn.setStyleSheet(f"QPushButton {{ color: {col}; }}")
        self.panel.modes[self.gesture] = mode

    def _on_osc_map(self):
        track_idx = self.osc_track.value()
        self.panel.osc_mappings[self.gesture] = (track_idx, self.osc_clip.value())
        name = self.panel.track_names.get(track_idx, "")
        if track_idx in self.panel.group_tracks:
            self.track_name.setText(f"[G] {name}")
            self.track_name.setStyleSheet("color: #ff8c00; font-size: 10px;")
        else:
            self.track_name.setText(name)
            self.track_name.setStyleSheet("color: #646470; font-size: 10px;")

    def _on_midi_note(self):
        self.panel.midi_notes[self.gesture] = self.midi_note.value()

    def mousePressEvent(self, ev):
        self.panel.select_slot(self.index)


class SamplerPanel(QMainWindow):
    """Gesture control panel — maps gestures to Ableton clips via OSC + MIDI."""

    def __init__(self, gestures, labels, colors, modes, cal, face_gestures):
        super().__init__()
        self.gestures = list(gestures)
        self.labels = dict(labels)
        self.colors = dict(colors)
        self.modes = modes
        self.cal = cal
        self.face_gestures = face_gestures

        self.face_enabled = True
        self.selected = 0
        self.thresholds = {
            "mouth_open": 0.06, "left_eyebrow": 0.04, "right_eyebrow": 0.04,
        }
        self.hand_margin = 0.0
        self.auto_cal_requested = False
        self.toggle_states = {}
        self.capture_mode = False
        self.capture_slot = -1
        self.capture_countdown = 0.0
        self.captured_gestures = {}
        self.live_face = {"mouth": 0.0, "left_brow": 0.0, "right_brow": 0.0}
        self.trigger_times = {}

        # OSC
        self.osc_enabled = False
        self.osc_client = None
        self.osc_mappings = {}
        for i, g in enumerate(gestures):
            self.osc_mappings[g] = (i, 0)
        self.track_names = {}  # track_idx → name
        self.group_tracks = set()  # indices of group (foldable) tracks
        self._num_tracks = 0
        self._track_info_pending = 0  # counter for pending queries
        self._tracks_dirty = False  # flag for main-thread UI update
        self._osc_server = None
        self._osc_server_thread = None

        # MIDI
        self.midi_enabled = False
        self.midi_port = None
        self.midi_channel = 9  # channel 10 (0-indexed)
        self.midi_notes = {}
        for i, g in enumerate(gestures):
            self.midi_notes[g] = 36 + i
        self._init_midi()

        self.setWindowTitle("Gesture Control")
        self.setMinimumSize(580, 620)
        self.setStyleSheet(DARK_SS)

        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        # === HEADER ===
        header = QHBoxLayout()
        title = QLabel("GESTURE CONTROL")
        title.setStyleSheet("color: #00b4ff; font-size: 16px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()

        self.face_btn = QPushButton("FACE ON")
        self.face_btn.setCheckable(True)
        self.face_btn.setChecked(True)
        self.face_btn.clicked.connect(self._toggle_face)
        header.addWidget(self.face_btn)

        self.capture_btn = QPushButton("CAPTURE")
        self.capture_btn.setStyleSheet("QPushButton { color: #0050ff; }")
        self.capture_btn.clicked.connect(self._toggle_capture)
        header.addWidget(self.capture_btn)

        self.autocal_btn = QPushButton("AUTO CAL")
        self.autocal_btn.clicked.connect(lambda: setattr(self, 'auto_cal_requested', True))
        header.addWidget(self.autocal_btn)

        main_layout.addLayout(header)

        # === OSC CONNECTION ===
        osc_group = QGroupBox("ABLETON OSC")
        osc_layout = QHBoxLayout(osc_group)
        osc_layout.setSpacing(8)

        self.osc_btn = QPushButton("CONNECT")
        self.osc_btn.setCheckable(True)
        self.osc_btn.setStyleSheet(
            "QPushButton { color: #e040ff; } "
            "QPushButton:checked { background: #e040ff; color: #000; }")
        self.osc_btn.clicked.connect(self._toggle_osc)
        osc_layout.addWidget(self.osc_btn)

        osc_layout.addWidget(QLabel("IP:"))
        self.osc_ip = QLineEdit("127.0.0.1")
        self.osc_ip.setFixedWidth(100)
        osc_layout.addWidget(self.osc_ip)

        osc_layout.addWidget(QLabel("Port:"))
        self.osc_port = QSpinBox()
        self.osc_port.setRange(1, 65535)
        self.osc_port.setValue(11000)
        self.osc_port.setFixedWidth(60)
        osc_layout.addWidget(self.osc_port)

        self.osc_status = QLabel("Disconnected")
        self.osc_status.setStyleSheet("color: #646470;")
        osc_layout.addWidget(self.osc_status)
        osc_layout.addStretch()

        main_layout.addWidget(osc_group)

        # === MIDI ===
        midi_bar = QHBoxLayout()
        self.midi_btn = QPushButton("MIDI OFF")
        self.midi_btn.setCheckable(True)
        self.midi_btn.clicked.connect(self._toggle_midi)
        midi_bar.addWidget(self.midi_btn)

        midi_bar.addWidget(QLabel("Ch:"))
        self.midi_ch = QSpinBox()
        self.midi_ch.setRange(1, 16)
        self.midi_ch.setValue(10)
        self.midi_ch.setFixedWidth(44)
        self.midi_ch.valueChanged.connect(lambda v: setattr(self, 'midi_channel', v - 1))
        midi_bar.addWidget(self.midi_ch)

        self.midi_status = QLabel("")
        self.midi_status.setStyleSheet("color: #646470;")
        midi_bar.addWidget(self.midi_status)
        midi_bar.addStretch()
        main_layout.addLayout(midi_bar)

        # === TRANSPORT ===
        transport_group = QGroupBox("ABLETON TRANSPORT")
        transport_layout = QHBoxLayout(transport_group)
        transport_layout.setSpacing(6)

        self.play_btn = QPushButton("▶ PLAY")
        self.play_btn.setStyleSheet("QPushButton { color: #00d25a; font-weight: bold; }")
        self.play_btn.clicked.connect(lambda: self._osc_send("/live/song/start_playing"))
        transport_layout.addWidget(self.play_btn)

        self.stop_btn = QPushButton("■ STOP")
        self.stop_btn.setStyleSheet("QPushButton { color: #ff3040; }")
        self.stop_btn.clicked.connect(lambda: self._osc_send("/live/song/stop_playing"))
        transport_layout.addWidget(self.stop_btn)

        self.stop_clips_btn = QPushButton("STOP ALL CLIPS")
        self.stop_clips_btn.clicked.connect(lambda: self._osc_send("/live/song/stop_all_clips"))
        transport_layout.addWidget(self.stop_clips_btn)

        transport_layout.addStretch()
        main_layout.addWidget(transport_group)

        # Divider
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(line)

        # === GESTURE SLOTS ===
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        slots_widget = QWidget()
        self.slots_layout = QVBoxLayout(slots_widget)
        self.slots_layout.setContentsMargins(0, 0, 0, 0)
        self.slots_layout.setSpacing(2)

        self.slot_widgets = []
        for i, g in enumerate(self.gestures):
            slot = GestureSlot(i, g, self.labels.get(g, g), self.colors.get(g, "#646464"), self)
            self.slot_widgets.append(slot)
            self.slots_layout.addWidget(slot)
        self.slots_layout.addStretch()

        scroll.setWidget(slots_widget)
        main_layout.addWidget(scroll, stretch=2)

        # Divider
        line2 = QFrame()
        line2.setFrameShape(QFrame.Shape.HLine)
        main_layout.addWidget(line2)

        # === DETECTION THRESHOLDS ===
        self.face_group = QGroupBox("DETECTION THRESHOLDS")
        face_layout = QGridLayout(self.face_group)
        self.thresh_sliders = {}
        for row, (g, label) in enumerate([
            ("mouth_open", "Mouth"),
            ("left_eyebrow", "L Eyebrow"),
            ("right_eyebrow", "R Eyebrow"),
        ]):
            face_layout.addWidget(QLabel(label), row, 0)
            sl = QSlider(Qt.Orientation.Horizontal)
            sl.setRange(5, 100)
            sl.setValue(int(self.thresholds[g] * 1000))
            sl.valueChanged.connect(lambda v, gg=g: self._set_threshold(gg, v / 1000.0))
            face_layout.addWidget(sl, row, 1)

            live_lbl = QLabel("0.000")
            live_lbl.setFixedWidth(40)
            live_lbl.setStyleSheet("color: #00d25a;")
            face_layout.addWidget(live_lbl, row, 2)

            val_label = QLabel(f"{self.thresholds[g]:.3f}")
            val_label.setFixedWidth(40)
            val_label.setStyleSheet("color: #ff8c00;")
            face_layout.addWidget(val_label, row, 3)

            self.thresh_sliders[g] = (sl, val_label, live_lbl)

        face_layout.addWidget(QLabel("Hand Margin"), 3, 0)
        self.hand_margin_sl = QSlider(Qt.Orientation.Horizontal)
        self.hand_margin_sl.setRange(0, 80)
        self.hand_margin_sl.valueChanged.connect(lambda v: setattr(self, 'hand_margin', v / 1000.0))
        face_layout.addWidget(self.hand_margin_sl, 3, 1)

        main_layout.addWidget(self.face_group)

        # === FOOTER ===
        footer = QHBoxLayout()
        self.status_label = QLabel("Ready — connect OSC to Ableton")
        self.status_label.setStyleSheet("color: #646470;")
        footer.addWidget(self.status_label)
        footer.addStretch()
        main_layout.addLayout(footer)

        self.select_slot(0)

        # Update timer
        self._timer = QTimer()
        self._timer.timeout.connect(self._tick)
        self._timer.start(66)

        super().show()

    # --- MIDI ---
    def _init_midi(self):
        try:
            self.midi_port = mido.open_output("Gesture Controller", virtual=True)
            print("MIDI: Virtual port 'Gesture Controller' created")
        except Exception as e:
            print(f"MIDI: Could not create virtual port: {e}")
            self.midi_port = None

    def midi_fire(self, gesture):
        if not self.midi_enabled or not self.midi_port:
            return
        note = self.midi_notes.get(gesture, 36)
        try:
            self.midi_port.send(mido.Message("note_on", channel=self.midi_channel,
                                              note=note, velocity=127))
            # Auto note-off after 150ms
            def off():
                import time
                time.sleep(0.15)
                try:
                    self.midi_port.send(mido.Message("note_off", channel=self.midi_channel,
                                                      note=note, velocity=0))
                except Exception:
                    pass
            threading.Thread(target=off, daemon=True).start()
        except Exception:
            pass

    def midi_stop(self, gesture):
        if not self.midi_enabled or not self.midi_port:
            return
        note = self.midi_notes.get(gesture, 36)
        try:
            self.midi_port.send(mido.Message("note_off", channel=self.midi_channel,
                                              note=note, velocity=0))
        except Exception:
            pass

    # --- Public API ---
    def get_threshold(self, gesture):
        return self.thresholds.get(gesture, 0.03)

    def select_slot(self, idx):
        self.selected = idx
        for i, slot in enumerate(self.slot_widgets):
            sel = (i == idx)
            if sel:
                slot.setStyleSheet(f"QFrame {{ border-left: 3px solid {slot.color}; "
                                   f"border: 1px solid #00b4ff; border-radius: 2px; }}")
            else:
                slot.setStyleSheet(f"QFrame {{ border-left: 3px solid {slot.color}; border-radius: 2px; }}")

    def osc_fire(self, gesture):
        if not self.osc_enabled or not self.osc_client:
            return
        track, clip = self.osc_mappings.get(gesture, (0, 0))
        try:
            self.osc_client.send_message("/live/clip/fire", [track, clip])
        except Exception:
            pass

    def osc_stop(self, gesture):
        if not self.osc_enabled or not self.osc_client:
            return
        track, clip = self.osc_mappings.get(gesture, (0, 0))
        try:
            self.osc_client.send_message("/live/clip/stop", [track, clip])
        except Exception:
            pass

    def update(self):
        QApplication.processEvents()

    def cleanup(self):
        self._stop_osc_server()
        if self.midi_port:
            self.midi_port.close()

    # --- Internal ---
    def _tick(self):
        # Apply track info from OSC thread on main thread
        if self._tracks_dirty:
            self._tracks_dirty = False
            non_group = [i for i in range(self._num_tracks) if i not in self.group_tracks]
            for slot_idx, slot in enumerate(self.slot_widgets):
                if slot_idx < len(non_group):
                    slot.osc_track.setValue(non_group[slot_idx])
            self._update_track_labels()
            n_grp = len(self.group_tracks)
            print(f"Tracks: {self._num_tracks} total, {n_grp} group(s) skipped")

        now = time.time()
        for slot in self.slot_widgets:
            active = (now - self.trigger_times.get(slot.gesture, 0)) < 0.25
            slot.set_active(active)

        face_keys = {"mouth_open": "mouth", "left_eyebrow": "left_brow", "right_eyebrow": "right_brow"}
        for g, fk in face_keys.items():
            if g in self.thresh_sliders:
                _, _, live_lbl = self.thresh_sliders[g]
                val = self.live_face.get(fk, 0.0)
                live_lbl.setText(f"{val:.3f}")
                thresh = self.thresholds[g]
                live_lbl.setStyleSheet("color: #ff3040;" if val > thresh else "color: #00d25a;")

    def _osc_send(self, address, args=None):
        if self.osc_enabled and self.osc_client:
            try:
                self.osc_client.send_message(address, args or [])
            except Exception as e:
                self.osc_status.setText(f"Error: {e}")

    def _toggle_face(self):
        self.face_enabled = self.face_btn.isChecked()
        self.face_btn.setText("FACE ON" if self.face_enabled else "FACE OFF")

    def _toggle_midi(self):
        self.midi_enabled = self.midi_btn.isChecked()
        self.midi_btn.setText("MIDI ON" if self.midi_enabled else "MIDI OFF")

    def _toggle_osc(self):
        self.osc_enabled = self.osc_btn.isChecked()
        if self.osc_enabled:
            try:
                ip = self.osc_ip.text().strip() or "127.0.0.1"
                port = self.osc_port.value()
                self._start_osc_server()
                self.osc_client = SimpleUDPClient(ip, port)
                self.osc_btn.setText("CONNECTED")
                self.osc_status.setText(f"Connected → {ip}:{port}")
                self.osc_status.setStyleSheet("color: #00d25a;")
                self.status_label.setText("OSC active — gestures → Ableton")
                self._fetch_tracks()
            except Exception as e:
                self.osc_enabled = False
                self.osc_btn.setChecked(False)
                self.osc_btn.setText("CONNECT")
                self.osc_status.setText(f"Error: {e}")
                self.osc_status.setStyleSheet("color: #ff3040;")
        else:
            self.osc_client = None
            self._stop_osc_server()
            self.osc_btn.setText("CONNECT")
            self.osc_status.setText("Disconnected")
            self.osc_status.setStyleSheet("color: #646470;")

    def _toggle_capture(self):
        if not self.capture_mode:
            self.capture_mode = True
            self.capture_slot = self.selected
            self.capture_countdown = time.time()
            self.capture_btn.setText("STOP")
            self.capture_btn.setStyleSheet("QPushButton { color: #ff3040; font-weight: bold; }")
        else:
            self.capture_mode = False
            self.capture_btn.setText("CAPTURE")
            self.capture_btn.setStyleSheet("QPushButton { color: #0050ff; }")

    def _start_osc_server(self):
        if self._osc_server:
            return
        dispatcher = Dispatcher()
        dispatcher.map("/live/track/get/name", self._on_track_name)
        dispatcher.map("/live/track/get/is_foldable", self._on_track_foldable)
        dispatcher.map("/live/song/get/num_tracks", self._on_num_tracks)
        # Try a range of ports for the response listener
        for port in range(11001, 11010):
            try:
                self._osc_server = ThreadingOSCUDPServer(("0.0.0.0", port), dispatcher)
                self._osc_server_thread = threading.Thread(
                    target=self._osc_server.serve_forever, daemon=True)
                self._osc_server_thread.start()
                self._osc_listen_port = port
                return
            except OSError:
                continue
        print("OSC: Could not start response listener")

    def _stop_osc_server(self):
        if self._osc_server:
            self._osc_server.shutdown()
            self._osc_server = None
            self._osc_server_thread = None
        self.track_names.clear()
        self._update_track_labels()

    def _osc_query(self, address, args=None):
        """Send an OSC message from the response server's socket so replies come back to it."""
        if not self._osc_server:
            return
        msg = OscMessageBuilder(address=address)
        for a in (args or []):
            msg.add_arg(a)
        ip = self.osc_ip.text().strip() or "127.0.0.1"
        port = self.osc_port.value()
        self._osc_server.socket.sendto(msg.build().dgram, (ip, port))

    def _on_num_tracks(self, address, *args):
        if args:
            n = int(args[0])
            self._num_tracks = n
            self._track_info_pending = n * 2  # name + foldable for each
            for i in range(n):
                self._osc_query("/live/track/get/name", [i])
                self._osc_query("/live/track/get/is_foldable", [i])

    def _on_track_name(self, address, *args):
        if len(args) >= 2:
            idx, name = int(args[0]), str(args[1])
            self.track_names[idx] = name
            self._track_info_pending -= 1
            if self._track_info_pending <= 0:
                self._tracks_dirty = True

    def _on_track_foldable(self, address, *args):
        if len(args) >= 2:
            idx, foldable = int(args[0]), bool(args[1])
            if foldable:
                self.group_tracks.add(idx)
            self._track_info_pending -= 1
            if self._track_info_pending <= 0:
                self._tracks_dirty = True

    def _fetch_tracks(self):
        self.track_names.clear()
        self.group_tracks.clear()
        self._osc_query("/live/song/get/num_tracks")

    def _update_track_labels(self):
        for slot in self.slot_widgets:
            track_idx = slot.osc_track.value()
            name = self.track_names.get(track_idx, "")
            if track_idx in self.group_tracks:
                slot.track_name.setText(f"[G] {name}")
                slot.track_name.setStyleSheet("color: #ff8c00; font-size: 10px;")
            else:
                slot.track_name.setText(name)
                slot.track_name.setStyleSheet("color: #646470; font-size: 10px;")

    def _set_threshold(self, gesture, val):
        self.thresholds[gesture] = val
        sl, val_label, _ = self.thresh_sliders[gesture]
        val_label.setText(f"{val:.3f}")
