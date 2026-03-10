"""
Audio engine: built-in sampler + MIDI output.
When MIDI mode is on, gestures send MIDI notes to a virtual port
that any DAW/sampler (Ableton, Logic, Just-a-Sample, etc.) can receive.
"""
import os
import threading
import numpy as np
import sounddevice as sd
import soundfile as sf
import mido

SAMPLE_RATE = 44100


class Sample:
    def __init__(self, name, file_path, data, sr):
        self.name = name
        self.file_path = file_path
        self.data = data
        self.sample_rate = sr
        self.volume = 0.8
        self.trim_start = 0.0  # 0-1 fraction
        self.trim_end = 1.0    # 0-1 fraction

    @classmethod
    def from_file(cls, path):
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            dur = len(data) / sr
            new_len = int(dur * SAMPLE_RATE)
            data = np.interp(np.linspace(0, len(data) - 1, new_len), np.arange(len(data)), data)
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak
        return cls(os.path.splitext(os.path.basename(path))[0], path,
                   data.astype(np.float32), SAMPLE_RATE)


class Sampler:
    """Low-latency polyphonic sampler + MIDI virtual port output."""

    def __init__(self, max_voices=16, buffer_size=256):
        self.max_voices = max_voices
        self.samples = {}
        self._voices = []
        self._lock = threading.Lock()

        # Audio stream
        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=buffer_size, latency="low",
            callback=self._audio_cb)
        self._stream.start()

        # MIDI
        self.midi_enabled = False
        self.midi_port = None
        self.midi_channel = 9  # drum channel (0-indexed, so channel 10)
        # Note mapping: sample name -> MIDI note
        self.midi_notes = {}
        self._init_midi()

    def _init_midi(self):
        try:
            self.midi_port = mido.open_output("Gesture Sampler", virtual=True)
            print("MIDI: Virtual port 'Gesture Sampler' created")
        except Exception as e:
            print(f"MIDI: Could not create virtual port: {e}")
            self.midi_port = None

    def _audio_cb(self, outdata, frames, time_info, status):
        output = np.zeros(frames, dtype=np.float32)
        with self._lock:
            alive = []
            for data, pos, vol in self._voices:
                rem = len(data) - pos
                if rem <= 0:
                    continue
                n = min(frames, rem)
                output[:n] += data[pos:pos + n] * vol
                npos = pos + n
                if npos < len(data):
                    alive.append((data, npos, vol))
            self._voices = alive
        np.clip(output, -1.0, 1.0, out=output)
        outdata[:, 0] = output

    def load_sample(self, path):
        s = Sample.from_file(path)
        self.samples[s.name] = s
        # Auto-assign MIDI note (36=kick, 38=snare, etc.)
        if s.name not in self.midi_notes:
            base_notes = {"kick": 36, "snare": 38, "hihat": 42, "tom": 45,
                          "clap": 39, "cowbell": 56, "rimshot": 37, "cymbal": 49}
            self.midi_notes[s.name] = base_notes.get(s.name, 36 + len(self.midi_notes))
        return s.name

    def load_directory(self, dir_path):
        exts = (".wav", ".flac", ".ogg", ".aiff")
        loaded = []
        if not os.path.isdir(dir_path):
            return loaded
        for f in sorted(os.listdir(dir_path)):
            if f.lower().endswith(exts):
                try:
                    loaded.append(self.load_sample(os.path.join(dir_path, f)))
                except Exception as e:
                    print(f"  Warning: {f}: {e}")
        return loaded

    def play(self, name, volume=1.0):
        """Play a sample (built-in audio) and/or send MIDI note."""
        # MIDI output
        if self.midi_enabled and self.midi_port and name in self.midi_notes:
            note = self.midi_notes[name]
            vel = int(max(1, min(127, volume * 127)))
            self.midi_port.send(mido.Message("note_on", channel=self.midi_channel,
                                              note=note, velocity=vel))
            # Short note-off after a bit (in a thread to not block)
            def note_off():
                import time
                time.sleep(0.15)
                try:
                    self.midi_port.send(mido.Message("note_off", channel=self.midi_channel,
                                                      note=note, velocity=0))
                except Exception:
                    pass
            threading.Thread(target=note_off, daemon=True).start()

        # Built-in audio (always play unless only MIDI)
        sample = self.samples.get(name)
        if sample is None:
            return
        # Apply trim region
        start = int(sample.trim_start * len(sample.data))
        end = int(sample.trim_end * len(sample.data))
        trimmed = sample.data[start:end] if end > start else sample.data
        with self._lock:
            while len(self._voices) >= self.max_voices:
                self._voices.pop(0)
            self._voices.append((trimmed, 0, volume * sample.volume))

    def set_volume(self, name, vol):
        s = self.samples.get(name)
        if s:
            s.volume = max(0.0, min(1.0, vol))

    def get_sample_names(self):
        return sorted(self.samples.keys())

    def get_waveform(self, name, width=100):
        if width <= 0:
            return []
        s = self.samples.get(name)
        if not s or len(s.data) == 0:
            return [0.0] * width
        data = np.abs(s.data)
        chunk = max(1, len(data) // width)
        return [float(data[i * chunk:min((i + 1) * chunk, len(data))].max())
                if i * chunk < len(data) else 0.0 for i in range(width)]

    def stop(self):
        self._stream.stop()
        self._stream.close()
        if self.midi_port:
            self.midi_port.close()
