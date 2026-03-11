"""
Audio engine: lock-free polyphonic sampler + MIDI output.
CoreAudio via PortAudio. Lock-free deque for RT-safe voice submission.
Supports BPM warping (time-stretch via rubberband).
"""
import os
import collections
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
        self.data = np.ascontiguousarray(data, dtype=np.float32)
        self.sample_rate = sr
        self.volume = 0.8
        self.trim_start = 0.0
        self.trim_end = 1.0
        self.loop_start = 0.0   # loop region within trimmed audio (0-1)
        self.loop_end = 1.0
        self.fade_in = 0.0      # seconds
        self.fade_out = 0.0     # seconds
        self.pitch_semitones = 0  # pitch shift in semitones
        self.reverse = False
        self.original_bpm = 0.0
        self._warped_cache = {}
        self._processed_cache = None
        self._processed_key = None

    @classmethod
    def from_file(cls, path):
        data, sr = sf.read(path, dtype="float32")
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != SAMPLE_RATE:
            dur = len(data) / sr
            new_len = int(dur * SAMPLE_RATE)
            data = np.interp(np.linspace(0, len(data) - 1, new_len),
                             np.arange(len(data)), data)
        peak = np.abs(data).max()
        if peak > 0:
            data = data / peak
        return cls(os.path.splitext(os.path.basename(path))[0], path,
                   data.astype(np.float32), SAMPLE_RATE)

    def get_trimmed(self):
        """Return trimmed + processed audio (fade, reverse, pitch). Cached."""
        cache_key = (self.trim_start, self.trim_end, self.fade_in, self.fade_out,
                     self.pitch_semitones, self.reverse)
        if self._processed_cache is not None and self._processed_key == cache_key:
            return self._processed_cache

        start = int(self.trim_start * len(self.data))
        end = int(self.trim_end * len(self.data))
        out = self.data[start:end].copy() if end > start else self.data.copy()

        # Reverse
        if self.reverse:
            out = out[::-1]

        # Fade in
        if self.fade_in > 0:
            n = min(len(out), int(self.fade_in * SAMPLE_RATE))
            if n > 0:
                out[:n] *= np.linspace(0, 1, n, dtype=np.float32)

        # Fade out
        if self.fade_out > 0:
            n = min(len(out), int(self.fade_out * SAMPLE_RATE))
            if n > 0:
                out[-n:] *= np.linspace(1, 0, n, dtype=np.float32)

        # Pitch shift
        if self.pitch_semitones != 0:
            try:
                import pyrubberband as pyrb
                out = pyrb.pitch_shift(out, SAMPLE_RATE, self.pitch_semitones)
                out = out.astype(np.float32)
            except Exception:
                pass

        out = np.ascontiguousarray(out)
        self._processed_cache = out
        self._processed_key = cache_key
        return out

    def get_warped(self, target_bpm):
        """Return time-stretched audio for target BPM. Cached."""
        if self.original_bpm <= 0 or target_bpm <= 0:
            return self.get_trimmed()
        ratio = self.original_bpm / target_bpm
        if abs(ratio - 1.0) < 0.01:
            return self.get_trimmed()

        # Check cache
        cache_key = (round(target_bpm, 1), round(self.trim_start, 3), round(self.trim_end, 3))
        if cache_key in self._warped_cache:
            return self._warped_cache[cache_key]

        trimmed = self.get_trimmed()
        try:
            import pyrubberband as pyrb
            stretched = pyrb.time_stretch(trimmed, SAMPLE_RATE, ratio)
            stretched = np.ascontiguousarray(stretched.astype(np.float32))
            self._warped_cache[cache_key] = stretched
            return stretched
        except Exception as e:
            print(f"Warp failed for {self.name}: {e}")
            return trimmed


class Sampler:
    """Lock-free polyphonic sampler + MIDI virtual port output."""

    def __init__(self, max_voices=16):
        self.max_voices = max_voices
        self.samples = {}

        # Lock-free voice queue
        self._pending = collections.deque()
        self._active = []  # Only touched by audio callback
        # Track which sample names are currently playing
        self._playing_names = set()  # Updated by audio callback

        # Query device
        dev_info = sd.query_devices(sd.default.device[1])
        self._channels = min(2, dev_info['max_output_channels'])
        device_latency = dev_info['default_high_output_latency']
        print(f"Audio: {dev_info['name']}, {self._channels}ch, "
              f"{int(dev_info['default_samplerate'])}Hz, latency={device_latency*1000:.1f}ms")

        self._mix_buf = np.zeros((8192, self._channels), dtype=np.float32)

        self._stream = sd.OutputStream(
            samplerate=SAMPLE_RATE,
            channels=self._channels,
            dtype="float32",
            blocksize=0,
            latency=device_latency,
            callback=self._audio_cb,
            clip_off=True,
        )
        self._stream.start()

        # MIDI
        self.midi_enabled = False
        self.midi_port = None
        self.midi_channel = 9
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
        # Drain pending voices — separate stop commands from new voices
        stop_names = set()
        new_voices = []
        while True:
            try:
                voice = self._pending.popleft()
                name, data, pos, vol = voice
                if len(data) == 0:
                    stop_names.add(name)  # stop command
                else:
                    new_voices.append(voice)
            except IndexError:
                break

        # Also choke: if a new voice has same name as active, kill the old one
        for name, _, _, _ in new_voices:
            stop_names.add(name)

        # Remove stopped voices from active list
        if stop_names:
            self._active = [v for v in self._active if v[0] not in stop_names]

        self._active.extend(new_voices)

        buf = self._mix_buf[:frames]
        buf[:] = 0.0

        playing = set()
        write = 0
        for i in range(len(self._active)):
            name, data, pos, vol = self._active[i]
            rem = len(data) - pos
            if rem <= 0:
                continue
            n = min(frames, rem)
            buf[:n, 0] += data[pos:pos + n] * vol
            new_pos = pos + n
            if new_pos < len(data):
                self._active[write] = (name, data, new_pos, vol)
                write += 1
                playing.add(name)
        del self._active[write:]
        self._playing_names = playing

        np.clip(buf[:, 0], -1.0, 1.0, out=buf[:, 0])
        for ch in range(1, self._channels):
            buf[:, ch] = buf[:, 0]
        outdata[:] = buf

    def is_playing(self, name):
        """Check if a sample is currently playing (lock-free read)."""
        return name in self._playing_names

    def stop_sample(self, name):
        """Stop all voices playing a specific sample."""
        # Mark for removal by setting a flag — audio thread will skip them
        # We do this by adding a "stop" command to pending
        self._pending.append((name, np.zeros(0, dtype=np.float32), 0, 0.0))

    def load_sample(self, path):
        s = Sample.from_file(path)
        self.samples[s.name] = s
        if s.name not in self.midi_notes:
            base_notes = {"kick": 36, "snare": 38, "hihat": 42, "tom": 45,
                          "clap": 39, "cowbell": 56, "rimshot": 37, "cymbal": 49}
            self.midi_notes[s.name] = base_notes.get(s.name, 36 + len(self.midi_notes))
        return s.name

    def load_directory(self, dir_path):
        exts = (".wav", ".flac", ".ogg", ".aiff", ".mp3")
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

    def play(self, name, volume=1.0, warp_bpm=0):
        """Play a sample. warp_bpm > 0 time-stretches to that BPM."""
        # MIDI output
        if self.midi_enabled and self.midi_port and name in self.midi_notes:
            note = self.midi_notes[name]
            vel = int(max(1, min(127, volume * 127)))
            self.midi_port.send(mido.Message("note_on", channel=self.midi_channel,
                                              note=note, velocity=vel))
            def note_off():
                import time
                time.sleep(0.15)
                try:
                    self.midi_port.send(mido.Message("note_off", channel=self.midi_channel,
                                                      note=note, velocity=0))
                except Exception:
                    pass
            threading.Thread(target=note_off, daemon=True).start()

        sample = self.samples.get(name)
        if sample is None:
            return

        # Get audio data (warped or trimmed)
        if warp_bpm > 0 and sample.original_bpm > 0:
            audio = sample.get_warped(warp_bpm)
        else:
            audio = sample.get_trimmed()

        # Voice limiting
        if len(self._pending) + len(self._active) >= self.max_voices:
            try:
                self._pending.popleft()
            except IndexError:
                pass
        self._pending.append((name, audio, 0, volume * sample.volume))

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

    def get_duration(self, name):
        """Return sample duration in seconds."""
        s = self.samples.get(name)
        if not s or len(s.data) == 0:
            return 0.0
        trimmed_len = int((s.trim_end - s.trim_start) * len(s.data))
        return trimmed_len / SAMPLE_RATE

    def stop(self):
        self._stream.stop()
        self._stream.close()
        if self.midi_port:
            self.midi_port.close()
