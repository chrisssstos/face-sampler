"""Generate simple drum samples as WAV files so the project works out of the box."""
import numpy as np
import wave
import os

SAMPLE_RATE = 44100
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), "samples")


def save_wav(filename, data):
    data = np.clip(data, -1.0, 1.0)
    pcm = (data * 32767).astype(np.int16)
    path = os.path.join(SAMPLES_DIR, filename)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(pcm.tobytes())
    print(f"  Created {path}")


def make_kick(duration=0.4):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freq = 150 * np.exp(-t * 10)
    envelope = np.exp(-t * 8)
    signal = np.sin(2 * np.pi * freq * t) * envelope
    save_wav("kick.wav", signal)


def make_snare(duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-t * 15)
    tone = np.sin(2 * np.pi * 200 * t) * 0.5
    noise = np.random.uniform(-1, 1, len(t)) * 0.7
    signal = (tone + noise) * envelope
    save_wav("snare.wav", signal)


def make_hihat(duration=0.15):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-t * 40)
    noise = np.random.uniform(-1, 1, len(t))
    signal = noise * envelope * 0.6
    save_wav("hihat.wav", signal)


def make_tom(duration=0.35):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    freq = 100 * np.exp(-t * 5)
    envelope = np.exp(-t * 7)
    signal = np.sin(2 * np.pi * freq * t) * envelope * 0.8
    save_wav("tom.wav", signal)


def make_clap(duration=0.25):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    noise = np.random.uniform(-1, 1, len(t))
    # Multiple short bursts to simulate a clap
    burst = np.zeros_like(t)
    for offset in [0.0, 0.01, 0.02]:
        burst += np.exp(-(t - offset) ** 2 / 0.0005)
    burst = burst / burst.max()
    envelope = np.exp(-t * 20) * burst
    signal = noise * envelope * 0.7
    save_wav("clap.wav", signal)


def make_cowbell(duration=0.3):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-t * 12)
    signal = (np.sin(2 * np.pi * 800 * t) * 0.6 + np.sin(2 * np.pi * 540 * t) * 0.4) * envelope
    save_wav("cowbell.wav", signal)


def make_rimshot(duration=0.15):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-t * 30)
    tone = np.sin(2 * np.pi * 1200 * t) * 0.4
    noise = np.random.uniform(-1, 1, len(t)) * 0.6
    signal = (tone + noise) * envelope
    save_wav("rimshot.wav", signal)


def make_cymbal(duration=0.6):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), endpoint=False)
    envelope = np.exp(-t * 5)
    noise = np.random.uniform(-1, 1, len(t))
    # Layer multiple high frequencies for metallic sound
    harmonics = sum(np.sin(2 * np.pi * f * t) * 0.15 for f in [3000, 4500, 6000, 7500])
    signal = (noise * 0.5 + harmonics) * envelope * 0.5
    save_wav("cymbal.wav", signal)


ALL_GENERATORS = {
    "kick": make_kick,
    "snare": make_snare,
    "hihat": make_hihat,
    "tom": make_tom,
    "clap": make_clap,
    "cowbell": make_cowbell,
    "rimshot": make_rimshot,
    "cymbal": make_cymbal,
}


if __name__ == "__main__":
    os.makedirs(SAMPLES_DIR, exist_ok=True)
    print("Generating drum samples...")
    for name, gen_fn in ALL_GENERATORS.items():
        gen_fn()
    print("Done!")
