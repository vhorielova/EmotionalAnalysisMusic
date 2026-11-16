import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from config import RANDOM_STATE, TEST_SIZE, SAMPLE_RATE, SEGMENT_SAMPLES, SEGMENT_FRAMES_CNN, MAX_SEGMENTS_AST
import json
from pathlib import Path
from collections import Counter
from config import N_FFT, HOP_LENGTH, N_MELS, FMAX, SR, SPECTROGRAMS_DIR, WAVEFORMS_DIR

def compute_spectrogram(path):
    y, sr = librosa.load(path, sr=SR)
    S = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, fmax=FMAX
    )
    S_dB = librosa.power_to_db(S, ref=1.0, top_db=None)
    return S_dB.astype(np.float32)

def load_audio_to_spectrogram(audio_dir, song_ids):
    out_dir = Path(SPECTROGRAMS_DIR)
    out_dir.mkdir(exist_ok=True)

    for song_id in song_ids:
        out_path = out_dir / f"{song_id}.npy"
        if out_path.exists():
            continue

        audio_path = Path(audio_dir) / f"{song_id}.mp3"
        if not audio_path.exists():
            print("No audio file for song", song_id)
            continue

        S_dB = compute_spectrogram(audio_path)
        np.save(out_path, S_dB)


def check_spectrogram_shapes():
    shapes = Counter()
    for file in Path(SPECTROGRAMS_DIR).glob("*.npy"):
        shapes[np.load(file, mmap_mode="r").shape] += 1
    print(shapes)

def display_spectrogram(S_dB, sr=None):
    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis='time',
                                   y_axis='mel', sr=sr,
                                   fmax=8000, ax=ax)
    fig.colorbar(img, ax=ax, format="%+2.0f dB")
    ax.set(title='Mel-frequency spectrogram')
    plt.tight_layout()
    plt.show()

def load_audio_to_waveform(audio_dir, song_ids):
    out_dir = Path(WAVEFORMS_DIR)
    out_dir.mkdir(exist_ok=True)

    for song_id in song_ids:
        out_path = out_dir / f"{song_id}.npy"
        if out_path.exists():
            continue

        y, _ = librosa.load(Path(audio_dir) / f"{song_id}.mp3", sr=SAMPLE_RATE)

        fit = len(y) // SEGMENT_SAMPLES
        take = min(fit, MAX_SEGMENTS_AST)
        first = (fit - take) // 2

        y = y[first * SEGMENT_SAMPLES : (first + take) * SEGMENT_SAMPLES]
        np.save(out_path, y.astype(np.float16))

def check_waveform_shapes():
    shapes = Counter()

    for file in Path(WAVEFORMS_DIR).glob("*.npy"):
        y = np.load(file, mmap_mode="r")
        shapes[y.shape] += 1

    print(shapes)