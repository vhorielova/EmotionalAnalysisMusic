import torch
from config import SPECTROGRAMS_DIR, SEGMENT_FRAMES_CNN, MAX_SEGMENTS_PER_SONG
import numpy as np

class SpectrogramDataset(torch.utils.data.Dataset):
    def __init__(self, song_ids, labels, mean=0.0, std=1.0, augment=False):
        self.song_ids = song_ids
        self.labels = labels
        self.segments = []
        self.mean = mean
        self.std = std
        self.augment = augment
        for song_id in song_ids:
            file_path = f"{SPECTROGRAMS_DIR}/{song_id}.npy"
            n_frames = np.load(file_path, mmap_mode="r").shape[1]
            fit = n_frames // SEGMENT_FRAMES_CNN
            take = min(fit, MAX_SEGMENTS_PER_SONG)
            first = (fit - take) // 2
            segment_starts = [(first + k) * SEGMENT_FRAMES_CNN for k in range(take)]
            self.segments.extend((song_id, i) for i in segment_starts)

    def __len__(self):
        return len(self.segments)
    def __getitem__(self, idx):
        song_id, start = self.segments[idx]

        file_path = f"{SPECTROGRAMS_DIR}/{song_id}.npy"

        spectrogram = np.load(file_path, mmap_mode="r")
        segment = np.array(spectrogram[:, start : start + SEGMENT_FRAMES_CNN])
        segment = (segment-self.mean)/self.std

        if self.augment:
            min_val = segment.min()

            f_width = np.random.randint(0, 16)
            f_start = np.random.randint(0, max(1, segment.shape[0] - f_width))
            segment[f_start: f_start + f_width, :] = min_val

            t_width = np.random.randint(0, 24)
            t_start = np.random.randint(0, max(1, segment.shape[1] - t_width))
            segment[:, t_start: t_start + t_width] = min_val

        segment_tensor = torch.from_numpy(segment).unsqueeze(0)
        label = self.labels[song_id]

        return segment_tensor, label

def compute_mean_std(dataset):
    total = 0.0
    total_sq = 0.0
    count = 0

    for song_id, start in dataset.segments:
        spectrogram = np.load(f"{SPECTROGRAMS_DIR}/{song_id}.npy", mmap_mode="r")
        segment = np.array(spectrogram[:, start : start + SEGMENT_FRAMES_CNN], dtype=np.float64)

        total += segment.sum()
        total_sq += np.square(segment).sum()
        count += segment.size

    mean = total/count
    std = np.sqrt(total_sq / count - mean ** 2)
    return float(mean), float(std)