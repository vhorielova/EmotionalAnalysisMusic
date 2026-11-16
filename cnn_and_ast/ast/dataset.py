import numpy as np
import torch
from torch.utils.data import Dataset
from config import WAVEFORMS_DIR, SEGMENT_SAMPLES, MAX_SEGMENTS_AST, SAMPLE_RATE
import random


class ASTDataset(Dataset):
    def __init__(self, song_ids, labels, extractor, mode="eval"):
        self.song_ids = song_ids
        self.labels = labels
        self.segments = []
        self.extractor = extractor
        self.mode = mode

        self.starts = {}
        for song_id in song_ids:
            file_path = f"{WAVEFORMS_DIR}/{song_id}.npy"
            n_frames = np.load(file_path, mmap_mode="r").shape[0]
            fit = n_frames // SEGMENT_SAMPLES
            take = min(fit, MAX_SEGMENTS_AST)
            first = (fit - take) // 2
            segment_starts = [(first + k) * SEGMENT_SAMPLES for k in range(take)]

            self.starts[song_id] = segment_starts
            if self.mode == "eval":
                self.segments.extend((song_id, s) for s in segment_starts)
            else:
                self.segments.append((song_id, segment_starts[0]))

    def __len__(self):
        if self.mode == "eval":
            return len(self.segments)
        else:
            return len(self.song_ids)

    def __getitem__(self, idx):
        song_id, start = self.segments[idx]

        if self.mode == "train":
            start = random.choice(self.starts[song_id])

        file_path = f"{WAVEFORMS_DIR}/{song_id}.npy"

        waveform = np.load(file_path, mmap_mode="r")
        segment = np.array(
            waveform[start: start + SEGMENT_SAMPLES],
            dtype=np.float32
        )

        inputs = self.extractor(
            segment,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )

        input_values = inputs["input_values"].squeeze(0)

        label = torch.tensor(
            self.labels[song_id],
            dtype=torch.long
        )

        return input_values, label
