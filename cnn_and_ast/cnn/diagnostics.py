import numpy as np
import pandas as pd
from config import SPECTROGRAMS_DIR


def print_batch_info(train_loader, val_loader, test_loader):
    x, y = next(iter(train_loader))
    print(x.shape, x.dtype)
    print(y.shape, y.dtype)
    print(x.mean().item(), x.std().item())
    print(len(train_loader), len(val_loader), len(test_loader))


def print_model_info(model, train_loader, device):
    x, y = next(iter(train_loader))
    out = model(x.to(device))
    print(out.shape)
    print(sum(p.numel() for p in model.parameters()))


def check_label_balance(dataset, long_threshold=2100):
    labels = dataset.labels

    n_frames = {
        song_id: np.load(f"{SPECTROGRAMS_DIR}/{song_id}.npy", mmap_mode="r").shape[1]
        for song_id in dataset.song_ids
    }
    long_ids = [song_id for song_id, n in n_frames.items() if n > long_threshold]

    segment_song_ids = [song_id for song_id, _ in dataset.segments]
    long_set = set(long_ids)
    long_segments = sum(song_id in long_set for song_id in segment_song_ids)
    print(f"Number of songs: {len(dataset.song_ids)}, long (>{long_threshold} frames): {len(long_ids)}")
    print(f"Number of segments: {len(segment_song_ids)}, from long songs: {long_segments} "
          f"({long_segments / len(segment_song_ids):.1%})")

    per_song = pd.Series([labels[s] for s in dataset.song_ids]).value_counts(normalize=True)
    per_segment = pd.Series([labels[s] for s in segment_song_ids]).value_counts(normalize=True)
    only_long = pd.Series([labels[s] for s in long_ids]).value_counts(normalize=True)

    table = pd.DataFrame({
        "per song": per_song,
        "per segment": per_segment,
        "only long songs": only_long,
    }).sort_index().fillna(0).round(3)
    print(table)

    shapes = {dataset[i][0].shape for i in range(0, len(dataset), 50)}
    print(shapes)