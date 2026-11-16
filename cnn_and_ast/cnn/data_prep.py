from sklearn.model_selection import train_test_split
from config import VAL_SIZE, RANDOM_STATE
import pandas as pd
from cnn_and_ast.cnn.dataset import SpectrogramDataset, compute_mean_std
from torch.utils.data import DataLoader
from cnn_and_ast.audio_loading import load_audio_to_spectrogram, check_spectrogram_shapes
import kagglehub
from pathlib import Path


def split_ids(labels, train_ids, test_ids):
    train_ids = train_ids.tolist()
    test_ids = test_ids.tolist()

    y_train_ids = pd.Series([labels[s] for s in train_ids])

    train_ids, val_ids = train_test_split(train_ids, test_size=VAL_SIZE, random_state=RANDOM_STATE,
                                          stratify=y_train_ids)

    print("Test size: ", len(test_ids))
    print("Train size: ", len(train_ids))
    print("Validation size: ", len(val_ids))

    print(pd.Series([labels[s] for s in test_ids]).value_counts(normalize=True))
    print(y_train_ids.value_counts(normalize=True))
    print(pd.Series([labels[s] for s in val_ids]).value_counts(normalize=True))

    print(set(train_ids) & set(val_ids), set(train_ids) & set(test_ids), set(val_ids) & set(test_ids))

    return train_ids, val_ids, test_ids

def build_datasets(labels, train_ids, val_ids, test_ids):
    train_labels = {key: labels[key] for key in labels if key in train_ids}
    test_labels = {key: labels[key] for key in labels if key in test_ids}
    val_labels = {key: labels[key] for key in labels if key in val_ids}

    train_dataset = SpectrogramDataset(train_ids, train_labels)
    test_dataset = SpectrogramDataset(test_ids, test_labels)
    val_dataset = SpectrogramDataset(val_ids, val_labels)

    mean, std = compute_mean_std(train_dataset)

    train_dataset = SpectrogramDataset(train_ids, train_labels, mean, std, True)
    test_dataset = SpectrogramDataset(test_ids, test_labels, mean, std)
    val_dataset = SpectrogramDataset(val_ids, val_labels, mean, std)

    return train_dataset, val_dataset, test_dataset


def build_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    return train_loader, val_loader, test_loader


def prepare_data(df, train_ids, test_ids):
    ensure_spectrograms(df)

    labels = df.set_index("song_id")["emotion"].to_dict()

    train_ids, val_ids, test_ids = split_ids(labels, train_ids, test_ids)
    train_dataset, val_dataset, test_dataset = build_datasets(labels, train_ids, val_ids, test_ids)
    train_loader, val_loader, test_loader = build_loaders(train_dataset, val_dataset, test_dataset)

    return train_loader, val_loader, test_loader, train_dataset, labels


def ensure_spectrograms(df):
    dataset_path = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")

    audio_dir = Path(dataset_path) / "DEAM_audio/MEMD_audio"
    song_ids = df["song_id"].tolist()
    load_audio_to_spectrogram(audio_dir, song_ids)
    check_spectrogram_shapes()