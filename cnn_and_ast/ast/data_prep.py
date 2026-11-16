import kagglehub
from cnn_and_ast.audio_loading import load_audio_to_waveform, check_waveform_shapes
from pathlib import Path
from cnn_and_ast.ast.dataset import ASTDataset
from torch.utils.data import DataLoader
from cnn_and_ast.split_ids import split_ids

def prepare_data(df, train_ids, test_ids, extractor):
    ensure_waveforms(df)

    labels = df.set_index("song_id")["emotion"].to_dict()

    train_ids, val_ids, test_ids = split_ids(labels, train_ids, test_ids)
    train_dataset, val_dataset, test_dataset = build_datasets(labels, train_ids, val_ids, test_ids, extractor)
    train_loader, val_loader, test_loader = build_loaders(train_dataset, val_dataset, test_dataset)

    return train_loader, val_loader, test_loader, train_dataset, labels


def ensure_waveforms(df):
    dataset_path = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")

    audio_dir = Path(dataset_path) / "DEAM_audio/MEMD_audio"
    song_ids = df["song_id"].tolist()
    load_audio_to_waveform(audio_dir, song_ids)
    check_waveform_shapes()


def build_datasets(labels, train_ids, val_ids, test_ids, extractor):
    train_labels = {key: labels[key] for key in labels if key in train_ids}
    test_labels = {key: labels[key] for key in labels if key in test_ids}
    val_labels = {key: labels[key] for key in labels if key in val_ids}

    train_dataset = ASTDataset(train_ids, train_labels, extractor, mode="train")
    test_dataset = ASTDataset(test_ids, test_labels, extractor, mode="eval")
    val_dataset = ASTDataset(val_ids, val_labels, extractor, mode="eval")

    return train_dataset, val_dataset, test_dataset


def build_loaders(train_dataset, val_dataset, test_dataset):
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    for inputs, labels_batch in train_loader:
        print("Inputs shape:", inputs.shape)
        print("Labels shape:", labels_batch.shape)
        print("Labels:", labels_batch)

        break

    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return train_loader, val_loader, test_loader