import kagglehub
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE
def load_dataset():
    dataset_path = kagglehub.dataset_download("imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music")


    annotations_path_1 = Path(dataset_path) / "DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
    annotations_path_2 = Path(
        dataset_path) / "DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"

    df_annotations_1 = pd.read_csv(annotations_path_1)
    df_annotations_1.columns = df_annotations_1.columns.str.strip()

    df_annotations_2 = pd.read_csv(annotations_path_2)
    df_annotations_2.columns = df_annotations_2.columns.str.strip()
    df_annotations_2 = df_annotations_2[["song_id", "valence_mean", "valence_std", "arousal_mean", "arousal_std"]]

    df_annotations = pd.concat([df_annotations_1, df_annotations_2])

    features_dir = Path(dataset_path) / "features/features"

    all_features = []
    for file in features_dir.glob("*.csv"):
        song_id = int(file.stem)
        df_features_for_one = pd.read_csv(file, sep=";")

        features = df_features_for_one.drop(columns="frameTime")

        mean_features = features.mean().add_suffix("_mean")
        std_features = features.std().add_suffix("_std")
        row = {**mean_features.to_dict(), **std_features.to_dict()}

        row["song_id"] = song_id
        all_features.append(row)

    df_features = pd.DataFrame(all_features)

    df = df_features.merge(
        df_annotations,
        how="inner",
        on="song_id"
    )

    return df


def describe_data(df):
    print("First 5 records:\n", df.head())

    print(df["valence_mean"].describe())
    print(df["arousal_mean"].describe())

    print("Missing values:", df.isna().sum().sum())


def split_data(df):
    X = df.drop(columns=["song_id", "valence_mean", "arousal_mean", "emotion", "valence_std", "arousal_std",
                         "valence_high", "arousal_high"])
    y = df["emotion"]

    X_train, X_test, y_train, y_test, yv_train, yv_test, ya_train, ya_test = train_test_split(
        X, y, df["valence_high"], df["arousal_high"],
        test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    return X_train, X_test, y_train, y_test, yv_train, yv_test, ya_train, ya_test