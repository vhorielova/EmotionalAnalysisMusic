import kagglehub
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from config import RANDOM_STATE, TEST_SIZE, NEUTRAL_MARGIN
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


def split_data(df, valence_threshold, arousal_threshold):
    X = df.drop(columns=[
        "emotion",
        "valence_std",
        "arousal_std",
        "valence_high",
        "arousal_high"
    ])

    y = df[["emotion", "arousal_mean", "valence_mean"]]

    X_train, X_test, y_train, y_test, yv_train, yv_test, ya_train, ya_test = train_test_split(
        X, y, df["valence_high"], df["arousal_high"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y["emotion"]
    )

    X_train_ids = X_train[["song_id"]].copy()
    X_test_ids = X_test["song_id"].copy()

    if NEUTRAL_MARGIN > 0:
        train_mask = (
            ((X_train["valence_mean"] - valence_threshold).abs() >= NEUTRAL_MARGIN)
            &
            ((X_train["arousal_mean"] - arousal_threshold).abs() >= NEUTRAL_MARGIN)
        )
    else:
        train_mask = pd.Series(True, index=X_train.index)

    print(
        f"Songs kept after removing neutral zone: "
        f"{train_mask.sum()} of {len(train_mask)}"
    )

    X_train = X_train.loc[train_mask].copy()
    y_train = y_train.loc[train_mask].copy()
    yv_train = yv_train.loc[train_mask].copy()
    ya_train = ya_train.loc[train_mask].copy()
    X_train_ids = X_train_ids.loc[train_mask].copy()

    X_train = X_train.drop(
        columns=["song_id", "arousal_mean", "valence_mean"]
    )

    X_test = X_test.drop(
        columns=["song_id", "arousal_mean", "valence_mean"]
    )

    y_train = y_train["emotion"]
    y_test = y_test["emotion"]

    X_train_ids = X_train_ids["song_id"]

    print("X_train:", len(X_train))
    print("y_train:", len(y_train))
    print("yv_train:", len(yv_train))
    print("ya_train:", len(ya_train))

    print("X_test:", len(X_test))
    print("y_test:", len(y_test))
    print("yv_test:", len(yv_test))
    print("ya_test:", len(ya_test))

    return (
        X_train, X_test,
        y_train, y_test,
        yv_train, yv_test,
        ya_train, ya_test,
        X_train_ids, X_test_ids
    )