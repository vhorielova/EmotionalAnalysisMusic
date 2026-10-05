from config import THRESHOLD, NEUTRAL_MARGIN


def get_thresholds(df):
    if THRESHOLD is None:
        valence_threshold = df["valence_mean"].median()
        arousal_threshold = df["arousal_mean"].median()
    else:
        valence_threshold = THRESHOLD
        arousal_threshold = THRESHOLD
    print("Thresholds:", valence_threshold, arousal_threshold)
    return valence_threshold, arousal_threshold


def remove_neutral_zone(df, valence_threshold, arousal_threshold):
    if NEUTRAL_MARGIN <= 0:
        return df

    far_from_border = (
            ((df["valence_mean"] - valence_threshold).abs() >= NEUTRAL_MARGIN)
            & ((df["arousal_mean"] - arousal_threshold).abs() >= NEUTRAL_MARGIN)
    )
    print(f"Songs kept after removing neutral zone: {far_from_border.sum()} of {len(df)}")
    return df[far_from_border].reset_index(drop=True)


def get_emotion(row, valence_threshold, arousal_threshold):
    if (row["valence_mean"] < valence_threshold):
        if (row["arousal_mean"] < arousal_threshold):
            return 0
        else:
            return 1
    else:
        if (row["arousal_mean"] < arousal_threshold):
            return 2
        else:
            return 3


def add_labels(df, valence_threshold, arousal_threshold):
    df["emotion"] = df.apply(get_emotion, axis=1, args=(valence_threshold, arousal_threshold))

    df["valence_high"] = (df["valence_mean"] >= valence_threshold).astype(int)
    df["arousal_high"] = (df["arousal_mean"] >= arousal_threshold).astype(int)

    print(df[["song_id", "valence_mean", "arousal_mean", "emotion"]].head())
    print(df["emotion"].value_counts().sort_index())
    print(df["emotion"].value_counts(normalize=True).sort_index().round(3))
    return df