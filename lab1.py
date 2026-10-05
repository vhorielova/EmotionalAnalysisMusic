import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


dataset_path = kagglehub.dataset_download(
  "imsparsh/deam-mediaeval-dataset-emotional-analysis-in-music"
)

annotations_path_1 = Path(dataset_path) / "DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_1_2000.csv"
annotations_path_2 = Path(dataset_path) / "DEAM_Annotations/annotations/annotations averaged per song/song_level/static_annotations_averaged_songs_2000_2058.csv"

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

print("First 5 records:\n", df.head())

print(df["valence_mean"].describe())
print(df["arousal_mean"].describe())

plt.figure(figsize=(8, 6))

plt.scatter(
    df["valence_mean"],
    df["arousal_mean"],
    alpha=0.5
)

plt.xlabel("Valence")
plt.ylabel("Arousal")
plt.title("Distribution of Music Emotions")


print(df.isna().sum().sum())

valence_threshold = df["valence_mean"].median()
arousal_threshold = df["arousal_mean"].median()
print(valence_threshold, arousal_threshold)

def get_emotion(row):
    if(row["valence_mean"] < valence_threshold):
        if(row["arousal_mean"] < arousal_threshold):
            return 0
        else:
            return 1
    else:
        if (row["arousal_mean"] < arousal_threshold):
            return 2
        else:
            return 3

df["emotion"] = df.apply(get_emotion, axis = 1)

print(df[["song_id", "valence_mean", "arousal_mean", "emotion"]].head())

print(df["emotion"].value_counts().sort_index())

print(df["emotion"].value_counts(normalize=True).sort_index().round(3))


X = df.drop(columns=["song_id", "valence_mean", "arousal_mean", "emotion", "valence_std", "arousal_std"])
y = df["emotion"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model_randforest = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    class_weight="balanced",
    random_state=42
)

model_svc = svm.SVC(
    kernel="rbf",
    C=2.0,
    class_weight="balanced"
)

model_svc.fit(X_train, y_train)
model_randforest.fit(X_train, y_train)

y_pred_svc = model_svc.predict(X_test)
y_pred_randforest = model_randforest.predict(X_test)

print(X_test[0:5])

print(y_test[0:5])
print(y_pred_svc[0:5])
print(y_pred_randforest[0:5])

target_names = ["sad/low energy", "angry/tense", "calm/relaxed", "happy/excited"]
print(classification_report(y_test, y_pred_svc, target_names=target_names, zero_division=0.0))
print(classification_report(y_test, y_pred_randforest, target_names=target_names, zero_division=0.0))

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_svc, display_labels=target_names)
plt.show()

disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred_randforest, display_labels=target_names)
plt.show()


