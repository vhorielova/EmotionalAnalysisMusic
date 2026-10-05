import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from config import EMOTION_NAMES


def plot_emotion_distribution(df, valence_threshold, arousal_threshold):
    plt.figure(figsize=(8, 6))
    plt.scatter(df["valence_mean"], df["arousal_mean"], alpha=0.5)
    plt.axvline(valence_threshold, color="k", linestyle="--")
    plt.axhline(arousal_threshold, color="k", linestyle="--")
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Distribution of Music Emotions")


def plot_confusion_matrices(pred_emotion, y_test, model_names):
    for name in model_names:
        disp = ConfusionMatrixDisplay.from_predictions(
            y_test, pred_emotion[name], display_labels=EMOTION_NAMES, normalize="true"
        )
        disp.ax_.set_title(name)