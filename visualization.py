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

def plot_results_table(results_df, title):

    display_df = results_df.copy()

    for column in ["Accuracy", "Precision", "Recall", "F1-Score"]:
        display_df[column] = display_df[column].map(
            lambda x: f"{x:.3f}"
        )

    fig, ax = plt.subplots(
        figsize=(12, 3.5)
    )

    ax.axis("off")

    table = ax.table(
        cellText=display_df.values,
        colLabels=display_df.columns,
        loc="center",
        cellLoc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)

    ax.set_title(
        title,
        fontsize=14,
        pad=20
    )

    plt.tight_layout()
    plt.show()