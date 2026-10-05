import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

from config import EMOTION_NAMES
from data_loading import load_dataset, describe_data, split_data
from labels import get_thresholds, remove_neutral_zone, add_labels
from models import build_models
from evaluation import run_all_models, evaluate_combined
from visualization import plot_emotion_distribution, plot_confusion_matrices

def main():
    # 1. data
    df = load_dataset()
    describe_data(df)

    # 2. labels
    valence_threshold, arousal_threshold = get_thresholds(df)
    plot_emotion_distribution(df, valence_threshold, arousal_threshold)

    df = remove_neutral_zone(df, valence_threshold, arousal_threshold)
    plot_emotion_distribution(df, valence_threshold, arousal_threshold)

    df = add_labels(df, valence_threshold, arousal_threshold)

    # 3. train / test split
    X_train, X_test, y_train, y_test, yv_train, yv_test, ya_train, ya_test = split_data(df)

    # 4. models
    models = build_models()

    # 5. experiments
    # a) your original task: 4 emotions at once
    pred_emotion = run_all_models(models, X_train, X_test, y_train, y_test,
                                  EMOTION_NAMES, "4 emotions")

    # b) two easier tasks: only arousal, only valence
    pred_arousal = run_all_models(models, X_train, X_test, ya_train, ya_test,
                                  ["low arousal", "high arousal"], "arousal (low / high)")
    pred_valence = run_all_models(models, X_train, X_test, yv_train, yv_test,
                                  ["low valence", "high valence"], "valence (low / high)")

    # c) 4 emotions built from the two binary answers
    evaluate_combined(models, pred_valence, pred_arousal, y_test)

    # 6. confusion matrices
    plot_confusion_matrices(pred_emotion, y_test, ["SVM", "Random forest"])
    plt.show()


if __name__ == "__main__":
    main()
