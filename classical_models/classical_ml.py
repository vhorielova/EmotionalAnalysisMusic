import matplotlib
matplotlib.use("macosx")
import matplotlib.pyplot as plt

from config import EMOTION_NAMES
from data_loading import load_dataset, describe_data, split_data
from labels import get_thresholds, remove_neutral_zone, add_labels
from classical_models.models import build_models
from classical_models.evaluation import run_all_models, evaluate_combined
from visualization import plot_emotion_distribution, plot_confusion_matrices, plot_results_table

from reporting.reporting import save_predictions

def safe_name(name):
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")

def run_classical_ml(X_train, X_test, y_train, y_test,
                     yv_train, yv_test, ya_train, ya_test,
                     X_test_ids, suffix):
    models = build_models()

    pred_emotion, results_emotion = run_all_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        EMOTION_NAMES,
        "4 emotions"
    )

    for name, y_pred in pred_emotion.items():
        save_predictions(f"{safe_name(name)}_{suffix}",
                         dict(zip(X_test_ids.tolist(), y_pred.tolist())))

    plot_results_table(
        results_emotion,
        "Comparison of models — 4 emotions"
    )

    pred_arousal, results_arousal = run_all_models(
        models,
        X_train,
        X_test,
        ya_train,
        ya_test,
        ["low arousal", "high arousal"],
        "arousal (low / high)"
    )

    plot_results_table(
        results_arousal,
        "Comparison of models — Arousal"
    )

    pred_valence, results_valence = run_all_models(
        models,
        X_train,
        X_test,
        yv_train,
        yv_test,
        ["low valence", "high valence"],
        "valence (low / high)"
    )

    plot_results_table(
        results_valence,
        "Comparison of models — Valence"
    )

    results_combined = evaluate_combined(
        models,
        pred_valence,
        pred_arousal,
        y_test
    )

    plot_results_table(
        results_combined,
        "4 emotions from Valence + Arousal"
    )

    # plot_confusion_matrices(pred_emotion, y_test, ["SVM", "Random forest"])
    plt.show()