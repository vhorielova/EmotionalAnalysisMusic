from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, f1_score, accuracy_score

from config import CV_FOLDS


def run_all_models(models, X_train, X_test, y_train, y_test, target_names, task_name):
    print("\n" + "=" * 70)
    print("TASK:", task_name)
    print("=" * 70)

    predictions = {}
    for name, (pipeline, param_grid) in models.items():
        search = GridSearchCV(pipeline, param_grid, cv=CV_FOLDS, scoring="f1_macro", n_jobs=-1)
        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        predictions[name] = y_pred

        print(f"\n--- {name} ---")
        print("Best settings:", search.best_params_)
        print(f"Cross-validation macro-F1 (train): {search.best_score_:.3f}")
        print(f"Test accuracy: {accuracy_score(y_test, y_pred):.3f} | "
              f"Test macro-F1: {f1_score(y_test, y_pred, average='macro'):.3f}")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0.0))

    return predictions


def evaluate_combined(models, pred_valence, pred_arousal, y_test):
    print("\n" + "=" * 70)
    print("TASK: 4 emotions built from the valence and arousal models")
    print("=" * 70)
    for name in models:
        y_pred_combined = 2 * pred_valence[name] + pred_arousal[name]
        print(f"{name:40s} accuracy {accuracy_score(y_test, y_pred_combined):.3f} | "
              f"macro-F1 {f1_score(y_test, y_pred_combined, average='macro'):.3f}")