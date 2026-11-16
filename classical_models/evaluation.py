from sklearn.model_selection import GridSearchCV
import pandas as pd

from config import CV_FOLDS


from sklearn.metrics import (
    classification_report,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)


def run_all_models(
        models,
        X_train,
        X_test,
        y_train,
        y_test,
        target_names,
        task_name
):
    print("\n" + "=" * 70)
    print("TASK:", task_name)
    print("=" * 70)

    predictions = {}
    results = []

    for name, (pipeline, param_grid) in models.items():

        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=CV_FOLDS,
            scoring="f1_macro",
            n_jobs=-1
        )

        search.fit(X_train, y_train)

        y_pred = search.predict(X_test)
        predictions[name] = y_pred

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )
        recall = recall_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )
        f1 = f1_score(
            y_test, y_pred,
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

        print(f"\n--- {name} ---")
        print("Best settings:", search.best_params_)
        print(f"Cross-validation macro-F1: {search.best_score_:.3f}")

        print(
            f"Test Accuracy: {accuracy:.3f} | "
            f"Test Macro-F1: {f1:.3f}"
        )

        print(
            classification_report(
                y_test,
                y_pred,
                target_names=target_names,
                zero_division=0.0
            )
        )

    results_df = pd.DataFrame(results)

    print("\n")
    print("=" * 70)
    print(f"RESULTS: {task_name}")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.3f}".format,
                "Precision": "{:.3f}".format,
                "Recall": "{:.3f}".format,
                "F1-Score": "{:.3f}".format
            }
        )
    )

    return predictions, results_df


def evaluate_combined(models, pred_valence, pred_arousal, y_test):

    results = []

    print("\n" + "=" * 70)
    print("TASK: 4 emotions built from valence and arousal")
    print("=" * 70)

    for name in models:

        y_pred_combined = (
            2 * pred_valence[name]
            + pred_arousal[name]
        )

        accuracy = accuracy_score(
            y_test,
            y_pred_combined
        )

        precision = precision_score(
            y_test,
            y_pred_combined,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred_combined,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred_combined,
            average="macro",
            zero_division=0
        )

        results.append({
            "Model": name,
            "Accuracy": accuracy,
            "Precision": precision,
            "Recall": recall,
            "F1-Score": f1
        })

    results_df = pd.DataFrame(results)

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.3f}".format,
                "Precision": "{:.3f}".format,
                "Recall": "{:.3f}".format,
                "F1-Score": "{:.3f}".format
            }
        )
    )

    return results_df