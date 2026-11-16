from cnn_and_ast.cnn.cnn import run_cnn
from classical_models.classical_ml import run_classical_ml
from data_loading import load_dataset, describe_data, split_data
from labels import get_thresholds, add_labels, remove_neutral_zone
from visualization import plot_emotion_distribution
from cnn_and_ast.ast.ast import run_ast
from reporting.reporting import make_report

def main():
    df = load_dataset()
    describe_data(df)

    valence_threshold, arousal_threshold = get_thresholds(df)
    plot_emotion_distribution(df, valence_threshold, arousal_threshold)

    plot_emotion_distribution(df, valence_threshold, arousal_threshold)

    df = add_labels(df, valence_threshold, arousal_threshold)

    X_train, X_test, y_train, y_test, yv_train, yv_test, ya_train, ya_test, X_train_ids, X_test_ids = split_data(df, valence_threshold, arousal_threshold)

    # run_classical_ml(X_train, X_test, y_train, y_test, yv_train, yv_test, ya_train, ya_test, X_test_ids, "full")
    #
    # run_cnn(df, X_train_ids, X_test_ids)
    #
    # run_ast(df, X_train_ids, X_test_ids)

    truth = df.set_index("song_id")["emotion"].to_dict()
    test_truth = {i: truth[i] for i in X_test_ids.tolist()}
    make_report(test_truth, reference="random_forest_full")



if __name__ == "__main__":
    main()
