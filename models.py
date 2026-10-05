from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier

from config import RANDOM_STATE


def build_models():
    models = {
        "Baseline (always most frequent class)": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", DummyClassifier(strategy="most_frequent"))]),
            {}
        ),
        "Logistic regression": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", LogisticRegression(max_iter=3000, class_weight="balanced"))]),
            {"clf__C": [0.001, 0.01, 0.1, 1]}
        ),
        "SVM": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", SVC(kernel="rbf", class_weight="balanced"))]),
            {"clf__C": [1, 10, 100], "clf__gamma": ["scale", 0.001, 0.01]}
        ),
        "Random forest": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                     random_state=RANDOM_STATE))]),
            {"clf__max_depth": [None, 20], "clf__min_samples_leaf": [1, 3]}
        ),
        "Gradient boosting": (
            Pipeline([("scaler", StandardScaler()),
                      ("clf", HistGradientBoostingClassifier(class_weight="balanced",
                                                             random_state=RANDOM_STATE))]),
            {"clf__learning_rate": [0.05, 0.1], "clf__max_depth": [None, 4]}
        ),
    }
    return models