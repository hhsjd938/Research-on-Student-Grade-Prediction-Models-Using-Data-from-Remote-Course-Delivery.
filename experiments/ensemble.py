import argparse
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from .common import classification_metrics, load_dataset, save_json


def voting_model(seed):
    estimators = [
        ("decision_tree", DecisionTreeClassifier(random_state=seed, class_weight="balanced")),
        ("random_forest", RandomForestClassifier(n_estimators=150, random_state=seed, class_weight="balanced", n_jobs=1)),
        ("adaboost", AdaBoostClassifier(random_state=seed)),
        ("gradient_boosting", GradientBoostingClassifier(random_state=seed)),
        ("knn", KNeighborsClassifier(n_neighbors=5)),
        ("logistic_regression", LogisticRegression(max_iter=3000, class_weight="balanced", random_state=seed)),
        ("svm", SVC(probability=True, class_weight="balanced", random_state=seed)),
        ("xgboost", XGBClassifier(n_estimators=150, tree_method="hist", n_jobs=1, random_state=seed)),
        ("lightgbm", LGBMClassifier(n_estimators=150, verbosity=-1, n_jobs=1, random_state=seed)),
    ]
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("ensemble", VotingClassifier(estimators=estimators, voting="soft", n_jobs=1)),
        ]
    )


def train(data_path, output_dir, seed=42, cv_folds=3, n_iter=8, n_jobs=-1):
    if cv_folds < 2 or n_iter < 1:
        raise ValueError("cv_folds must be at least 2 and n_iter must be positive")

    X, y = load_dataset(data_path, "及格與否")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = voting_model(seed)
    parameters = {
        "ensemble__knn__n_neighbors": [3, 5, 7, 11],
        "ensemble__knn__weights": ["uniform", "distance"],
        "ensemble__knn__p": [1, 2],
    }
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        model,
        parameters,
        n_iter=min(n_iter, 16),
        cv=cv,
        scoring="f1_macro",
        refit=True,
        n_jobs=n_jobs,
        random_state=seed,
        error_score="raise",
    )
    search.fit(X_train, y_train)
    predicted = search.predict(X_test)
    probabilities = search.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, predicted, probabilities)
    metrics.update(
        {
            "selection_metric": "cross_validation_f1_macro_on_training_set",
            "cv_f1_macro": float(search.best_score_),
            "best_params": search.best_params_,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "features": X.columns.tolist(),
        }
    )

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "test_metrics.json", metrics)
    joblib.dump(search.best_estimator_, directory / "best_model.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Soft voting ensemble for pass/fail prediction")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/ensemble")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--n-iter", type=int, default=8)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    result = train(args.data, args.output, args.seed, args.cv, args.n_iter, args.n_jobs)
    print(pd.Series({key: value for key, value in result.items() if key not in {"best_params", "features", "confusion_matrix"}}))
    print("Confusion matrix [0, 1]:", result["confusion_matrix"])


if __name__ == "__main__":
    main()
