import argparse
import math
from pathlib import Path

import joblib
import pandas as pd
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor

from .common import load_dataset, save_json


def candidate_models(seed):
    return {
        "random_forest": (
            RandomForestRegressor(random_state=seed, n_jobs=1),
            {"n_estimators": [100, 200, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3]},
        ),
        "gradient_boosting": (
            GradientBoostingRegressor(random_state=seed),
            {"n_estimators": [100, 200, 300], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [2, 3, 5]},
        ),
        "adaboost": (
            AdaBoostRegressor(random_state=seed),
            {"n_estimators": [50, 100, 200], "learning_rate": [0.03, 0.1, 0.5]},
        ),
        "xgboost": (
            XGBRegressor(random_state=seed, n_jobs=1, tree_method="hist"),
            {"n_estimators": [100, 300, 500], "learning_rate": [0.03, 0.05, 0.1], "max_depth": [3, 5, 7]},
        ),
        "lightgbm": (
            LGBMRegressor(random_state=seed, verbosity=-1, n_jobs=1),
            {"n_estimators": [100, 300, 500], "learning_rate": [0.03, 0.05, 0.1], "num_leaves": [15, 31, 63]},
        ),
        "catboost": (
            CatBoostRegressor(random_seed=seed, verbose=False, thread_count=1, allow_writing_files=False),
            {"iterations": [300, 600, 900], "learning_rate": [0.03, 0.05, 0.1], "depth": [4, 6, 8]},
        ),
        "ridge": (Ridge(), {"alpha": [0.01, 0.1, 1, 10, 100]}),
        "lasso": (Lasso(max_iter=20000, random_state=seed), {"alpha": [0.001, 0.01, 0.1, 1, 10]}),
        "svr": (SVR(), {"C": [0.1, 1, 10], "epsilon": [0.1, 0.5, 1], "kernel": ["rbf", "linear"]}),
    }


def train(data_path, output_dir, models=None, seed=42, cv_folds=5, n_iter=10, n_jobs=-1):
    if cv_folds < 2 or n_iter < 1:
        raise ValueError("cv_folds must be at least 2 and n_iter must be positive")

    X, y = load_dataset(data_path, "課程成績")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    candidates = candidate_models(seed)
    names = models or list(candidates)
    unknown = set(names) - set(candidates)
    if unknown:
        raise ValueError(f"Unknown models: {sorted(unknown)}")

    validation = KFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scores = []
    best_search = None
    best_name = None

    for name in names:
        estimator, parameters = candidates[name]
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if name in {"ridge", "lasso", "svr"}:
            steps.append(("scaler", StandardScaler()))
        steps.append(("model", estimator))
        pipeline = Pipeline(steps)
        search = RandomizedSearchCV(
            pipeline,
            param_distributions={f"model__{key}": value for key, value in parameters.items()},
            n_iter=min(n_iter, math.prod(len(value) for value in parameters.values())),
            scoring="r2",
            cv=validation,
            n_jobs=n_jobs,
            random_state=seed,
            error_score="raise",
            refit=True,
        )
        search.fit(X_train, y_train)
        scores.append({"model": name, "cv_r2": float(search.best_score_), "best_params": search.best_params_})
        if best_search is None or search.best_score_ > best_search.best_score_:
            best_search = search
            best_name = name

    predictions = best_search.predict(X_test)
    metrics = {
        "selected_model": best_name,
        "selection_metric": "cross_validation_r2_on_training_set",
        "cv_r2": float(best_search.best_score_),
        "test_r2": float(r2_score(y_test, predictions)),
        "test_mae": float(mean_absolute_error(y_test, predictions)),
        "test_mse": float(mean_squared_error(y_test, predictions)),
        "test_rmse": float(mean_squared_error(y_test, predictions) ** 0.5),
        "test_mape": float(mean_absolute_percentage_error(y_test, predictions)),
        "best_params": best_search.best_params_,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "features": X.columns.tolist(),
    }

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(scores).sort_values("cv_r2", ascending=False).to_csv(directory / "cv_results.csv", index=False)
    save_json(directory / "test_metrics.json", metrics)
    joblib.dump(best_search.best_estimator_, directory / "best_model.joblib")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Student grade regression and hyperparameter search")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/regression")
    parser.add_argument("--models", nargs="+", choices=sorted(candidate_models(42)))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--n-iter", type=int, default=10)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    result = train(args.data, args.output, args.models, args.seed, args.cv, args.n_iter, args.n_jobs)
    print(pd.Series({key: value for key, value in result.items() if key not in {"best_params", "features"}}))


if __name__ == "__main__":
    main()
