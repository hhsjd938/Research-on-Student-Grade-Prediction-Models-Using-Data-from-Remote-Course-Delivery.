import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import shap
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from .common import classification_metrics, load_dataset, save_json


def configure_font():
    installed = {item.name for item in font_manager.fontManager.ttflist}
    for family in ("Microsoft JhengHei", "PingFang TC", "Noto Sans CJK TC", "Noto Sans CJK JP", "WenQuanYi Zen Hei"):
        if family in installed:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams["axes.unicode_minus"] = False


def analyze(data_path, output_dir, seed=42, max_samples=300):
    if max_samples < 1:
        raise ValueError("max_samples must be positive")

    X, y = load_dataset(data_path, "及格與否")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                          tree_method="hist", random_state=seed, n_jobs=1)),
        ]
    )
    weights = compute_sample_weight(class_weight="balanced", y=y_train)
    model.fit(X_train, y_train, classifier__sample_weight=weights)
    predicted = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, predicted, probabilities)
    metrics.update({"n_train": len(X_train), "n_test": len(X_test), "features": X.columns.tolist()})

    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / "test_metrics.json", metrics)
    configure_font()

    sample = X_test.sample(n=min(max_samples, len(X_test)), random_state=seed)
    transformed = pd.DataFrame(
        model.named_steps["imputer"].transform(sample),
        columns=X.columns,
        index=sample.index,
    )
    explainer = shap.TreeExplainer(model.named_steps["classifier"])
    explanation = explainer(transformed)
    if explanation.values.ndim == 3:
        base = explanation.base_values
        if np.asarray(base).ndim == 2:
            base = np.asarray(base)[:, 1]
        explanation = shap.Explanation(
            values=explanation.values[:, :, 1],
            base_values=base,
            data=transformed.to_numpy(),
            feature_names=X.columns.tolist(),
        )

    importance = pd.DataFrame(
        {"feature": X.columns, "mean_absolute_shap": np.abs(explanation.values).mean(axis=0)}
    ).sort_values("mean_absolute_shap", ascending=False)
    importance.to_csv(directory / "feature_importance.csv", index=False)

    shap.plots.beeswarm(explanation, show=False)
    plt.tight_layout()
    plt.savefig(directory / "shap_summary.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    shap.plots.waterfall(explanation[0], show=False)
    plt.savefig(directory / "shap_waterfall.png", dpi=160, bbox_inches="tight")
    plt.close("all")

    top_feature = importance.iloc[0]["feature"]
    shap.plots.scatter(explanation[:, top_feature], show=False)
    plt.savefig(directory / "shap_dependence.png", dpi=160, bbox_inches="tight")
    plt.close("all")
    return metrics


def main():
    parser = argparse.ArgumentParser(description="XGBoost pass/fail prediction with SHAP explanations")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/shap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=300)
    args = parser.parse_args()
    result = analyze(args.data, args.output, args.seed, args.max_samples)
    print(pd.Series({key: value for key, value in result.items() if key not in {"features", "confusion_matrix"}}))
    print("Confusion matrix [0, 1]:", result["confusion_matrix"])


if __name__ == "__main__":
    main()
