from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


INVALID_VALUES = ["#NULL!", "#REF!", "??", "", " "]
TARGET_COLUMNS = {"課程成績", "及格與否", "分數等級"}
IDENTIFIER_COLUMNS = {"cou_code"}


def load_dataset(path, target):
    if target not in {"課程成績", "及格與否"}:
        raise ValueError(f"Unsupported target: {target}")

    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(file_path)

    data = None
    for encoding in ("utf-8-sig", "big5", "cp950", "latin1"):
        try:
            data = pd.read_csv(file_path, encoding=encoding, na_values=INVALID_VALUES)
            if target in data.columns:
                break
        except UnicodeDecodeError:
            continue

    if data is None or target not in data.columns:
        raise ValueError(f"The CSV must contain the target column: {target}")

    labels = pd.to_numeric(data[target], errors="coerce")
    valid_rows = labels.notna()
    if target == "及格與否":
        if not set(labels[valid_rows].unique()).issubset({0, 1}):
            raise ValueError("及格與否 must contain only 0 and 1")
        if labels[valid_rows].nunique() != 2:
            raise ValueError("及格與否 must contain both classes")
        labels = labels.astype("Int64")

    excluded = TARGET_COLUMNS | IDENTIFIER_COLUMNS
    features = data.drop(columns=list(excluded), errors="ignore")
    features = features.apply(pd.to_numeric, errors="coerce")
    features = features.loc[:, features.notna().any(axis=0)]
    if features.empty:
        raise ValueError("No usable numeric features")

    features = features.loc[valid_rows].reset_index(drop=True)
    labels = labels.loc[valid_rows].reset_index(drop=True)
    if target == "及格與否":
        labels = labels.astype(int)
    else:
        labels = labels.astype(float)
    return features, labels


def classification_metrics(y_true, predicted, probabilities):
    matrix = confusion_matrix(y_true, predicted, labels=[0, 1])
    result = {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "precision_macro": float(precision_score(y_true, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, predicted, average="macro", zero_division=0)),
        "recall_fail": float(recall_score(y_true, predicted, pos_label=0, zero_division=0)),
        "recall_pass": float(recall_score(y_true, predicted, pos_label=1, zero_division=0)),
        "confusion_matrix_labels": [0, 1],
        "confusion_matrix": matrix.tolist(),
    }
    result["roc_auc"] = float(roc_auc_score(y_true, probabilities)) if len(np.unique(y_true)) == 2 else None
    return result


def save_json(path, payload):
    import json

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
