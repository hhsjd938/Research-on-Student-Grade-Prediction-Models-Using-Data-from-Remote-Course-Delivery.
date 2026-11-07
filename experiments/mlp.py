import argparse
from copy import deepcopy
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .common import classification_metrics, load_dataset, save_json


class MLPModel(nn.Module):
    def __init__(self, input_dim, hidden_sizes=(100, 50), dropout=0.2):
        super().__init__()
        first, second = hidden_sizes
        self.network = nn.Sequential(
            nn.Linear(input_dim, first),
            nn.BatchNorm1d(first),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(first, second),
            nn.BatchNorm1d(second),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(second, 1),
        )

    def forward(self, features):
        return self.network(features).squeeze(-1)


def configure_font():
    installed = {entry.name for entry in font_manager.fontManager.ttflist}
    for family in (
        "Microsoft JhengHei",
        "PingFang TC",
        "Noto Sans CJK TC",
        "Noto Sans CJK JP",
        "WenQuanYi Zen Hei",
    ):
        if family in installed:
            plt.rcParams["font.sans-serif"] = [family]
            break
    plt.rcParams["axes.unicode_minus"] = False


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def select_device(requested):
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    return torch.device(requested)


def as_tensor(values, device):
    return torch.as_tensor(np.asarray(values, dtype=np.float32), device=device)


def predict_probabilities(model, features, device, batch_size=4096):
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = as_tensor(features[start:start + batch_size], device)
            probabilities = torch.sigmoid(model(batch)).cpu().numpy()
            predictions.append(probabilities)
    return np.concatenate(predictions)


def select_threshold(labels, probabilities):
    thresholds = np.linspace(0.05, 0.95, 19)
    results = [
        (float(f1_score(labels, probabilities >= threshold, average="macro", zero_division=0)), float(threshold))
        for threshold in thresholds
    ]
    score, threshold = max(results, key=lambda value: (value[0], -abs(value[1] - 0.5)))
    return threshold, score


def permutation_importance(model, features, labels, feature_names, threshold, repeats, seed, device):
    baseline = f1_score(
        labels,
        predict_probabilities(model, features, device) >= threshold,
        average="macro",
        zero_division=0,
    )
    generator = np.random.default_rng(seed)
    importance = []
    for index, name in enumerate(feature_names):
        differences = []
        for _ in range(repeats):
            shuffled = features.copy()
            shuffled[:, index] = generator.permutation(shuffled[:, index])
            predicted = predict_probabilities(model, shuffled, device) >= threshold
            score = f1_score(labels, predicted, average="macro", zero_division=0)
            differences.append(baseline - score)
        importance.append(
            {
                "feature": name,
                "f1_macro_decrease_mean": float(np.mean(differences)),
                "f1_macro_decrease_std": float(np.std(differences)),
            }
        )
    return pd.DataFrame(importance).sort_values("f1_macro_decrease_mean", ascending=False)


def train(
    data_path,
    output_dir,
    seed=42,
    epochs=600,
    batch_size=128,
    learning_rate=0.0001,
    patience=25,
    dropout=0.2,
    test_size=0.2,
    validation_size=0.2,
    importance_repeats=3,
    device="auto",
):
    if epochs < 1 or patience < 1 or batch_size < 2 or learning_rate <= 0:
        raise ValueError("epochs, patience, batch_size and learning_rate must be positive; batch_size must be at least 2")
    if not (0 < test_size < 1 and 0 < validation_size < 1):
        raise ValueError("test_size and validation_size must be between 0 and 1")
    if not 0 <= dropout < 1 or importance_repeats < 0:
        raise ValueError("dropout must be in [0, 1) and importance_repeats must be nonnegative")

    set_seed(seed)
    configure_font()
    compute_device = select_device(device)
    features, labels = load_dataset(data_path, "及格與否")
    train_val_x, test_x, train_val_y, test_y = train_test_split(
        features, labels, test_size=test_size, random_state=seed, stratify=labels
    )
    train_x, validation_x, train_y, validation_y = train_test_split(
        train_val_x,
        train_val_y,
        test_size=validation_size,
        random_state=seed,
        stratify=train_val_y,
    )
    if len(train_x) < 2:
        raise ValueError("The training split must contain at least two rows")

    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    scaler = StandardScaler()
    train_values = scaler.fit_transform(imputer.fit_transform(train_x)).astype(np.float32)
    validation_values = scaler.transform(imputer.transform(validation_x)).astype(np.float32)
    test_values = scaler.transform(imputer.transform(test_x)).astype(np.float32)

    train_labels = train_y.to_numpy(dtype=np.float32)
    validation_labels = validation_y.to_numpy(dtype=np.float32)
    class_counts = np.bincount(train_y.to_numpy(dtype=np.int64), minlength=2)
    class_weights = len(train_y) / (2.0 * class_counts)

    train_dataset = TensorDataset(
        torch.from_numpy(train_values), torch.from_numpy(train_labels)
    )
    batch_generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=batch_generator,
        drop_last=len(train_dataset) > batch_size and len(train_dataset) % batch_size == 1,
    )

    model = MLPModel(train_values.shape[1], dropout=dropout).to(compute_device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.BCEWithLogitsLoss(reduction="none")
    validation_criterion = nn.BCEWithLogitsLoss()
    weight_tensor = as_tensor(class_weights, compute_device)
    validation_tensor = as_tensor(validation_values, compute_device)
    validation_target = as_tensor(validation_labels, compute_device)
    best_state = deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        weighted_losses = []
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(compute_device)
            batch_y = batch_y.to(compute_device)
            optimizer.zero_grad()
            logits = model(batch_x)
            sample_weights = torch.where(batch_y >= 0.5, weight_tensor[1], weight_tensor[0])
            loss = (criterion(logits, batch_y) * sample_weights).mean()
            loss.backward()
            optimizer.step()
            weighted_losses.append((loss.item(), len(batch_y)))

        model.eval()
        with torch.no_grad():
            validation_logits = model(validation_tensor)
            validation_loss = validation_criterion(validation_logits, validation_target).item()
        training_loss = sum(value * count for value, count in weighted_losses) / sum(
            count for _, count in weighted_losses
        )
        history.append(
            {"epoch": epoch, "training_weighted_loss": training_loss, "validation_loss": validation_loss}
        )
        if validation_loss < best_loss - 1e-6:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                break

    model.load_state_dict(best_state)
    validation_probs = predict_probabilities(model, validation_values, compute_device)
    threshold, validation_f1 = select_threshold(validation_y, validation_probs)
    test_probs = predict_probabilities(model, test_values, compute_device)
    predictions = (test_probs >= threshold).astype(int)
    metrics = classification_metrics(test_y, predictions, test_probs)
    metrics.update(
        {
            "selection_metric": "minimum_unweighted_validation_bce_loss",
            "threshold_metric": "maximum_validation_f1_macro",
            "classification_threshold": threshold,
            "validation_f1_macro": validation_f1,
            "best_validation_loss": best_loss,
            "best_epoch": best_epoch,
            "epochs_trained": len(history),
            "n_train": len(train_y),
            "n_validation": len(validation_y),
            "n_test": len(test_y),
            "features": features.columns.tolist(),
            "architecture": {"hidden_sizes": [100, 50], "dropout": dropout},
            "loss_weighting": "inverse_class_frequency_on_training_set",
            "device": str(compute_device),
            "seed": seed,
        }
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    save_json(output / "test_metrics.json", metrics)
    pd.DataFrame(history).to_csv(output / "training_history.csv", index=False)
    joblib.dump(
        {"imputer": imputer, "scaler": scaler, "features": features.columns.tolist()},
        output / "preprocessing.joblib",
    )
    torch.save(
        {
            "state_dict": model.cpu().state_dict(),
            "input_dim": train_values.shape[1],
            "hidden_sizes": [100, 50],
            "dropout": dropout,
            "threshold": threshold,
        },
        output / "model.pt",
    )

    if len(np.unique(test_y)) == 2:
        fpr, tpr, _ = roc_curve(test_y, test_probs)
        fig, axis = plt.subplots()
        axis.plot(fpr, tpr, label=f"ROC-AUC: {roc_auc_score(test_y, test_probs):.3f}")
        axis.plot([0, 1], [0, 1], linestyle="--", color="gray")
        axis.set(xlabel="False positive rate", ylabel="True positive rate", title="MLP ROC curve")
        axis.legend()
        fig.savefig(output / "roc_curve.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    if importance_repeats:
        scores = permutation_importance(
            model.to(compute_device),
            validation_values,
            validation_y.to_numpy(),
            features.columns.tolist(),
            threshold,
            importance_repeats,
            seed,
            compute_device,
        )
        scores.to_csv(output / "feature_importance.csv", index=False)
        top = scores.head(20).iloc[::-1]
        fig, axis = plt.subplots(figsize=(9, 7))
        axis.barh(top["feature"], top["f1_macro_decrease_mean"])
        axis.set(xlabel="Validation macro-F1 decrease", title="Permutation feature importance")
        fig.savefig(output / "feature_importance.png", dpi=160, bbox_inches="tight")
        plt.close(fig)

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Student pass/fail prediction using a PyTorch MLP")
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="results/mlp")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.0001)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.2)
    parser.add_argument("--importance-repeats", type=int, default=3)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    arguments = parser.parse_args()
    result = train(
        arguments.data,
        arguments.output,
        seed=arguments.seed,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        patience=arguments.patience,
        dropout=arguments.dropout,
        test_size=arguments.test_size,
        validation_size=arguments.validation_size,
        importance_repeats=arguments.importance_repeats,
        device=arguments.device,
    )
    summary_keys = [
        "accuracy", "balanced_accuracy", "f1_macro", "roc_auc", "recall_fail", "recall_pass",
        "classification_threshold", "best_epoch", "n_train", "n_validation", "n_test",
    ]
    print(pd.Series({key: result[key] for key in summary_keys}))
    print("Confusion matrix [0, 1]:", result["confusion_matrix"])


if __name__ == "__main__":
    main()
