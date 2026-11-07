import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.mlp import MLPModel, select_threshold, train


class MLPWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        generator = np.random.default_rng(7)
        n = 160
        activity = generator.normal(size=n)
        homework = generator.normal(size=n)
        logits = 0.7 * activity + homework + generator.normal(size=n)
        is_pass = (logits > 0.2).astype(int)
        frame = pd.DataFrame(
            {
                "造訪次數": 40 + 10 * activity,
                "作業繳交率(%)": 70 + 5 * homework,
                "cou_code": [f"C{i % 8}" for i in range(n)],
                "課程成績": is_pass * 30 + 50,
                "及格與否": is_pass,
                "分數等級": is_pass + 2,
            }
        )
        frame.loc[0, "造訪次數"] = np.nan
        self.dataset_path = self.root / "students.csv"
        frame.to_csv(self.dataset_path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_threshold_is_selected_from_supplied_labels(self):
        threshold, score = select_threshold(
            np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.6, 0.9])
        )
        self.assertAlmostEqual(score, 1.0)
        self.assertAlmostEqual(threshold, 0.5)

    def test_training_and_artifacts(self):
        torch.set_num_threads(1)
        destination = self.root / "output"
        metrics = train(
            self.dataset_path,
            destination,
            seed=17,
            epochs=6,
            batch_size=16,
            patience=3,
            importance_repeats=1,
            device="cpu",
        )
        self.assertEqual(metrics["architecture"]["hidden_sizes"], [100, 50])
        self.assertTrue(0 < metrics["classification_threshold"] < 1)
        self.assertNotIn("課程成績", metrics["features"])
        self.assertNotIn("cou_code", metrics["features"])
        self.assertEqual(sum(map(sum, metrics["confusion_matrix"])), metrics["n_test"])
        for name in (
            "model.pt", "preprocessing.joblib", "test_metrics.json", "training_history.csv",
            "roc_curve.png", "feature_importance.csv", "feature_importance.png",
        ):
            self.assertTrue((destination / name).is_file(), name)
        checkpoint = torch.load(destination / "model.pt", map_location="cpu", weights_only=True)
        self.assertEqual(checkpoint["input_dim"], len(metrics["features"]))
        self.assertIsInstance(MLPModel(checkpoint["input_dim"]), torch.nn.Module)


if __name__ == "__main__":
    unittest.main()
