import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import load_dataset
from experiments.regression import train as train_regression


class WorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(42)
        n = 120
        activity = rng.normal(size=n)
        homework = rng.normal(size=n)
        grade = 72 + activity * 10 + homework * 5 + rng.normal(size=n) * 5
        dataset = pd.DataFrame(
            {
                "學年": rng.integers(110, 113, n),
                "cou_code": [f"C{i % 6}" for i in range(n)],
                "造訪次數": 100 + 20 * activity,
                "作業繳交率(%)": 60 + 15 * homework,
                "課程成績": grade,
                "及格與否": (grade >= 68).astype(int),
                "分數等級": np.floor(grade / 10),
            }
        )
        dataset.loc[0, "造訪次數"] = None
        self.path = self.root / "example.csv"
        dataset.to_csv(self.path, index=False)

    def tearDown(self):
        self.temporary.cleanup()

    def test_targets_and_identifiers_are_excluded(self):
        for target in ("課程成績", "及格與否"):
            features, labels = load_dataset(self.path, target)
            self.assertEqual(len(features), len(labels))
            self.assertTrue(set(features.columns).isdisjoint({"課程成績", "及格與否", "分數等級", "cou_code"}))
            self.assertIn("造訪次數", features.columns)

    def test_regression_workflow(self):
        result = train_regression(self.path, self.root / "results", models=["ridge"], cv_folds=2, n_iter=1, n_jobs=1)
        self.assertEqual(result["selected_model"], "ridge")
        self.assertTrue((self.root / "results" / "test_metrics.json").is_file())
        self.assertTrue((self.root / "results" / "best_model.joblib").is_file())


if __name__ == "__main__":
    unittest.main()
