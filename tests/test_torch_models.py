import importlib.util
import unittest

import numpy as np


class TestMLPClassifierTorch(unittest.TestCase):
    def test_configured_learning_rate_is_used_by_optimizer(self):
        try:
            from neureptrace.decoding.torch_models import MLPClassifierTorch
        except ImportError as exc:
            self.skipTest(f"PyTorch dependencies are not installed: {exc}")

        model = MLPClassifierTorch(
            input_dim=2,
            hidden_dim=3,
            output_dim=2,
            learning_rate=0.123,
        )

        optimizer = model.configure_optimizers()

        self.assertEqual(optimizer.param_groups[0]["lr"], 0.123)

    def test_seeded_data_loaders_are_reproducible(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch dependencies are not installed: No module named 'torch'")
        from neureptrace.decoding.classifiers import _build_pytorch_data_loaders

        features = np.arange(40).reshape(20, 2)
        labels = np.arange(20)

        first_loaders = _build_pytorch_data_loaders(features, labels, random_seed=123)
        second_loaders = _build_pytorch_data_loaders(features, labels, random_seed=123)

        self.assertEqual(
            self._labels_by_loader(first_loaders),
            self._labels_by_loader(second_loaders),
        )

    def _labels_by_loader(self, loaders):
        return [[batch_labels.tolist() for _, batch_labels in loader] for loader in loaders]


if __name__ == "__main__":
    unittest.main()
