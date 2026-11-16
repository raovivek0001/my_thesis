"""Configuration inheritance and write-boundary checks."""

import tempfile
import unittest
from pathlib import Path

from pinneik3d.common import DEFAULTS, PROJECT_ROOT, deep_merge, load_config, project_path, validate_config


class ConfigTests(unittest.TestCase):
    def test_nested_merge_is_independent(self):
        config = deep_merge(DEFAULTS, {"training": {"batch_size": 128}})
        self.assertEqual(config["training"]["batch_size"], 128)
        self.assertEqual(config["network"]["hidden_layers"], 10)
        config["domain"]["source"][0] = 0
        self.assertEqual(DEFAULTS["domain"]["source"][0], 1)

    def test_output_cannot_escape_project(self):
        with self.assertRaises(ValueError):
            project_path("../outside.json", output=True)
        self.assertEqual(project_path("results/x.json", output=True).parent, PROJECT_ROOT / "results")

    def test_invalid_source_is_rejected(self):
        config = deep_merge(DEFAULTS, {"domain": {"source": [3, 1, 1]}})
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_yaml_inheritance_and_cycles(self):
        with tempfile.TemporaryDirectory(dir=str(PROJECT_ROOT)) as directory:
            root = Path(directory)
            (root / "base.yaml").write_text("training:\n  batch_size: 64\n", encoding="utf-8")
            (root / "child.yaml").write_text("extends: base.yaml\ntraining:\n  iterations: 3\n", encoding="utf-8")
            config = load_config(root / "child.yaml")
            self.assertEqual(config["training"]["iterations"], 3)
            self.assertEqual(config["training"]["batch_size"], 64)
            (root / "base.yaml").write_text("extends: child.yaml\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(root / "child.yaml")
