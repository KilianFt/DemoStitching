import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import src.stitching.ds_stitching as ds_stitching
from src.util.ds_tools import Demonstration, Trajectory, apply_lpvds_demowise
from src.util.load_tools import load_demonstration_set, resolve_data_scales


class DataScaleLoadingTests(unittest.TestCase):
    def _write_single_demo_dataset(self, root: Path):
        demo_dir = root / "demonstration_0"
        demo_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "x": [[1.0, 2.0], [3.0, 4.0]],
            "x_dot": [[0.1, 0.2], [0.3, 0.4]],
        }
        with open(demo_dir / "trajectory_0.json", "w", encoding="utf-8") as f:
            json.dump(payload, f)

    def test_load_demonstration_set_scales_positions_and_velocities(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_single_demo_dataset(root)

            demos = load_demonstration_set(
                str(root),
                position_scale=10.0,
                velocity_scale=5.0,
            )
            self.assertEqual(len(demos), 1)

            traj = demos[0].trajectories[0]
            np.testing.assert_allclose(traj.x, np.array([[10.0, 20.0], [30.0, 40.0]]))
            np.testing.assert_allclose(traj.x_dot, np.array([[0.5, 1.0], [1.5, 2.0]]))

    def test_load_demonstration_set_velocity_defaults_to_position_scale(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            self._write_single_demo_dataset(root)

            demos = load_demonstration_set(str(root), position_scale=7.0, velocity_scale=None)
            traj = demos[0].trajectories[0]
            np.testing.assert_allclose(traj.x, np.array([[7.0, 14.0], [21.0, 28.0]]))
            np.testing.assert_allclose(traj.x_dot, np.array([[0.7, 1.4], [2.1, 2.8]]))

    def test_resolve_data_scales_supports_new_and_legacy_config_fields(self):
        pos, vel = resolve_data_scales(SimpleNamespace(data_position_scale=3.0, data_velocity_scale=None))
        self.assertEqual(pos, 3.0)
        self.assertEqual(vel, 3.0)

        pos, vel = resolve_data_scales(SimpleNamespace(damm_position_scale=4.0, damm_velocity_scale=2.0))
        self.assertEqual(pos, 4.0)
        self.assertEqual(vel, 2.0)


class LpvdsWiringTests(unittest.TestCase):
    def test_apply_lpvds_demowise_does_not_pass_damm_scale_kwargs(self):
        traj = Trajectory(
            x=np.array([[0.0, 0.0], [0.2, 0.0], [0.5, 0.2], [0.8, 0.2]]),
            x_dot=np.array([[0.2, 0.0], [0.3, 0.1], [0.3, 0.0], [0.2, -0.1]]),
        )
        demo_set = [Demonstration([traj])]
        config = SimpleNamespace(
            rel_scale=0.11,
            total_scale=0.66,
            nu_0=9,
            kappa_0=0.4,
            psi_dir_0=0.2,
            data_position_scale=8.0,
            data_velocity_scale=9.0,
        )

        captured_kwargs = []

        class DummyLPVDS:
            def __init__(self, *args, **kwargs):
                captured_kwargs.append(kwargs)

            def begin(self):
                return True

        with patch("src.util.ds_tools.lpvds_class", DummyLPVDS):
            ds_set, reversed_ds_set, _ = apply_lpvds_demowise(demo_set, config=config)

        self.assertEqual(len(ds_set), 1)
        self.assertEqual(len(reversed_ds_set), 1)
        self.assertEqual(len(captured_kwargs), 2)
        for kwargs in captured_kwargs:
            self.assertEqual(kwargs["rel_scale"], 0.11)
            self.assertEqual(kwargs["total_scale"], 0.66)
            self.assertEqual(kwargs["nu_0"], 9)
            self.assertEqual(kwargs["kappa_0"], 0.4)
            self.assertEqual(kwargs["psi_dir_0"], 0.2)
            self.assertNotIn("damm_position_scale", kwargs)
            self.assertNotIn("damm_velocity_scale", kwargs)

    def test_ds_stitching_new_lpvds_uses_core_hyperparams_only(self):
        x = np.array([[0.0, 0.0], [0.2, 0.1]])
        x_dot = np.array([[0.1, 0.0], [0.1, -0.1]])
        config = SimpleNamespace(
            rel_scale=0.11,
            total_scale=0.66,
            nu_0=9,
            kappa_0=0.4,
            psi_dir_0=0.2,
            data_position_scale=8.0,
            data_velocity_scale=8.0,
        )
        with patch("src.stitching.ds_stitching.lpvds_class", return_value=SimpleNamespace()) as lpvds_ctor:
            _ = ds_stitching._new_lpvds(x, x_dot, np.array([1.0, 1.0]), config)

        lpvds_ctor.assert_called_once()
        kwargs = lpvds_ctor.call_args.kwargs
        self.assertEqual(kwargs["rel_scale"], 0.11)
        self.assertEqual(kwargs["total_scale"], 0.66)
        self.assertEqual(kwargs["nu_0"], 9)
        self.assertEqual(kwargs["kappa_0"], 0.4)
        self.assertEqual(kwargs["psi_dir_0"], 0.2)
        self.assertNotIn("damm_position_scale", kwargs)
        self.assertNotIn("damm_velocity_scale", kwargs)


if __name__ == "__main__":
    unittest.main()
