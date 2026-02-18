"""Tests for the PyTorch GPU raycasting renderer."""

import math
import os
import tempfile
import unittest

import numpy as np

try:
    import torch

    _TORCH_AVAILABLE = True
except ImportError:
    _TORCH_AVAILABLE = False

from ratvision.raycasting_renderer import (
    RaycastingRenderer,
    BoxEnvironment,
    Landmark,
)


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch not installed")
class TestTorchRenderer(unittest.TestCase):
    """Tests for TorchRenderer."""

    def setUp(self):
        from ratvision.torch_renderer import TorchRenderer

        self.TorchRenderer = TorchRenderer

        self.env = BoxEnvironment(
            width=0.635,
            depth=0.635,
            height=0.5,
            wall_textures=None,
            floor_texture=None,
            wall_color=0.5,
            floor_color=0.3,
            ceiling_color=0.0,
        )
        self.renderer = self.TorchRenderer(
            env=self.env,
            config={"frame_dim": (32, 16)},
        )

    # ---- initialisation ----

    def test_default_config(self):
        r = self.TorchRenderer(env=self.env)
        self.assertEqual(r.config["frame_dim"], (128, 64))
        self.assertAlmostEqual(r.config["camera_height"], 0.035)
        self.assertAlmostEqual(r.config["hfov"], 4 * math.pi / 3)

    def test_custom_config(self):
        config = {"frame_dim": (64, 32), "camera_height": 0.05}
        r = self.TorchRenderer(env=self.env, config=config)
        self.assertEqual(r.config["frame_dim"], (64, 32))
        self.assertAlmostEqual(r.config["camera_height"], 0.05)

    def test_update_config(self):
        self.renderer.update_config({"frame_dim": (48, 24)})
        self.assertEqual(self.renderer._frame_W, 48)
        self.assertEqual(self.renderer._frame_H, 24)

    def test_invalid_config_key(self):
        self.renderer.update_config({"nonexistent_key": 42})
        self.assertNotIn("nonexistent_key", self.renderer.config)

    def test_config_not_dict_raises(self):
        with self.assertRaises(ValueError):
            self.renderer.update_config("not a dict")

    # ---- render_frame ----

    def test_render_frame_shape(self):
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.shape, (16, 32))

    def test_render_frame_is_tensor(self):
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertIsInstance(frame, torch.Tensor)

    def test_render_frame_value_range(self):
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertGreaterEqual(frame.min().item(), 0.0)
        self.assertLessEqual(frame.max().item(), 1.0)

    def test_render_frame_dtype(self):
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.dtype, torch.float32)

    def test_frame_varies_with_theta(self):
        f1 = self.renderer.render_frame(0.3, 0.3, 0)
        f2 = self.renderer.render_frame(0.3, 0.3, math.pi / 2)
        self.assertFalse(torch.allclose(f1, f2))

    def test_frame_varies_with_position(self):
        f1 = self.renderer.render_frame(0.1, 0.1, 0)
        f2 = self.renderer.render_frame(0.5, 0.5, 0)
        self.assertFalse(torch.allclose(f1, f2))

    def test_custom_resolution(self):
        self.renderer.update_config({"frame_dim": (8, 4)})
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.shape, (4, 8))

    # ---- batched forward ----

    def test_forward_batch(self):
        positions = torch.tensor([[0.3, 0.3], [0.4, 0.4]], dtype=torch.float32)
        thetas = torch.tensor([0.0, 0.5], dtype=torch.float32)
        frames = self.renderer(positions, thetas)
        self.assertEqual(frames.shape, (2, 16, 32))

    def test_forward_single(self):
        pos = torch.tensor([0.3, 0.3], dtype=torch.float32)
        theta = torch.tensor(0.0, dtype=torch.float32)
        frames = self.renderer(pos, theta)
        self.assertEqual(frames.shape, (1, 16, 32))

    # ---- render_path ----

    def test_render_path_lists(self):
        positions = [(0.3, 0.3), (0.4, 0.4)]
        hds = [0.0, 0.5]
        frames = self.renderer.render_path(positions, hds)
        self.assertEqual(frames.shape, (2, 16, 32))
        self.assertIsInstance(frames, torch.Tensor)

    def test_render_path_tensors(self):
        positions = torch.tensor([[0.3, 0.3], [0.4, 0.4]])
        hds = torch.tensor([0.0, 0.5])
        frames = self.renderer.render_path(positions, hds)
        self.assertEqual(frames.shape, (2, 16, 32))

    def test_render_path_type_error(self):
        with self.assertRaises(TypeError):
            self.renderer.render_path("bad", [0.0])

    def test_render_path_value_error(self):
        with self.assertRaises(ValueError):
            self.renderer.render_path([(0, 0)], [0.0, 1.0])

    # ---- save_frames ----

    def test_save_frames(self):
        positions = [(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]
        hds = [0.0, 0.5, 1.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            self.renderer.save_frames(positions, hds, output_dir=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(len(files), 3)
            self.assertTrue(all(f.endswith(".png") for f in files))

    # ---- to_numpy ----

    def test_to_numpy(self):
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        arr = self.TorchRenderer.to_numpy(frame)
        self.assertIsInstance(arr, np.ndarray)
        self.assertEqual(arr.shape, (16, 32))

    # ---- dtype config ----

    def test_float64_dtype(self):
        r = self.TorchRenderer(
            env=self.env, config={"frame_dim": (16, 8)}, dtype=torch.float64
        )
        frame = r.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.dtype, torch.float64)

    # ---- default textured environment ----

    def test_render_with_default_env(self):
        r = self.TorchRenderer(config={"frame_dim": (16, 8)})
        frame = r.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.shape, (8, 16))
        self.assertGreaterEqual(frame.min().item(), 0.0)
        self.assertLessEqual(frame.max().item(), 1.0)

    # ---- landmark ----

    def test_landmark_visible(self):
        """A large white landmark on a wall should produce bright pixels."""

        def white_square(u, v):
            c = np.ones_like(u)
            a = np.ones_like(u)
            return c, a

        env = BoxEnvironment(
            wall_color=0.0,
            floor_color=0.0,
            ceiling_color=0.0,
            landmarks=[
                Landmark(
                    wall_index=0,
                    uv_min=(0.0, 0.0),
                    uv_max=(1.0, 1.0),
                    shape_fn=white_square,
                ),
            ],
        )
        r = self.TorchRenderer(env=env, config={"frame_dim": (32, 16)})
        frame = r.render_frame(0.3, 0.1, 0)
        self.assertGreater(frame.max().item(), 0.8)

    # ---- consistency with RaycastingRenderer ----

    def test_consistency_with_numpy_renderer(self):
        """TorchRenderer output should closely match RaycastingRenderer."""
        env = BoxEnvironment(
            wall_color=0.5,
            floor_color=0.3,
            ceiling_color=0.0,
        )
        config = {"frame_dim": (32, 16)}
        np_renderer = RaycastingRenderer(env=env, config=config)
        torch_renderer = self.TorchRenderer(env=env, config=config, dtype=torch.float64)

        for x, y, theta in [(0.3, 0.3, 0.0), (0.1, 0.5, 1.2), (0.5, 0.1, -0.8)]:
            np_frame = np_renderer.render_frame(x, y, theta)
            torch_frame = torch_renderer.render_frame(x, y, theta).cpu().numpy()
            np.testing.assert_allclose(
                torch_frame,
                np_frame,
                atol=1e-6,
                err_msg=f"Mismatch at position ({x}, {y}, {theta})",
            )


@unittest.skipUnless(_TORCH_AVAILABLE, "PyTorch not installed")
@unittest.skipUnless(
    _TORCH_AVAILABLE and torch.cuda.is_available(),
    "CUDA not available",
)
class TestTorchRendererCUDA(unittest.TestCase):
    """Tests that require a CUDA GPU."""

    def test_to_cuda(self):
        from ratvision.torch_renderer import TorchRenderer

        env = BoxEnvironment(wall_color=0.5, floor_color=0.3, ceiling_color=0.0)
        r = TorchRenderer(env=env, config={"frame_dim": (16, 8)}).to("cuda")
        pos = torch.tensor([0.3, 0.3], device="cuda")
        theta = torch.tensor(0.0, device="cuda")
        frame = r(pos, theta)
        self.assertEqual(frame.device.type, "cuda")
        self.assertEqual(frame.shape, (1, 8, 16))

    def test_batch_on_cuda(self):
        from ratvision.torch_renderer import TorchRenderer

        env = BoxEnvironment(wall_color=0.5, floor_color=0.3, ceiling_color=0.0)
        r = TorchRenderer(env=env, config={"frame_dim": (16, 8)}).to("cuda")
        positions = torch.rand(64, 2, device="cuda") * 0.5 + 0.05
        thetas = torch.rand(64, device="cuda") * 2 * math.pi - math.pi
        frames = r(positions, thetas)
        self.assertEqual(frames.shape, (64, 8, 16))
        self.assertEqual(frames.device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
