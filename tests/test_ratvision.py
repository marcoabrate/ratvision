import math
import os
import tempfile
import unittest

import numpy as np

from ratvision.renderer import Renderer
from ratvision.raycasting_renderer import (
    RaycastingRenderer,
    BoxEnvironment,
    Landmark,
    default_box_environment,
)


class TestRenderer(unittest.TestCase):
    def test_renderer_initialization(self):
        '''
        Test that the Renderer initializes correctly with updated config.
        '''
        config = {'output_dir': 'new/output/dir', 'camera_height': 0.04}
        renderer = Renderer(blender_exec='', config=config)

        keys = config.keys()
        self.assertEqual([renderer.config[k] for k in keys], [config[k] for k in keys])

        renderer_no_config = Renderer(blender_exec='')
        self.assertEqual(renderer_no_config.config, Renderer.DEFAULT_CONFIG)

    def test_render_method_type_error(self):
        '''
        Test that render method raises TypeError for invalid inputs.
        '''
        renderer = Renderer(blender_exec='')
        with self.assertRaises(TypeError):
            renderer.render('not a list', [])
        with self.assertRaises(TypeError):
            renderer.render([], 'not a list')

    def test_render_method_value_error(self):
        '''
        Test that render method raises ValueError for mismatched list lengths.
        '''
        renderer = Renderer(blender_exec='')
        positions = [(0, 0)]
        head_directions = [0, 0]
        with self.assertRaises(ValueError):
            renderer.render(positions, head_directions)


class TestRaycastingRenderer(unittest.TestCase):
    """Tests for the raycasting-based renderer."""

    def setUp(self):
        """Create a renderer with a minimal flat-colour environment for fast tests."""
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
        self.renderer = RaycastingRenderer(
            env=self.env,
            config={'frame_dim': (32, 16)},
        )

    # ---- initialisation ----

    def test_default_config(self):
        """Default config values are set correctly."""
        r = RaycastingRenderer(env=self.env)
        self.assertEqual(r.config['frame_dim'], (128, 64))
        self.assertAlmostEqual(r.config['camera_height'], 0.035)
        self.assertAlmostEqual(r.config['hfov'], 4 * math.pi / 3)
        self.assertAlmostEqual(r.config['vfov'], 2 * math.pi / 3)

    def test_custom_config(self):
        """Custom config overrides defaults."""
        config = {'frame_dim': (64, 32), 'camera_height': 0.05}
        r = RaycastingRenderer(env=self.env, config=config)
        self.assertEqual(r.config['frame_dim'], (64, 32))
        self.assertAlmostEqual(r.config['camera_height'], 0.05)

    def test_update_config(self):
        """update_config applies new values and recomputes rays."""
        self.renderer.update_config({'frame_dim': (48, 24)})
        self.assertEqual(self.renderer._frame_W, 48)
        self.assertEqual(self.renderer._frame_H, 24)

    def test_invalid_config_key(self):
        """Unknown config keys are silently skipped."""
        self.renderer.update_config({'nonexistent_key': 42})
        self.assertNotIn('nonexistent_key', self.renderer.config)

    def test_config_not_dict_raises(self):
        """Non-dict config raises ValueError."""
        with self.assertRaises(ValueError):
            self.renderer.update_config('not a dict')

    # ---- render_frame ----

    def test_render_frame_shape(self):
        """render_frame returns correct shape."""
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.shape, (16, 32))

    def test_render_frame_value_range(self):
        """Pixel values are in [0, 1]."""
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertGreaterEqual(frame.min(), 0.0)
        self.assertLessEqual(frame.max(), 1.0)

    def test_render_frame_dtype(self):
        """Output dtype is float64."""
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.dtype, np.float64)

    def test_frame_varies_with_theta(self):
        """Different head directions produce different images."""
        f1 = self.renderer.render_frame(0.3, 0.3, 0)
        f2 = self.renderer.render_frame(0.3, 0.3, math.pi / 2)
        self.assertFalse(np.allclose(f1, f2))

    def test_frame_varies_with_position(self):
        """Different positions produce different images."""
        f1 = self.renderer.render_frame(0.1, 0.1, 0)
        f2 = self.renderer.render_frame(0.5, 0.5, 0)
        self.assertFalse(np.allclose(f1, f2))

    def test_custom_resolution(self):
        """frame_dim is respected."""
        self.renderer.update_config({'frame_dim': (8, 4)})
        frame = self.renderer.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.shape, (4, 8))

    # ---- render_path ----

    def test_render_path_shape(self):
        """render_path returns (N, H, W)."""
        positions = [(0.3, 0.3), (0.4, 0.4)]
        hds = [0.0, 0.5]
        frames = self.renderer.render_path(positions, hds)
        self.assertEqual(frames.shape, (2, 16, 32))

    def test_render_path_type_error(self):
        """Non-list inputs raise TypeError."""
        with self.assertRaises(TypeError):
            self.renderer.render_path('bad', [0.0])
        with self.assertRaises(TypeError):
            self.renderer.render_path([(0, 0)], 'bad')

    def test_render_path_value_error(self):
        """Mismatched lengths raise ValueError."""
        with self.assertRaises(ValueError):
            self.renderer.render_path([(0, 0)], [0.0, 1.0])

    # ---- save_frames ----

    def test_save_frames(self):
        """save_frames writes PNGs to disk."""
        positions = [(0.3, 0.3), (0.4, 0.4), (0.5, 0.5)]
        hds = [0.0, 0.5, 1.0]
        with tempfile.TemporaryDirectory() as tmpdir:
            self.renderer.save_frames(positions, hds, output_dir=tmpdir)
            files = sorted(os.listdir(tmpdir))
            self.assertEqual(len(files), 3)
            self.assertTrue(all(f.endswith('.png') for f in files))

    # ---- default environment ----

    def test_default_box_environment(self):
        """default_box_environment loads without error and has correct structure."""
        env = default_box_environment()
        self.assertAlmostEqual(env.width, 0.635)
        self.assertAlmostEqual(env.depth, 0.635)
        self.assertAlmostEqual(env.height, 0.5)
        self.assertIsNotNone(env.wall_textures)
        self.assertEqual(len(env.wall_textures), 4)
        self.assertIsNotNone(env.floor_texture)
        self.assertEqual(len(env.landmarks), 3)

    def test_render_with_default_env(self):
        """Rendering with the full textured default environment works."""
        r = RaycastingRenderer(config={'frame_dim': (16, 8)})
        frame = r.render_frame(0.3, 0.3, 0)
        self.assertEqual(frame.shape, (8, 16))
        self.assertGreaterEqual(frame.min(), 0.0)
        self.assertLessEqual(frame.max(), 1.0)

    # ---- landmarks ----

    def test_landmark_visible(self):
        """A large landmark on a wall should be visible in the rendered frame."""
        # Create a white square landmark covering the north wall
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
        r = RaycastingRenderer(env=env, config={'frame_dim': (32, 16)})
        # Look north from close to south wall → landmark covers many pixels
        frame = r.render_frame(0.3, 0.1, 0)
        # Some pixels should be close to 1.0 (the white landmark)
        self.assertGreater(frame.max(), 0.8)


if __name__ == '__main__':
    unittest.main()


