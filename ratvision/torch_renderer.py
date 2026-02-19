"""GPU-accelerated raycasting renderer using PyTorch.

This module provides :class:`TorchRenderer`, a ``torch.nn.Module``.
All rendering happens on the same device as the
module (CPU or CUDA), and the batch dimension is fully vectorised:
no Python loops over frames.

Textures and pre-computed ray grids are stored as registered buffers,
so a single ``.to(device)`` call moves everything to GPU.

Typical usage in a training loop::

    from ratvision import TorchRenderer
    import torch

    renderer = TorchRenderer(config={'frame_dim': (64, 32)}).to('cuda')

    positions = torch.rand(256, 2, device='cuda') * 0.585 + 0.025
    thetas    = torch.rand(256, device='cuda') * 2 * 3.14159 - 3.14159

    frames = renderer(positions, thetas)   # (256, 32, 64) on cuda
    # feed directly into a CNN: frames.unsqueeze(1) → (256, 1, 32, 64)
"""

import math
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Import environment definition and helpers from the existing module.
from .box_environment import (
    BoxEnvironment,
    Landmark,
    default_box_environment,
)


# ---------------------------------------------------------------------------
# Landmark pre-rasterisation
# ---------------------------------------------------------------------------


def _rasterise_landmark(
    landmark: Landmark,
    resolution: int = 128,
) -> torch.Tensor:
    """Pre-rasterise a landmark's shape_fn into a (2, H, W) float tensor.

    Channel 0 is grayscale colour, channel 1 is alpha.  Both in [0, 1].

    Args:
        landmark: A :class:`Landmark` with a NumPy-based ``shape_fn``.
        resolution: Pixel resolution for the rasterised texture.

    Returns:
        ``torch.Tensor`` of shape ``(2, resolution, resolution)``.
    """
    u = np.linspace(0, 1, resolution, dtype=np.float64)
    v = np.linspace(0, 1, resolution, dtype=np.float64)
    uu, vv = np.meshgrid(u, v)  # (res, res), v increases downward in array
    # shape_fn expects v=0 at bottom, v=1 at top — flip rows so row-0 = v_max
    vv_flipped = vv[::-1].copy()
    color, alpha = landmark.shape_fn(uu, vv_flipped)
    stacked = np.stack([color, alpha], axis=0)  # (2, res, res)
    return torch.from_numpy(stacked.astype(np.float32))


# ---------------------------------------------------------------------------
# TorchRenderer
# ---------------------------------------------------------------------------


class TorchRenderer(nn.Module):
    """GPU-accelerated analytical raycasting renderer for box environments.

    This is a ``torch.nn.Module`` counterpart of
    :class:`~ratvision.raycasting_renderer.RaycastingRenderer`.  It uses
    equirectangular projection to cast rays from agent positions and
    computes wall / floor / ceiling intersections analytically.  Textures
    are sampled via ``torch.nn.functional.grid_sample`` (bilinear).

    Call the module directly (``forward``) with batched positions and
    head directions to get a batch of rendered frames as a tensor on the
    same device.

    Example::

        renderer = TorchRenderer().to('cuda')
        frames = renderer(positions, thetas)  # (B, H, W) tensor

    Args:
        env: A :class:`BoxEnvironment` describing the scene.  If *None*,
            the built-in default box environment is loaded.
        config: Configuration dictionary.  Supported keys:

            - ``frame_dim``: ``(width, height)`` in pixels.  Default ``(128, 64)``.
            - ``camera_height``: Camera height in metres.  Default ``0.035``.
            - ``hfov``: Horizontal FOV in radians.  Default ``4π/3`` (240°).
            - ``vfov``: Vertical FOV in radians.  Default ``2π/3`` (120°).
            - ``output_dir``: Path for saved PNGs.  Default ``./output``.
        dtype: Torch dtype for internal computation.  Default ``torch.float32``.
        landmark_resolution: Resolution at which analytical landmarks are
            pre-rasterised to textures.  Default ``128``.
    """

    DEFAULT_CONFIG: Dict = {
        "frame_dim": (128, 64),
        "camera_height": 0.035,
        "hfov": 4 * math.pi / 3,  # 240 degrees
        "vfov": 2 * math.pi / 3,  # 120 degrees
        "output_dir": None,
    }

    CONFIG_DESCRIPTION: Dict = {
        "frame_dim": "Dimensions of the rendered frames (width, height), in pixels.",
        "camera_height": "Height of the camera from the ground in metres.  Default 0.035 m.",
        "hfov": "Horizontal field of view in radians.  Default 4π/3 (240°).",
        "vfov": "Vertical field of view in radians.  Default 2π/3 (120°).",
        "output_dir": 'Path where rendered images will be saved.  If None, "./output" is used.',
    }

    def __init__(
        self,
        env: Optional[BoxEnvironment] = None,
        config: Optional[Dict] = None,
        dtype: torch.dtype = torch.float32,
        landmark_resolution: int = 128,
    ):
        super().__init__()

        # ---- environment ----
        if env is None:
            print("[*] no environment provided, loading default box environment.")
            env = default_box_environment()
        self._env = env
        self._dtype = dtype
        self._landmark_resolution = landmark_resolution

        # ---- config ----
        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self._apply_config(config)
        else:
            print("[*] no configuration provided, using default.")

        if self.config["output_dir"] is None:
            self.config["output_dir"] = os.path.join(os.getcwd(), "output")

        # ---- register all numeric data as buffers ----
        self._register_environment_buffers()
        self._register_ray_buffers()

    # ------------------------------------------------------------------
    # Configuration helpers
    # ------------------------------------------------------------------

    def _apply_config(self, config: Dict) -> None:
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary.")
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"[-] {key} is not a valid configuration key, skipping.")

    def update_config(self, config: Dict) -> None:
        """Update configuration and recompute ray buffers.

        Args:
            config: Dictionary of configuration overrides.
        """
        self._apply_config(config)
        self._register_ray_buffers()

    @staticmethod
    def config_description() -> None:
        """Print a description of each configuration key."""
        print("[*] configuration description:")
        for key, value in TorchRenderer.CONFIG_DESCRIPTION.items():
            print(f"\t{key}: {value}")

    def print_config(self) -> None:
        """Print the current configuration."""
        print("[*] current configuration:")
        for key, value in self.config.items():
            print(f"\t{key}: {value}")

    # ------------------------------------------------------------------
    # Buffer registration
    # ------------------------------------------------------------------

    def _register_environment_buffers(self) -> None:
        """Convert environment data (textures, geometry) to registered buffers."""
        env = self._env
        dtype = self._dtype

        # ---- scalar geometry ----
        self.register_buffer(
            "_box_dims",
            torch.tensor([env.width, env.depth, env.height], dtype=dtype),
        )
        self.register_buffer(
            "_surface_colors",
            torch.tensor(
                [env.wall_color, env.floor_color, env.ceiling_color], dtype=dtype
            ),
        )

        # ---- wall textures ----
        # Stored as (4, 1, tex_H, tex_W) for grid_sample — or None flag.
        if env.wall_textures is not None:
            wall_list = []
            for tex in env.wall_textures:
                t = torch.from_numpy(
                    tex.astype(np.float32 if dtype == torch.float32 else np.float64)
                )
                wall_list.append(t.unsqueeze(0).unsqueeze(0))  # (1, 1, H, W)
            # We store them individually so they can have different sizes.
            for i, wt in enumerate(wall_list):
                self.register_buffer(f"_wall_tex_{i}", wt.to(dtype))
            self._has_wall_textures = True
            self._n_wall_textures = len(wall_list)
        else:
            self._has_wall_textures = False
            self._n_wall_textures = 0

        # ---- floor texture ----
        if env.floor_texture is not None:
            ft = torch.from_numpy(
                env.floor_texture.astype(
                    np.float32 if dtype == torch.float32 else np.float64
                )
            )
            self.register_buffer("_floor_tex", ft.unsqueeze(0).unsqueeze(0).to(dtype))
            self._has_floor_texture = True
        else:
            self._has_floor_texture = False

        # ---- wall geometry ----
        # Pack into tensors for vectorised intersection.
        # normals: (4, 3), origins: (4, 3), u_axes: (4, 3), v_axes: (4, 3)
        # u_extents: (4,), v_extents: (4,)
        w, d, h = env.width, env.depth, env.height
        normals = torch.tensor(
            [
                [0.0, -1.0, 0.0],  # wall 0 (north)
                [0.0, 1.0, 0.0],  # wall 1 (south)
                [-1.0, 0.0, 0.0],  # wall 2 (east)
                [1.0, 0.0, 0.0],  # wall 3 (west)
            ],
            dtype=dtype,
        )
        origins = torch.tensor(
            [
                [0.0, d, 0.0],
                [w, 0.0, 0.0],
                [w, 0.0, 0.0],
                [0.0, d, 0.0],
            ],
            dtype=dtype,
        )
        u_axes = torch.tensor(
            [
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0],
            ],
            dtype=dtype,
        )
        v_axes = torch.tensor(
            [
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=dtype,
        )
        u_extents = torch.tensor([w, w, d, d], dtype=dtype)
        v_extents = torch.tensor([h, h, h, h], dtype=dtype)
        # d_plane = dot(normal, origin) per wall — shape (4,)
        d_plane = (normals * origins).sum(dim=-1)

        self.register_buffer("_wall_normals", normals)
        self.register_buffer("_wall_origins", origins)
        self.register_buffer("_wall_u_axes", u_axes)
        self.register_buffer("_wall_v_axes", v_axes)
        self.register_buffer("_wall_u_extents", u_extents)
        self.register_buffer("_wall_v_extents", v_extents)
        self.register_buffer("_wall_d_plane", d_plane)

        # ---- landmarks ----
        # Pre-rasterise each landmark and store metadata.
        self._landmark_wall_indices: List[int] = []
        self._landmark_uv_bounds: List[Tuple[float, float, float, float]] = []
        for i, lm in enumerate(env.landmarks):
            tex = _rasterise_landmark(lm, self._landmark_resolution).to(dtype)
            self.register_buffer(f"_lm_tex_{i}", tex.unsqueeze(0))  # (1, 2, H, W)
            self._landmark_wall_indices.append(lm.wall_index)
            self._landmark_uv_bounds.append(
                (lm.uv_min[0], lm.uv_min[1], lm.uv_max[0], lm.uv_max[1])
            )
        self._n_landmarks = len(env.landmarks)

    def _register_ray_buffers(self) -> None:
        """Pre-compute per-pixel azimuth and elevation offset buffers."""
        W, H = self.config["frame_dim"]
        hfov = self.config["hfov"]
        vfov = self.config["vfov"]
        dtype = self._dtype

        col = torch.arange(W, dtype=dtype)
        delta_az = hfov * (0.5 - col / max(W - 1, 1))  # (W,)

        row = torch.arange(H, dtype=dtype)
        delta_el = vfov * (0.5 - row / max(H - 1, 1))  # (H,)

        self.register_buffer("_delta_az", delta_az.unsqueeze(0))  # (1, W)
        self.register_buffer("_delta_el", delta_el.unsqueeze(-1))  # (H, 1)
        self._frame_W = W
        self._frame_H = H

    # ------------------------------------------------------------------
    # Texture sampling helper
    # ------------------------------------------------------------------

    @staticmethod
    def _grid_sample_texture(
        texture: torch.Tensor,
        u: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        """Sample a texture via grid_sample (bilinear, border-clamped).

        Args:
            texture: ``(1, C, tex_H, tex_W)`` texture tensor.
            u: ``(*, )`` horizontal UV coords in [0, 1].
            v: ``(*, )`` vertical UV coords in [0, 1].

        Returns:
            ``(*, C)`` sampled values (squeezed if C == 1).
        """
        orig_shape = u.shape
        # grid_sample expects (N, H_out, W_out, 2) grid in [-1, 1]
        # Map u ∈ [0,1] → [-1,1];  v ∈ [0,1] → [1,-1] (v=0 bottom → +1, v=1 top → -1)
        gx = 2.0 * u.reshape(1, 1, -1) - 1.0
        gy = 1.0 - 2.0 * v.reshape(1, 1, -1)
        grid = torch.stack([gx, gy], dim=-1)  # (1, 1, N_pixels, 2)

        C = texture.shape[1]
        grid = grid.clamp(-1, 1)
        sampled = F.grid_sample(
            texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, C, 1, N_pixels)
        sampled = sampled.reshape(C, *orig_shape)  # (C, *orig_shape)
        if C == 1:
            return sampled.squeeze(0)
        return sampled

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    @torch.no_grad()
    def forward(
        self,
        positions: torch.Tensor,
        head_directions: torch.Tensor,
    ) -> torch.Tensor:
        """Render a batch of frames.

        Args:
            positions: Agent positions, shape ``(B, 2)`` or ``(2,)`` (metres).
                       Automatically unsqueezed to ``(1, 2)`` if 1-D.
            head_directions: Head directions in radians, shape ``(B,)`` or
                scalar.  ``0`` = north (+Y), positive = counter-clockwise.

        Returns:
            Grayscale frames as ``torch.Tensor`` of shape ``(B, H, W)``
            with values in [0, 1], on the same device as the module.
        """
        # ---- input normalisation ----
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if head_directions.dim() == 0:
            head_directions = head_directions.unsqueeze(0)
        B = positions.shape[0]
        H, W = self._frame_H, self._frame_W
        dtype = self._dtype
        device = positions.device

        cam_z = torch.tensor(self.config["camera_height"], dtype=dtype, device=device)
        width = self._box_dims[0]
        depth = self._box_dims[1]
        height = self._box_dims[2]
        wall_color = self._surface_colors[0]
        floor_color = self._surface_colors[1]
        ceiling_color = self._surface_colors[2]

        # ---- ray directions (B, H, W) ----
        theta = head_directions.to(dtype)  # (B,)
        az = theta[:, None, None] + self._delta_az[None, :, :]  # (B, 1, W)
        el = self._delta_el[None, :, :]  # (1, H, 1)

        cos_el = torch.cos(el)
        dx = -torch.sin(az) * cos_el  # (B, H, W)
        dy = torch.cos(az) * cos_el  # (B, H, W)
        dz = torch.sin(el).expand(B, H, W)  # (B, H, W)

        # origin: (B, 3)
        ox = positions[:, 0]
        oy = positions[:, 1]
        oz = cam_z.expand(B)

        # ---- output buffers ----
        image = torch.full((B, H, W), ceiling_color.item(), dtype=dtype, device=device)
        closest_t = torch.full((B, H, W), float("inf"), dtype=dtype, device=device)

        # ---- wall intersections (loop over 4 walls) ----
        for wi in range(4):
            normal = self._wall_normals[wi]  # (3,)
            d_pl = self._wall_d_plane[wi]  # scalar
            u_axis = self._wall_u_axes[wi]  # (3,)
            v_axis = self._wall_v_axes[wi]  # (3,)
            w_origin = self._wall_origins[wi]  # (3,)
            u_ext = self._wall_u_extents[wi]
            v_ext = self._wall_v_extents[wi]

            # denom = dot(normal, ray_dir) — per pixel: (B, H, W)
            denom = normal[0] * dx + normal[1] * dy + normal[2] * dz
            # numer = d_plane - dot(normal, origin) — per frame: (B, 1, 1)
            numer = d_pl - (normal[0] * ox + normal[1] * oy + normal[2] * oz)
            numer = numer[:, None, None]

            valid = denom.abs() > 1e-12
            t_hit = torch.where(
                valid,
                numer / denom,
                torch.tensor(float("inf"), dtype=dtype, device=device),
            )

            hit_mask = valid & (t_hit > 1e-6) & (t_hit < closest_t)
            if not hit_mask.any():
                continue

            # hit points (B, H, W)
            hit_x = ox[:, None, None] + t_hit * dx
            hit_y = oy[:, None, None] + t_hit * dy
            hit_z = oz[:, None, None] + t_hit * dz

            # UV coordinates on wall
            rel_x = hit_x - w_origin[0]
            rel_y = hit_y - w_origin[1]
            rel_z = hit_z - w_origin[2]
            u_coord = (
                rel_x * u_axis[0] + rel_y * u_axis[1] + rel_z * u_axis[2]
            ) / u_ext
            v_coord = (
                rel_x * v_axis[0] + rel_y * v_axis[1] + rel_z * v_axis[2]
            ) / v_ext

            on_wall = (
                hit_mask
                & (u_coord >= 0)
                & (u_coord <= 1)
                & (v_coord >= 0)
                & (v_coord <= 1)
            )
            if not on_wall.any():
                continue

            # sample wall texture
            if self._has_wall_textures and wi < self._n_wall_textures:
                wall_tex = getattr(self, f"_wall_tex_{wi}")  # (1, 1, tex_H, tex_W)
                color = self._grid_sample_texture(
                    wall_tex, u_coord, v_coord
                )  # (B, H, W)
            else:
                color = wall_color.expand(B, H, W)

            # overlay landmarks on this wall
            for li in range(self._n_landmarks):
                if self._landmark_wall_indices[li] != wi:
                    continue
                u_min, v_min, u_max, v_max = self._landmark_uv_bounds[li]
                lm_u = (u_coord - u_min) / (u_max - u_min)
                lm_v = (v_coord - v_min) / (v_max - v_min)
                in_bounds = (
                    on_wall & (lm_u >= 0) & (lm_u <= 1) & (lm_v >= 0) & (lm_v <= 1)
                )
                if not in_bounds.any():
                    continue

                lm_tex = getattr(self, f"_lm_tex_{li}")  # (1, 2, lm_H, lm_W)
                lm_sampled = self._grid_sample_texture(
                    lm_tex, lm_u, lm_v
                )  # (2, B, H, W)
                lm_color = lm_sampled[0]  # (B, H, W)
                lm_alpha = lm_sampled[1]  # (B, H, W)
                blended = lm_alpha * lm_color + (1.0 - lm_alpha) * color
                color = torch.where(in_bounds, blended, color)

            image = torch.where(on_wall, color, image)
            closest_t = torch.where(on_wall, t_hit, closest_t)

        # ---- floor intersection (z = 0) ----
        t_floor = torch.where(
            dz.abs() > 1e-12,
            -cam_z / dz,
            torch.tensor(float("inf"), dtype=dtype, device=device),
        )
        fx = ox[:, None, None] + t_floor * dx
        fy = oy[:, None, None] + t_floor * dy
        floor_hit = (
            (t_floor > 1e-6)
            & (t_floor < closest_t)
            & (fx >= 0)
            & (fx <= width)
            & (fy >= 0)
            & (fy <= depth)
        )
        if floor_hit.any():
            if self._has_floor_texture:
                fu = fx / width
                fv = fy / depth
                f_color = self._grid_sample_texture(self._floor_tex, fu, fv)
                image = torch.where(floor_hit, f_color, image)
            else:
                image = torch.where(floor_hit, floor_color, image)
            closest_t = torch.where(floor_hit, t_floor, closest_t)

        # ---- ceiling intersection (z = height) ----
        t_ceil = torch.where(
            dz.abs() > 1e-12,
            (height - cam_z) / dz,
            torch.tensor(float("inf"), dtype=dtype, device=device),
        )
        cx = ox[:, None, None] + t_ceil * dx
        cy = oy[:, None, None] + t_ceil * dy
        ceil_hit = (
            (t_ceil > 1e-6)
            & (t_ceil < closest_t)
            & (cx >= 0)
            & (cx <= width)
            & (cy >= 0)
            & (cy <= depth)
        )
        if ceil_hit.any():
            image = torch.where(ceil_hit, ceiling_color, image)

        return image

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def render_frame(self, x: float, y: float, theta: float) -> torch.Tensor:
        """Render a single frame (convenience wrapper around ``forward``).

        Args:
            x: X-coordinate of the agent (metres).
            y: Y-coordinate of the agent (metres).
            theta: Head direction in radians.

        Returns:
            ``torch.Tensor`` of shape ``(H, W)`` with values in [0, 1].
        """
        device = self._box_dims.device
        pos = torch.tensor([x, y], dtype=self._dtype, device=device)
        hd = torch.tensor(theta, dtype=self._dtype, device=device)
        return self.forward(pos, hd).squeeze(0)

    def render_path(
        self,
        positions,
        head_directions,
    ) -> torch.Tensor:
        """Render frames for an entire path.

        Accepts the same list-of-tuples / list-of-floats format as
        :meth:`RaycastingRenderer.render_path`, or tensors directly.

        Args:
            positions: ``(N, 2)`` tensor **or** list of ``(x, y)`` tuples.
            head_directions: ``(N,)`` tensor **or** list of floats (radians).

        Returns:
            ``torch.Tensor`` of shape ``(N, H, W)`` with values in [0, 1].

        Raises:
            TypeError: If inputs are not lists or tensors.
            ValueError: If inputs have different lengths.
        """
        device = self._box_dims.device

        # Accept lists (like the NumPy renderer) or tensors.
        if isinstance(positions, list):
            positions = torch.tensor(positions, dtype=self._dtype, device=device)
        if isinstance(head_directions, list):
            head_directions = torch.tensor(
                head_directions, dtype=self._dtype, device=device
            )

        if not isinstance(positions, torch.Tensor) or not isinstance(
            head_directions, torch.Tensor
        ):
            raise TypeError("positions and head_directions must be lists or tensors.")
        if positions.shape[0] != head_directions.shape[0]:
            raise ValueError(
                "positions and head_directions must have the same number of elements."
            )

        return self.forward(positions, head_directions)

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_frames(
        self,
        positions,
        head_directions,
        output_dir: Optional[str] = None,
    ) -> None:
        """Render a path and save each frame as a grayscale PNG.

        Args:
            positions: ``(N, 2)`` tensor or list of ``(x, y)`` tuples.
            head_directions: ``(N,)`` tensor or list of floats (radians).
            output_dir: Directory for output PNGs.  Defaults to
                ``self.config['output_dir']``.
        """
        from PIL import Image

        if output_dir is None:
            output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        frames = self.render_path(positions, head_directions)
        frames_np = frames.cpu().numpy()
        n_frames = frames_np.shape[0]
        digits = len(str(n_frames))

        for i in range(n_frames):
            fname = f"frame{str(i + 1).zfill(digits)}.png"
            img_array = (frames_np[i] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_array, mode="L").save(os.path.join(output_dir, fname))

        print(f'[+] saved {n_frames} frames to "{output_dir}"')

    @staticmethod
    def to_numpy(frames: torch.Tensor) -> np.ndarray:
        """Convert a tensor of frames to a NumPy array.

        Args:
            frames: ``torch.Tensor`` of any shape.

        Returns:
            ``np.ndarray`` on CPU.
        """
        return frames.cpu().numpy()
