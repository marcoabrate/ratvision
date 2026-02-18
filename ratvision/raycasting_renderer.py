import math
import os
from typing import Dict, List, Tuple

import numpy as np


from ratvision.box_environment import BoxEnvironment, default_box_environment


# ---------------------------------------------------------------------------
# Bilinear texture sampler
# ---------------------------------------------------------------------------


def _sample_texture(texture: np.ndarray, u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Sample a 2-D texture with bilinear interpolation.

    Args:
        texture: (tex_H, tex_W) grayscale array in [0, 1].
        u: (...) array of horizontal coords in [0, 1].
        v: (...) array of vertical coords in [0, 1].

    Returns:
        Sampled values with same shape as *u* / *v*.
    """
    th, tw = texture.shape[:2]

    # map UV → pixel coords (clamp to valid range)
    px = np.clip(u * (tw - 1), 0, tw - 1)
    py = np.clip((1.0 - v) * (th - 1), 0, th - 1)  # v=0 bottom, v=1 top → row 0 is top

    x0 = np.floor(px).astype(int)
    y0 = np.floor(py).astype(int)
    x1 = np.minimum(x0 + 1, tw - 1)
    y1 = np.minimum(y0 + 1, th - 1)

    fx = px - x0
    fy = py - y0

    val = (
        texture[y0, x0] * (1 - fx) * (1 - fy)
        + texture[y0, x1] * fx * (1 - fy)
        + texture[y1, x0] * (1 - fx) * fy
        + texture[y1, x1] * fx * fy
    )
    return val


# ---------------------------------------------------------------------------
# Raycasting renderer
# ---------------------------------------------------------------------------


class RaycastingRenderer:
    """Fast analytical raycasting renderer for box environments.

    Uses equirectangular projection to cast rays from the agent position
    and compute wall / floor / ceiling intersections analytically.
    Textures and landmarks are sampled via bilinear interpolation.

    This renderer is intended as a fast, Blender-free alternative to
    :class:`ratvision.BlenderRenderer` for simple box environments.
    Typical throughput is >1000 frames/s at 32x16 resolution on CPU.

    Example:

        from ratvision import RaycastingRenderer

        renderer = RaycastingRenderer()            # default box
        frame = renderer.render_frame(0.3, 0.3, 0) # (H, W) numpy array
    """

    DEFAULT_CONFIG = {
        "frame_dim": (128, 64),
        "camera_height": 0.035,
        "hfov": 4 * math.pi / 3,  # 240 degrees
        "vfov": 2 * math.pi / 3,  # 120 degrees
        "output_dir": None,
    }

    CONFIG_DESCRIPTION = {
        "frame_dim": "Dimensions of the rendered frames (width, height), in pixels.",
        "camera_height": "Height of the camera from the ground in metres. Default is 0.035 m.",
        "hfov": "Horizontal field of view in radians.  Default is 4π/3 (240°).",
        "vfov": "Vertical field of view in radians.  Default is 2π/3 (120°).",
        "output_dir": 'Path where rendered images will be saved. If None, "./output" is used.',
    }

    def __init__(self, env: BoxEnvironment = None, config: Dict = None):
        """
        Args:
            env: A :class:`BoxEnvironment` describing the scene.  If *None*,
                the built-in default box environment is loaded.
            config: Configuration dictionary (see ``config_description()``).
                If *None*, default settings are used.
        """
        if env is None:
            print("[*] no environment provided, loading default box environment.")
            self.env = default_box_environment()
        else:
            self.env = env

        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.update_config(config)
        else:
            print("[*] no configuration provided, using default.")
            self._print_config_message()

        if self.config["output_dir"] is None:
            self.config["output_dir"] = os.path.join(os.getcwd(), "output")

        # pre-compute per-pixel angular offsets
        self._precompute_rays()

    @staticmethod
    def _print_config_message() -> None:
        print()
        print("you can check the configuration description by calling")
        print('the "config_description" function or by checking the documentation,')
        print('and update the configuration by calling the "update_config" function.')
        print()

    @staticmethod
    def config_description() -> None:
        """Print a description of each configuration key."""
        print("[*] configuration description:")
        for key, value in RaycastingRenderer.CONFIG_DESCRIPTION.items():
            print(f"\t{key}: {value}")
        RaycastingRenderer._print_config_message()

    def print_config(self) -> None:
        """Print the current configuration."""
        print("[*] current configuration:")
        for key, value in self.config.items():
            print(f"\t{key}: {value}")
        self._print_config_message()

    def update_config(self, config: Dict) -> None:
        """Update configuration with new values. Only known keys are applied.

        Args:
            config: Dictionary of configuration overrides.
        """
        if not isinstance(config, dict):
            raise ValueError("config must be a dictionary.")
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f"[-] {key} is not a valid configuration key, skipping.")

        # re-compute rays if resolution or FOV changed
        self._precompute_rays()

    # ------------------------------------------------------------------
    # Ray generation (equirectangular projection)
    # ------------------------------------------------------------------

    def _precompute_rays(self) -> None:
        """Pre-compute per-pixel azimuth and elevation offsets."""
        W, H = self.config["frame_dim"]
        hfov = self.config["hfov"]
        vfov = self.config["vfov"]

        # column → azimuth offset (left of centre is positive = CCW)
        col = np.arange(W, dtype=np.float64)
        delta_az = hfov * (0.5 - col / max(W - 1, 1))

        # row → elevation offset (top of image is positive = up)
        row = np.arange(H, dtype=np.float64)
        delta_el = vfov * (0.5 - row / max(H - 1, 1))

        # store as (H, W) broadcast-ready arrays
        self._delta_az = delta_az[np.newaxis, :]  # (1, W)
        self._delta_el = delta_el[:, np.newaxis]  # (H, 1)
        self._frame_H = H
        self._frame_W = W

    # ------------------------------------------------------------------
    # Wall geometry
    # ------------------------------------------------------------------

    def _wall_planes(self):
        """Return wall plane definitions.

        Each entry: (normal, point_on_plane, u_axis, v_axis,
                     wall_origin, u_extent, v_extent, wall_index)

        UV convention per wall:
          - u runs along the wall (0 → 1), v runs up (0 = floor, 1 = top).
          - wall_origin is the corner where (u, v) = (0, 0).
        """
        w, d, h = self.env.width, self.env.depth, self.env.height

        # Wall 0 (north): y = depth, faces -Y
        # Looking at this wall from inside, left is +X direction.
        # u: 0 at x=0, 1 at x=w  →  u_axis = (+1, 0, 0)
        wall0 = (
            np.array([0.0, -1.0, 0.0]),  # normal (inward)
            np.array([0.0, d, 0.0]),  # wall_origin (u=0, v=0)
            np.array([1.0, 0.0, 0.0]),  # u_axis
            np.array([0.0, 0.0, 1.0]),  # v_axis
            w,
            h,
            0,
        )
        # Wall 1 (south): y = 0, faces +Y
        # Looking at this wall from inside, left is -X direction.
        wall1 = (
            np.array([0.0, 1.0, 0.0]),
            np.array([w, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            w,
            h,
            1,
        )
        # Wall 2 (east): x = width, faces -X
        # Looking from inside, left is +Y direction.
        wall2 = (
            np.array([-1.0, 0.0, 0.0]),
            np.array([w, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            d,
            h,
            2,
        )
        # Wall 3 (west): x = 0, faces +X
        # Looking from inside, left is -Y direction.
        wall3 = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, d, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            d,
            h,
            3,
        )
        return [wall0, wall1, wall2, wall3]

    # ------------------------------------------------------------------
    # Core rendering
    # ------------------------------------------------------------------

    def render_frame(self, x: float, y: float, theta: float) -> np.ndarray:
        """Render a single frame from position (x, y) at head direction theta.

        Args:
            x: X-coordinate of the agent (metres).
            y: Y-coordinate of the agent (metres).
            theta: Head direction in radians.  theta = 0 points along +Y
                (north).  Positive theta is counter-clockwise.

        Returns:
            Grayscale image as ``np.ndarray`` of shape ``(H, W)`` with
            values in [0, 1].
        """
        H, W = self._frame_H, self._frame_W
        cam_z = self.config["camera_height"]
        origin = np.array([x, y, cam_z])

        # per-pixel world-space ray directions (equirectangular)
        az = theta + self._delta_az  # (1, W) broadcast with scalar
        el = self._delta_el  # (H, 1)

        cos_el = np.cos(el)
        # theta=0 → north (+Y);  positive theta → CCW
        dx = -np.sin(az) * cos_el  # (H, W)
        dy = np.cos(az) * cos_el  # (H, W)
        dz = np.sin(el) * np.ones_like(dx)  # (H, W)

        # output image and depth buffer
        image = np.full((H, W), self.env.ceiling_color, dtype=np.float64)
        closest_t = np.full((H, W), np.inf)

        # ---- intersect walls ----
        walls = self._wall_planes()
        for normal, wall_origin, u_axis, v_axis, u_ext, v_ext, wall_idx in walls:
            # plane equation: dot(normal, P) = dot(normal, wall_origin)
            d_plane = np.dot(normal, wall_origin)
            denom = normal[0] * dx + normal[1] * dy + normal[2] * dz
            numer = d_plane - (
                normal[0] * origin[0] + normal[1] * origin[1] + normal[2] * origin[2]
            )

            valid = np.abs(denom) > 1e-12
            t_hit = np.full((H, W), np.inf)
            t_hit[valid] = numer / denom[valid]

            hit_mask = valid & (t_hit > 1e-6) & (t_hit < closest_t)
            if not np.any(hit_mask):
                continue

            # hit points
            hit_x = origin[0] + t_hit * dx
            hit_y = origin[1] + t_hit * dy
            hit_z = origin[2] + t_hit * dz
            hit_pts = np.stack([hit_x, hit_y, hit_z], axis=-1)  # (H, W, 3)

            # UV on wall
            rel = hit_pts - wall_origin
            u_coord = np.sum(rel * u_axis, axis=-1) / u_ext
            v_coord = np.sum(rel * v_axis, axis=-1) / v_ext

            on_wall = (
                hit_mask
                & (u_coord >= 0)
                & (u_coord <= 1)
                & (v_coord >= 0)
                & (v_coord <= 1)
            )
            if not np.any(on_wall):
                continue

            # sample wall texture or flat colour
            if self.env.wall_textures is not None and wall_idx < len(
                self.env.wall_textures
            ):
                color = _sample_texture(
                    self.env.wall_textures[wall_idx], u_coord, v_coord
                )
            else:
                color = np.full((H, W), self.env.wall_color)

            # overlay landmarks
            for lm in self.env.landmarks:
                if lm.wall_index != wall_idx:
                    continue
                lm_u = (u_coord - lm.uv_min[0]) / (lm.uv_max[0] - lm.uv_min[0])
                lm_v = (v_coord - lm.uv_min[1]) / (lm.uv_max[1] - lm.uv_min[1])
                in_bounds = (
                    on_wall & (lm_u >= 0) & (lm_u <= 1) & (lm_v >= 0) & (lm_v <= 1)
                )

                if not np.any(in_bounds):
                    continue

                lm_color, lm_alpha = lm.shape_fn(lm_u, lm_v)
                color = np.where(
                    in_bounds,
                    lm_alpha * lm_color + (1 - lm_alpha) * color,
                    color,
                )

            image[on_wall] = color[on_wall]
            closest_t[on_wall] = t_hit[on_wall]

        # ---- intersect floor (z = 0) ----
        with np.errstate(divide="ignore", invalid="ignore"):
            t_floor = np.where(
                np.abs(dz) > 1e-12,
                -cam_z / dz,
                np.inf,
            )
        fx = origin[0] + t_floor * dx
        fy = origin[1] + t_floor * dy
        floor_hit = (
            (t_floor > 1e-6)
            & (t_floor < closest_t)
            & (fx >= 0)
            & (fx <= self.env.width)
            & (fy >= 0)
            & (fy <= self.env.depth)
        )
        if np.any(floor_hit):
            if self.env.floor_texture is not None:
                fu = fx / self.env.width
                fv = fy / self.env.depth
                floor_color = _sample_texture(self.env.floor_texture, fu, fv)
                image[floor_hit] = floor_color[floor_hit]
            else:
                image[floor_hit] = self.env.floor_color
            closest_t[floor_hit] = t_floor[floor_hit]

        # ---- intersect ceiling (z = height) ----
        with np.errstate(divide="ignore", invalid="ignore"):
            t_ceil = np.where(
                np.abs(dz) > 1e-12,
                (self.env.height - cam_z) / dz,
                np.inf,
            )
        cx = origin[0] + t_ceil * dx
        cy = origin[1] + t_ceil * dy
        ceil_hit = (
            (t_ceil > 1e-6)
            & (t_ceil < closest_t)
            & (cx >= 0)
            & (cx <= self.env.width)
            & (cy >= 0)
            & (cy <= self.env.depth)
        )
        if np.any(ceil_hit):
            image[ceil_hit] = self.env.ceiling_color

        return image

    def render_path(
        self,
        positions: List[Tuple[float, float]],
        head_directions: List[float],
    ) -> np.ndarray:
        """Render frames for an entire path.

        Args:
            positions: List of (x, y) coordinate tuples.
            head_directions: List of head directions in radians.

        Returns:
            ``np.ndarray`` of shape ``(N, H, W)`` with values in [0, 1].

        Raises:
            TypeError: If inputs are not lists.
            ValueError: If inputs have different lengths.
        """
        if not isinstance(positions, list) or not isinstance(head_directions, list):
            raise TypeError("positions and head_directions must be lists.")
        if len(positions) != len(head_directions):
            raise ValueError(
                "positions and head_directions must have the same number of elements."
            )
        if any(hd < -2 * math.pi or hd > 2 * math.pi for hd in head_directions):
            print("[!!!] remember that head_directions should be in radians.")

        n = len(positions)
        frames = np.empty((n, self._frame_H, self._frame_W), dtype=np.float64)
        for i in range(n):
            frames[i] = self.render_frame(
                positions[i][0], positions[i][1], head_directions[i]
            )
        return frames

    # ------------------------------------------------------------------
    # I/O helpers
    # ------------------------------------------------------------------

    def save_frames(
        self,
        positions: List[Tuple[float, float]],
        head_directions: List[float],
        output_dir: str = None,
    ) -> None:
        """Render a path and save each frame as a PNG file.

        Output filenames follow the same convention as :class:`BlenderRenderer`:
        ``frame0001.png``, ``frame0002.png``, etc.

        Args:
            positions: List of (x, y) coordinate tuples.
            head_directions: List of head directions in radians.
            output_dir: Directory for output PNGs.  Defaults to
                ``self.config['output_dir']``.
        """
        from PIL import Image

        if output_dir is None:
            output_dir = self.config["output_dir"]
        os.makedirs(output_dir, exist_ok=True)

        frames = self.render_path(positions, head_directions)
        n_frames = frames.shape[0]
        digits = len(str(n_frames))

        for i in range(n_frames):
            fname = f"frame{str(i + 1).zfill(digits)}.png"
            img_array = (frames[i] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_array, mode="L").save(os.path.join(output_dir, fname))

        print(f'[+] saved {n_frames} frames to "{output_dir}"')
