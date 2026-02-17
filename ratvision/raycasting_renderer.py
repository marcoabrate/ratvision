import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

import numpy as np

if sys.version_info < (3, 9):
    import importlib_resources
    files = importlib_resources.files
else:
    from importlib.resources import files

import matplotlib.pyplot as plt
from matplotlib import animation


# ---------------------------------------------------------------------------
# Data classes for environment definition
# ---------------------------------------------------------------------------

@dataclass
class Landmark:
    """A textured landmark placed on a wall.

    Attributes:
        wall_index: Which wall the landmark is on (0-3).
        uv_min: (u_min, v_min) bottom-left corner in wall-local UV coords [0,1]².
        uv_max: (u_max, v_max) top-right corner in wall-local UV coords [0,1]².
        texture: Grayscale texture as np.ndarray with values in [0, 1].
        alpha: Optional alpha mask (same shape as texture). If None, all
            non-zero texture pixels are considered opaque.
    """
    wall_index: int
    uv_min: Tuple[float, float]
    uv_max: Tuple[float, float]
    texture: np.ndarray
    alpha: Optional[np.ndarray] = None


@dataclass
class BoxEnvironment:
    """A simple axis-aligned box environment.

    The box origin is at one corner, so coordinates go from
    (0, 0) to (width, depth), and the floor is at z = 0.

    Wall indexing (looking down from above):
        - Wall 0 (north): y = depth, faces -Y
        - Wall 1 (south): y = 0,     faces +Y
        - Wall 2 (east):  x = width, faces -X
        - Wall 3 (west):  x = 0,     faces +X

    Attributes:
        width: Box extent along the x-axis (metres).
        depth: Box extent along the y-axis (metres).
        height: Wall height along the z-axis (metres).
        wall_textures: List of 4 grayscale textures (np.ndarray, values in [0,1]),
            one per wall following the wall index convention.
            If None, flat ``wall_color`` is used for all walls.
        floor_texture: Grayscale floor texture. If None, flat ``floor_color`` is used.
        wall_color: Flat grayscale value used when ``wall_textures`` is None.
        floor_color: Flat grayscale value used when ``floor_texture`` is None.
        ceiling_color: Flat grayscale value for the ceiling.
        landmarks: Landmarks placed on walls.
    """
    width: float = 0.635
    depth: float = 0.635
    height: float = 0.5
    wall_textures: Optional[List[np.ndarray]] = None
    floor_texture: Optional[np.ndarray] = None
    wall_color: float = 0.5
    floor_color: float = 0.3
    ceiling_color: float = 0.0
    landmarks: List[Landmark] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Texture loading helpers
# ---------------------------------------------------------------------------

def _load_texture(path) -> np.ndarray:
    """Load an image file and return a grayscale numpy array in [0, 1]."""
    from PIL import Image
    img = Image.open(path).convert('L')
    return np.asarray(img, dtype=np.float64) / 255.0


def _load_texture_with_alpha(path) -> Tuple[np.ndarray, np.ndarray]:
    """Load an image and return (grayscale, alpha) arrays in [0, 1].

    If the image has no alpha channel, the alpha mask is set to 1
    wherever the grayscale value is > 0.
    """
    from PIL import Image
    img = Image.open(path)
    if img.mode == 'RGBA':
        arr = np.asarray(img, dtype=np.float64) / 255.0
        gray = 0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        alpha = arr[..., 3]
    elif img.mode == 'LA':
        arr = np.asarray(img, dtype=np.float64) / 255.0
        gray = arr[..., 0]
        alpha = arr[..., 1]
    else:
        gray = np.asarray(img.convert('L'), dtype=np.float64) / 255.0
        alpha = (gray > 0).astype(np.float64)
    return gray, alpha


# ---------------------------------------------------------------------------
# Procedural landmark generators
# ---------------------------------------------------------------------------

def _make_circle_ring_landmark(
    wall_index: int,
    wall_u_extent: float,
    wall_v_extent: float,
    diameter: float = 0.5,
    stroke: float = 0.1,
    color: float = 1.0,
    tex_res: int = 256,
) -> Landmark:
    """Create a circle ring (annulus) landmark centred on a wall.

    Args:
        wall_index: Index of the wall to place the landmark on.
        wall_u_extent: Physical width of the wall along its u-axis (metres).
        wall_v_extent: Physical height of the wall (metres).
        diameter: Outer diameter of the circle (metres).
        stroke: Width of the ring stroke (metres).
        color: Grayscale fill of the ring (0 = black, 1 = white).
        tex_res: Texture resolution in pixels (square).
    """
    region_w = min(diameter, wall_u_extent)
    region_h = min(diameter, wall_v_extent)

    u_half = region_w / (2 * wall_u_extent)
    v_half = region_h / (2 * wall_v_extent)
    uv_min = (0.5 - u_half, 0.5 - v_half)
    uv_max = (0.5 + u_half, 0.5 + v_half)

    cols = np.arange(tex_res)
    rows = np.arange(tex_res)
    cc, rr = np.meshgrid(cols, rows)

    # World-space coordinates centred at circle centre
    x = (cc / (tex_res - 1) - 0.5) * region_w
    y = (0.5 - rr / (tex_res - 1)) * region_h

    dist = np.sqrt(x ** 2 + y ** 2)
    outer_r = diameter / 2
    inner_r = outer_r - stroke

    ring = (dist >= inner_r) & (dist <= outer_r)
    texture = np.full((tex_res, tex_res), color, dtype=np.float64)
    alpha = ring.astype(np.float64)

    return Landmark(
        wall_index=wall_index,
        uv_min=uv_min,
        uv_max=uv_max,
        texture=texture,
        alpha=alpha,
    )


def _make_triangle_landmark(
    wall_index: int,
    wall_u_extent: float,
    wall_v_extent: float,
    tri_height: float = 0.5,
    color: float = 0.0,
    side: str = 'right',
    tex_res: int = 256,
) -> Landmark:
    """Create an equilateral triangle landmark on a wall.

    The triangle points upward with its base at the bottom.

    Args:
        wall_index: Index of the wall.
        wall_u_extent: Physical width of the wall along its u-axis (metres).
        wall_v_extent: Physical height of the wall (metres).
        tri_height: Height of the triangle (metres).
        color: Grayscale fill (0 = black, 1 = white).
        side: Horizontal placement — ``'left'``, ``'right'``, or ``'center'``.
        tex_res: Texture resolution in pixels (square).
    """
    base = 2 * tri_height / math.sqrt(3)
    region_w = min(base, wall_u_extent)
    region_h = min(tri_height, wall_v_extent)

    region_u = region_w / wall_u_extent
    region_v = region_h / wall_v_extent

    if side == 'right':
        uv_min = (1.0 - region_u, 0.5 - region_v / 2)
        uv_max = (1.0, 0.5 + region_v / 2)
    elif side == 'left':
        uv_min = (0.0, 0.5 - region_v / 2)
        uv_max = (region_u, 0.5 + region_v / 2)
    else:
        uv_min = (0.5 - region_u / 2, 0.5 - region_v / 2)
        uv_max = (0.5 + region_u / 2, 0.5 + region_v / 2)

    cols = np.arange(tex_res)
    rows = np.arange(tex_res)
    cc, rr = np.meshgrid(cols, rows)

    u = cc / (tex_res - 1)
    v = 1.0 - rr / (tex_res - 1)

    # Equilateral triangle: base at bottom (v=0), apex at top (v=1)
    inside = (u >= 0.5 * v) & (u <= 1.0 - 0.5 * v)

    texture = np.full((tex_res, tex_res), color, dtype=np.float64)
    alpha = inside.astype(np.float64)

    return Landmark(
        wall_index=wall_index,
        uv_min=uv_min,
        uv_max=uv_max,
        texture=texture,
        alpha=alpha,
    )


def _make_striped_rect_landmark(
    wall_index: int,
    wall_u_extent: float,
    wall_v_extent: float,
    rect_width: float = 0.3,
    rect_height: float = 0.5,
    stripe_width: float = 0.1,
    tex_res: int = 256,
) -> Landmark:
    """Create a rectangle with diagonal black-and-white stripes.

    Stripes are at 45 degrees in world space, with the specified
    perpendicular width.

    Args:
        wall_index: Index of the wall.
        wall_u_extent: Physical width of the wall along its u-axis (metres).
        wall_v_extent: Physical height of the wall (metres).
        rect_width: Width of the rectangle (metres).
        rect_height: Height of the rectangle (metres).
        stripe_width: Perpendicular width of each stripe (metres).
        tex_res: Texture resolution in pixels (square).
    """
    region_w = min(rect_width, wall_u_extent)
    region_h = min(rect_height, wall_v_extent)

    u_half = region_w / (2 * wall_u_extent)
    v_half = region_h / (2 * wall_v_extent)
    uv_min = (0.5 - u_half, 0.5 - v_half)
    uv_max = (0.5 + u_half, 0.5 + v_half)

    cols = np.arange(tex_res)
    rows = np.arange(tex_res)
    cc, rr = np.meshgrid(cols, rows)

    # World-space coordinates inside the rectangle
    x = cc / (tex_res - 1) * region_w
    y = (1.0 - rr / (tex_res - 1)) * region_h

    # 45° stripes: project onto the perpendicular direction (1,1)/√2
    d = (x + y) / math.sqrt(2)
    stripe_idx = np.floor(d / stripe_width).astype(int) % 2

    texture = np.where(stripe_idx == 0, 1.0, 0.0)
    alpha = np.ones((tex_res, tex_res), dtype=np.float64)

    return Landmark(
        wall_index=wall_index,
        uv_min=uv_min,
        uv_max=uv_max,
        texture=texture,
        alpha=alpha,
    )


def default_box_environment() -> BoxEnvironment:
    """Create the default 0.635 m box environment with real textures.

    Loads wall textures, floor texture, and landmark images from the
    ``ratvision.environments`` package data directory.

    Returns:
        A fully configured :class:`BoxEnvironment`.
    """
    env_dir = files('ratvision.environments')

    # Load wall textures (one per wall: north, south, east, west)
    wall_files = [
        'wall_texture.jpg',      # wall 0 (north)
        'wall_concrete1.jpg',    # wall 1 (south)
        'wall_concrete2.jpg',    # wall 2 (east)
        'wall_concrete3.webp',   # wall 3 (west)
    ]
    wall_textures = [_load_texture(env_dir.joinpath(f)) for f in wall_files]

    # Load floor texture
    floor_texture = _load_texture(env_dir.joinpath('floor_texture.webp'))

    # Generate geometric landmarks
    # Walls 0,1 have u_extent = width; walls 2,3 have u_extent = depth
    w, d, h = 0.635, 0.635, 0.5

    landmarks = [
        # White circle ring centred on wall 0 (north)
        _make_circle_ring_landmark(
            wall_index=0, wall_u_extent=w, wall_v_extent=h,
            diameter=0.5, stroke=0.1, color=1.0,
        ),
        # Black equilateral triangle on the side of wall 1 (south)
        _make_triangle_landmark(
            wall_index=1, wall_u_extent=w, wall_v_extent=h,
            tri_height=0.5, color=0.0, side='right',
        ),
        # Diagonally striped rectangle centred on wall 2 (east)
        _make_striped_rect_landmark(
            wall_index=2, wall_u_extent=d, wall_v_extent=h,
            rect_width=0.3, rect_height=0.5, stripe_width=0.1,
        ),
    ]

    return BoxEnvironment(
        width=0.635,
        depth=0.635,
        height=0.5,
        wall_textures=wall_textures,
        floor_texture=floor_texture,
        ceiling_color=0.0,
        landmarks=landmarks,
    )


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
        texture[y0, x0] * (1 - fx) * (1 - fy) +
        texture[y0, x1] * fx * (1 - fy) +
        texture[y1, x0] * (1 - fx) * fy +
        texture[y1, x1] * fx * fy
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
    :class:`ratvision.Renderer` for simple box environments.
    Typical throughput is >1000 frames/s at 32×16 resolution on CPU.

    Example::

        from ratvision import RaycastingRenderer

        renderer = RaycastingRenderer()            # default box
        frame = renderer.render_frame(0.3, 0.3, 0) # (H, W) numpy array
    """

    DEFAULT_CONFIG = {
        'frame_dim': (128, 64),
        'camera_height': 0.035,
        'hfov': 4 * math.pi / 3,   # 240 degrees
        'vfov': 2 * math.pi / 3,   # 120 degrees
        'output_dir': None,
    }

    CONFIG_DESCRIPTION = {
        'frame_dim': 'Dimensions of the rendered frames (width, height), in pixels.',
        'camera_height': 'Height of the camera from the ground in metres. Default is 0.035 m.',
        'hfov': 'Horizontal field of view in radians.  Default is 4π/3 (240°).',
        'vfov': 'Vertical field of view in radians.  Default is 2π/3 (120°).',
        'output_dir': 'Path where rendered images will be saved. If None, "./output" is used.',
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
            print('[*] no environment provided, loading default box environment.')
            self.env = default_box_environment()
        else:
            self.env = env

        self.config = self.DEFAULT_CONFIG.copy()
        if config is not None:
            self.update_config(config)
        else:
            print('[*] no configuration provided, using default.')
            self._print_config_message()

        if self.config['output_dir'] is None:
            self.config['output_dir'] = os.path.join(os.getcwd(), 'output')

        # pre-compute per-pixel angular offsets
        self._precompute_rays()

    # ------------------------------------------------------------------
    # Configuration helpers (same pattern as Renderer)
    # ------------------------------------------------------------------

    @staticmethod
    def _print_config_message() -> None:
        print()
        print('you can check the configuration description by calling')
        print('the "config_description" function or by checking the documentation,')
        print('and update the configuration by calling the "update_config" function.')
        print()

    @staticmethod
    def config_description() -> None:
        """Print a description of each configuration key."""
        print('[*] configuration description:')
        for key, value in RaycastingRenderer.CONFIG_DESCRIPTION.items():
            print(f'\t{key}: {value}')
        RaycastingRenderer._print_config_message()

    def print_config(self) -> None:
        """Print the current configuration."""
        print('[*] current configuration:')
        for key, value in self.config.items():
            print(f'\t{key}: {value}')
        self._print_config_message()

    def update_config(self, config: Dict) -> None:
        """Update configuration with new values. Only known keys are applied.

        Args:
            config: Dictionary of configuration overrides.
        """
        if not isinstance(config, dict):
            raise ValueError('config must be a dictionary.')
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f'[-] {key} is not a valid configuration key, skipping.')

        # re-compute rays if resolution or FOV changed
        self._precompute_rays()

    # ------------------------------------------------------------------
    # Ray generation (equirectangular projection)
    # ------------------------------------------------------------------

    def _precompute_rays(self) -> None:
        """Pre-compute per-pixel azimuth and elevation offsets."""
        W, H = self.config['frame_dim']
        hfov = self.config['hfov']
        vfov = self.config['vfov']

        # column → azimuth offset (left of centre is positive = CCW)
        col = np.arange(W, dtype=np.float64)
        delta_az = hfov * (0.5 - col / max(W - 1, 1))

        # row → elevation offset (top of image is positive = up)
        row = np.arange(H, dtype=np.float64)
        delta_el = vfov * (0.5 - row / max(H - 1, 1))

        # store as (H, W) broadcast-ready arrays
        self._delta_az = delta_az[np.newaxis, :]       # (1, W)
        self._delta_el = delta_el[:, np.newaxis]        # (H, 1)
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
            np.array([0.0, -1.0, 0.0]),     # normal (inward)
            np.array([0.0, d, 0.0]),         # wall_origin (u=0, v=0)
            np.array([1.0, 0.0, 0.0]),       # u_axis
            np.array([0.0, 0.0, 1.0]),       # v_axis
            w, h, 0
        )
        # Wall 1 (south): y = 0, faces +Y
        # Looking at this wall from inside, left is -X direction.
        wall1 = (
            np.array([0.0, 1.0, 0.0]),
            np.array([w, 0.0, 0.0]),
            np.array([-1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            w, h, 1
        )
        # Wall 2 (east): x = width, faces -X
        # Looking from inside, left is +Y direction.
        wall2 = (
            np.array([-1.0, 0.0, 0.0]),
            np.array([w, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            d, h, 2
        )
        # Wall 3 (west): x = 0, faces +X
        # Looking from inside, left is -Y direction.
        wall3 = (
            np.array([1.0, 0.0, 0.0]),
            np.array([0.0, d, 0.0]),
            np.array([0.0, -1.0, 0.0]),
            np.array([0.0, 0.0, 1.0]),
            d, h, 3
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
        cam_z = self.config['camera_height']
        origin = np.array([x, y, cam_z])

        # per-pixel world-space ray directions (equirectangular)
        az = theta + self._delta_az   # (1, W) broadcast with scalar
        el = self._delta_el           # (H, 1)

        cos_el = np.cos(el)
        # theta=0 → north (+Y);  positive theta → CCW
        dx = -np.sin(az) * cos_el     # (H, W)
        dy =  np.cos(az) * cos_el     # (H, W)
        dz =  np.sin(el) * np.ones_like(dx)  # (H, W)

        # output image and depth buffer
        image = np.full((H, W), self.env.ceiling_color, dtype=np.float64)
        closest_t = np.full((H, W), np.inf)

        # ---- intersect walls ----
        walls = self._wall_planes()
        for normal, wall_origin, u_axis, v_axis, u_ext, v_ext, wall_idx in walls:
            # plane equation: dot(normal, P) = dot(normal, wall_origin)
            d_plane = np.dot(normal, wall_origin)
            denom = normal[0] * dx + normal[1] * dy + normal[2] * dz
            numer = d_plane - (normal[0] * origin[0] + normal[1] * origin[1] + normal[2] * origin[2])

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

            on_wall = hit_mask & (u_coord >= 0) & (u_coord <= 1) & (v_coord >= 0) & (v_coord <= 1)
            if not np.any(on_wall):
                continue

            # sample wall texture or flat colour
            if self.env.wall_textures is not None and wall_idx < len(self.env.wall_textures):
                color = _sample_texture(self.env.wall_textures[wall_idx], u_coord, v_coord)
            else:
                color = np.full((H, W), self.env.wall_color)

            # overlay landmarks
            for lm in self.env.landmarks:
                if lm.wall_index != wall_idx:
                    continue
                lm_u = (u_coord - lm.uv_min[0]) / (lm.uv_max[0] - lm.uv_min[0])
                lm_v = (v_coord - lm.uv_min[1]) / (lm.uv_max[1] - lm.uv_min[1])
                in_bounds = (lm_u >= 0) & (lm_u <= 1) & (lm_v >= 0) & (lm_v <= 1)
                lm_color = _sample_texture(lm.texture, lm_u, lm_v)
                if lm.alpha is not None:
                    lm_alpha = _sample_texture(lm.alpha, lm_u, lm_v)
                else:
                    lm_alpha = (lm_color > 0).astype(np.float64)
                blend_mask = on_wall & in_bounds
                color = np.where(
                    blend_mask,
                    lm_alpha * lm_color + (1 - lm_alpha) * color,
                    color,
                )

            image[on_wall] = color[on_wall]
            closest_t[on_wall] = t_hit[on_wall]

        # ---- intersect floor (z = 0) ----
        with np.errstate(divide='ignore', invalid='ignore'):
            t_floor = np.where(
                np.abs(dz) > 1e-12,
                -cam_z / dz,
                np.inf,
            )
        fx = origin[0] + t_floor * dx
        fy = origin[1] + t_floor * dy
        floor_hit = (
            (t_floor > 1e-6) & (t_floor < closest_t) &
            (fx >= 0) & (fx <= self.env.width) &
            (fy >= 0) & (fy <= self.env.depth)
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
        with np.errstate(divide='ignore', invalid='ignore'):
            t_ceil = np.where(
                np.abs(dz) > 1e-12,
                (self.env.height - cam_z) / dz,
                np.inf,
            )
        cx = origin[0] + t_ceil * dx
        cy = origin[1] + t_ceil * dy
        ceil_hit = (
            (t_ceil > 1e-6) & (t_ceil < closest_t) &
            (cx >= 0) & (cx <= self.env.width) &
            (cy >= 0) & (cy <= self.env.depth)
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
            raise TypeError('positions and head_directions must be lists.')
        if len(positions) != len(head_directions):
            raise ValueError('positions and head_directions must have the same number of elements.')
        if any(hd < -2 * math.pi or hd > 2 * math.pi for hd in head_directions):
            print('[!!!] remember that head_directions should be in radians.')

        n = len(positions)
        frames = np.empty((n, self._frame_H, self._frame_W), dtype=np.float64)
        for i in range(n):
            frames[i] = self.render_frame(positions[i][0], positions[i][1], head_directions[i])
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

        Output filenames follow the same convention as :class:`Renderer`:
        ``frame0001.png``, ``frame0002.png``, etc.

        Args:
            positions: List of (x, y) coordinate tuples.
            head_directions: List of head directions in radians.
            output_dir: Directory for output PNGs.  Defaults to
                ``self.config['output_dir']``.
        """
        from PIL import Image

        if output_dir is None:
            output_dir = self.config['output_dir']
        os.makedirs(output_dir, exist_ok=True)

        frames = self.render_path(positions, head_directions)
        n_frames = frames.shape[0]
        digits = len(str(n_frames))

        for i in range(n_frames):
            fname = f'frame{str(i + 1).zfill(digits)}.png'
            img_array = (frames[i] * 255).clip(0, 255).astype(np.uint8)
            Image.fromarray(img_array, mode='L').save(os.path.join(output_dir, fname))

        print(f'[+] saved {n_frames} frames to "{output_dir}"')

    def get_video_animation(self, frames: np.ndarray = None, fps: int = 10) -> animation.FuncAnimation:
        """Create a matplotlib animation from rendered frames on disk.

        Frames are loaded from ``self.config['output_dir']``.

        Args:
            frames: Pre-rendered frames as a numpy array of shape (N, H, W).
            fps: Frames per second for the animation.

        Returns:
            ``matplotlib.animation.FuncAnimation`` that can be saved
            with ``.save("filename.mp4")`` or displayed in a notebook.
        """

        if frames is None:
            from PIL import Image

            output_dir = self.config['output_dir']
            if not os.path.exists(output_dir):
                print(f'[-] output directory "{output_dir}" does not exist.')
                print('you first need to render or save frames.')
                return None

            frame_files = sorted([
                os.path.join(output_dir, f) for f in os.listdir(output_dir)
                if os.path.isfile(os.path.join(output_dir, f))
            ])

            if len(frame_files) == 0:
                print(f'[-] no frames found in "{output_dir}".')
                return None

            frames = [Image.open(ff) for ff in frame_files]

        print(f'[+] animating {len(frames)} frames at {fps} fps...')

        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
        im = ax.imshow(frames[0], cmap='gray')
        plt.axis('off')
        plt.close()

        def init():
            im.set_data(frames[0])

        def animate(i):
            im.set_data(frames[i])
            return im

        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frames),
            interval=1_000 / fps,
        )
        return anim
