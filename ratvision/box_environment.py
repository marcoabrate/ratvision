"""Box environment definition and procedural landmark generators.

This module defines the `BoxEnvironment` and `Landmark` dataclasses that
describe the scene rendered by the ratvision renderers, as well as helper
functions for loading textures and creating procedural landmarks (circles,
triangles, striped rectangles).

A convenience function `default_box_environment()` builds the standard
0.635 m box with real-photo wall and floor textures and three geometric
landmarks.
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from importlib.resources import files

# ---------------------------------------------------------------------------
# Data classes for environment definition
# ---------------------------------------------------------------------------


@dataclass
class Landmark:
    """A landmark placed on a wall, defined by an analytical shape function.

    Attributes:
        wall_index: Which wall the landmark is on (0-3).
        uv_min: (u_min, v_min) bottom-left corner in wall-local UV coords [0,1]².
        uv_max: (u_max, v_max) top-right corner in wall-local UV coords [0,1]².
        shape_fn: A callable ``(u, v) -> (color, alpha)`` where *u*, *v* are
            ``np.ndarray`` of landmark-local coordinates in [0, 1]².
            Returns *color* and *alpha* arrays of the same shape, both in [0, 1].
    """

    wall_index: int
    uv_min: Tuple[float, float]
    uv_max: Tuple[float, float]
    shape_fn: (
        object  # Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]
    )


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


def _gaussian_kernel1d(sigma, radius):
    """
    Computes a 1-D Gaussian convolution kernel.
    """
    sigma2 = sigma * sigma
    x = np.arange(-radius, radius + 1)
    phi_x = np.exp(-0.5 / sigma2 * x**2)
    phi_x = phi_x / phi_x.sum()

    return phi_x


def _gaussian_filter(img, sigma, truncate=4.0):
    sd = float(sigma)
    radius = int(truncate * sd + 0.5)
    kernel = _gaussian_kernel1d(sd, radius)

    img_out = img.copy()
    for axis in range(img_out.ndim):
        # Pad with reflect mode along the current axis
        pad_width = [(0, 0)] * img_out.ndim
        pad_width[axis] = (radius, radius)
        img_padded = np.pad(img_out, pad_width, mode="symmetric")

        img_out = np.apply_along_axis(
            lambda x: np.convolve(x, kernel, mode="valid"), axis, img_padded
        )
    return img_out


def _load_texture(path, blur_sigma=3) -> np.ndarray:
    """
    Load an image file and return a grayscale numpy array in [0, 1].

    Args:
        path: Path to the image file.
        blur_sigma: If > 0, apply a Gaussian blur with this sigma
            (in pixels) to the loaded image.

    Returns:
        A 2-D numpy array of shape ``(H, W)`` with values in [0, 1].
    """
    from PIL import Image

    img = Image.open(path).convert("L")
    img_np = np.asarray(img, dtype=np.float64) / 255.0
    if blur_sigma > 0:
        img_np = _gaussian_filter(img_np, sigma=blur_sigma)
    return img_np


# ---------------------------------------------------------------------------
# Procedural landmark generators
# ---------------------------------------------------------------------------


def _compute_uv_region(
    region_w: float,
    region_h: float,
    wall_u_extent: float,
    wall_v_extent: float,
    position: str = "centre",
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Compute UV min/max for a landmark region on a wall.

    Args:
        region_w: Width of the region in metres.
        region_h: Height of the region in metres.
        wall_u_extent: Physical width of the wall (metres).
        wall_v_extent: Physical height of the wall (metres).
        position: Horizontal placement — ``'left'``, ``'centre'``/``'center'``,
            or ``'right'``.

    Returns:
        ``(uv_min, uv_max)`` tuples.
    """
    region_u = region_w / wall_u_extent
    region_v = region_h / wall_v_extent

    if position == "left":
        u_min = 0.0
        u_max = region_u
    elif position == "right":
        u_min = 1.0 - region_u
        u_max = 1.0
    else:  # 'centre' / 'center'
        u_min = 0.5 - region_u / 2
        u_max = 0.5 + region_u / 2

    v_min = 0.5 - region_v / 2
    v_max = 0.5 + region_v / 2
    return (u_min, v_min), (u_max, v_max)


def _make_circle_ring_landmark(
    wall_index: int,
    wall_u_extent: float,
    wall_v_extent: float,
    diameter: float = 0.5,
    stroke: float = 0.1,
    color: float = 1.0,
    position: str = "centre",
) -> Landmark:
    """Create a circle ring (annulus) landmark on a wall.

    Args:
        wall_index: Index of the wall to place the landmark on.
        wall_u_extent: Physical width of the wall along its u-axis (metres).
        wall_v_extent: Physical height of the wall (metres).
        diameter: Outer diameter of the circle (metres).
        stroke: Width of the ring stroke (metres).
        color: Grayscale fill of the ring (0 = black, 1 = white).
        position: Horizontal placement — ``'left'``, ``'centre'``/``'center'``,
            or ``'right'``.
    """
    region_w = min(diameter, wall_u_extent)
    region_h = min(diameter, wall_v_extent)
    uv_min, uv_max = _compute_uv_region(
        region_w,
        region_h,
        wall_u_extent,
        wall_v_extent,
        position,
    )

    outer_r = diameter / 2
    inner_r = outer_r - stroke

    def shape_fn(u: np.ndarray, v: np.ndarray):
        x = (u - 0.5) * region_w
        y = (v - 0.5) * region_h
        dist = np.sqrt(x**2 + y**2)
        ring = (dist >= inner_r) & (dist <= outer_r)
        c = np.full_like(u, color)
        a = ring.astype(np.float64)
        return c, a

    return Landmark(
        wall_index=wall_index, uv_min=uv_min, uv_max=uv_max, shape_fn=shape_fn
    )


def _make_triangle_landmark(
    wall_index: int,
    wall_u_extent: float,
    wall_v_extent: float,
    tri_height: float = 0.5,
    color: float = 0.0,
    position: str = "right",
) -> Landmark:
    """Create an equilateral triangle landmark on a wall.

    The triangle points upward with its base at the bottom.

    Args:
        wall_index: Index of the wall.
        wall_u_extent: Physical width of the wall along its u-axis (metres).
        wall_v_extent: Physical height of the wall (metres).
        tri_height: Height of the triangle (metres).
        color: Grayscale fill (0 = black, 1 = white).
        position: Horizontal placement — ``'left'``, ``'centre'``/``'center'``,
            or ``'right'``.
    """
    base = 2 * tri_height / math.sqrt(3)
    region_w = min(base, wall_u_extent)
    region_h = min(tri_height, wall_v_extent)
    uv_min, uv_max = _compute_uv_region(
        region_w,
        region_h,
        wall_u_extent,
        wall_v_extent,
        position,
    )

    def shape_fn(u: np.ndarray, v: np.ndarray):
        # Equilateral triangle: base at bottom (v=0), apex at top (v=1)
        inside = (u >= 0.5 * v) & (u <= 1.0 - 0.5 * v)
        c = np.full_like(u, color)
        a = inside.astype(np.float64)
        return c, a

    return Landmark(
        wall_index=wall_index, uv_min=uv_min, uv_max=uv_max, shape_fn=shape_fn
    )


def _make_striped_rect_landmark(
    wall_index: int,
    wall_u_extent: float,
    wall_v_extent: float,
    rect_width: float = 0.4,
    rect_height: float = 0.3,
    stripe_width: float = 0.1,
    position: str = "centre",
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
        position: Horizontal placement — ``'left'``, ``'centre'``/``'center'``,
            or ``'right'``.
    """
    region_w = min(rect_width, wall_u_extent)
    region_h = min(rect_height, wall_v_extent)
    uv_min, uv_max = _compute_uv_region(
        region_w,
        region_h,
        wall_u_extent,
        wall_v_extent,
        position,
    )

    def shape_fn(u: np.ndarray, v: np.ndarray):
        # Map [0,1]^2 to world-space coordinates inside the rectangle
        x = u * region_w
        y = v * region_h
        # 45° stripes: project onto (1,1)/sqrt(2)
        d = (x + y) / math.sqrt(2)
        stripe_idx = np.floor(d / stripe_width).astype(int) % 2
        c = np.where(stripe_idx == 0, 0.0, 1.0)
        a = np.ones_like(u)
        return c, a

    return Landmark(
        wall_index=wall_index, uv_min=uv_min, uv_max=uv_max, shape_fn=shape_fn
    )


def default_box_environment() -> BoxEnvironment:
    """Create the default 0.635 m box environment with real textures.

    Loads wall textures, floor texture, and landmark images from the
    ``ratvision.environments`` package data directory.

    Returns:
        A fully configured :class:`BoxEnvironment`.
    """
    env_dir = files("ratvision.environments")

    # Load wall textures (one per wall: north, south, east, west)
    wall_files = [
        "wall_concrete1_635x500.png",  # wall 0 (north)
        "wall_texture_635x500.png",  # wall 1 (south)
        "wall_concrete2_635x500.png",  # wall 2 (east)
        "wall_concrete3_635x500.png",  # wall 3 (west)
    ]
    wall_textures = [_load_texture(env_dir.joinpath(f)) for f in wall_files]

    # Load floor texture
    floor_texture = _load_texture(env_dir.joinpath("floor_texture_635x635.png"))

    # Generate geometric landmarks
    length, height = 0.635, 0.5

    landmarks = [
        # Diagonally striped rectangle centred on wall 0 (north)
        _make_striped_rect_landmark(
            wall_index=0,
            wall_u_extent=length,
            wall_v_extent=height,
            rect_width=0.4,
            rect_height=0.3,
            stripe_width=0.1,
        ),
        # White circle ring centred on wall 1 (south)
        _make_circle_ring_landmark(
            wall_index=1,
            wall_u_extent=length,
            wall_v_extent=height,
            diameter=0.4,
            stroke=0.07,
            color=1.0,
        ),
        # Black equilateral triangle on the side of wall 2 (east)
        _make_triangle_landmark(
            wall_index=2,
            wall_u_extent=length,
            wall_v_extent=height,
            tri_height=0.4,
            color=0.0,
            position="left",
        ),
    ]

    return BoxEnvironment(
        width=length,
        depth=length,
        height=height,
        wall_textures=wall_textures,
        floor_texture=floor_texture,
        ceiling_color=0.0,
        landmarks=landmarks,
    )
