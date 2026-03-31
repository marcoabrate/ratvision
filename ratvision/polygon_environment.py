"""Polygon environment definition for arbitrary 2D polygon layouts.

This module defines the :class:`PolygonEnvironment` dataclass that describes
an environment whose floor plan is an arbitrary convex, concave, or irregular
polygon with 3–10 vertices.  Each edge of the polygon becomes a vertical wall.

Wall and landmark textures are cycled (wrapped with modulo) when the number
of polygon edges exceeds the number of provided textures or landmarks.

The environment can be consumed by raycasting and torch renderers that
support polygon wall planes (see :meth:`PolygonEnvironment.wall_planes`).
"""

import math
from typing import List, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
from importlib.resources import files

from .box_environment import (
    Landmark,
    _load_texture,
    _compute_uv_region,
    _make_circle_ring_landmark,
    _make_triangle_landmark,
    _make_striped_rect_landmark,
)


# ---------------------------------------------------------------------------
# Data class for polygon environment
# ---------------------------------------------------------------------------


@dataclass
class PolygonEnvironment:
    """An environment whose floor plan is an arbitrary 2-D polygon.

    Vertices are given as a list of ``(x, y)`` pairs in counter-clockwise
    order.  Each consecutive edge ``vertices[i] -> vertices[(i+1) % N]``
    defines a vertical wall.  The polygon must be simple (no
    self-intersections).

    Wall indexing follows edge order: wall *i* connects vertex *i* to
    vertex *(i + 1) % N*.  The inward-facing normal for a CCW polygon
    points to the right of the edge direction.

    Attributes:
        vertices: 2-D polygon vertices as ``[(x0, y0), …]``, counter-clockwise.
        height: Wall height along the z-axis (metres).
        wall_textures: Grayscale textures (``np.ndarray``, values in [0, 1]),
            one per wall.  If fewer textures than walls are provided they
            are cycled with modulo indexing.  ``None`` uses flat ``wall_color``.
        floor_texture: Grayscale floor texture mapped to the polygon's
            bounding box.  ``None`` uses flat ``floor_color``.
        wall_color: Flat grayscale value used when ``wall_textures`` is None.
        floor_color: Flat grayscale value used when ``floor_texture`` is None.
        ceiling_color: Flat grayscale value for the ceiling.
        landmarks: Landmarks placed on walls.  When walls > landmarks the
            landmark list is cycled with modulo indexing.
    """

    vertices: List[Tuple[float, float]]
    height: float = 0.5
    wall_textures: Optional[List[np.ndarray]] = None
    floor_texture: Optional[np.ndarray] = None
    wall_color: float = 0.5
    floor_color: float = 0.3
    ceiling_color: float = 0.0
    landmarks: List[Landmark] = field(default_factory=list)

    # -- derived properties --------------------------------------------------

    @property
    def n_walls(self) -> int:
        """Number of walls (= number of vertices)."""
        return len(self.vertices)

    @property
    def bounding_box(self) -> Tuple[float, float, float, float]:
        """Axis-aligned bounding box as ``(x_min, y_min, x_max, y_max)``."""
        verts = np.asarray(self.vertices)
        return (
            float(verts[:, 0].min()),
            float(verts[:, 1].min()),
            float(verts[:, 0].max()),
            float(verts[:, 1].max()),
        )

    @property
    def bb_width(self) -> float:
        """Bounding-box width (x extent)."""
        x_min, _, x_max, _ = self.bounding_box
        return x_max - x_min

    @property
    def bb_depth(self) -> float:
        """Bounding-box depth (y extent)."""
        _, y_min, _, y_max = self.bounding_box
        return y_max - y_min

    # -- wall geometry -------------------------------------------------------

    def wall_planes(
        self,
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, int]]:
        """Compute wall plane definitions for all polygon edges.

        Returns a list of tuples, one per wall::

            (normal, wall_origin, u_axis, v_axis, u_extent, v_extent, wall_index)

        where

        * *normal* – inward-facing unit normal (3-D).
        * *wall_origin* – 3-D point where ``(u, v) = (0, 0)`` on the wall.
        * *u_axis* – unit direction along the wall (from vertex *i* to
          vertex *i + 1*).
        * *v_axis* – ``[0, 0, 1]`` (vertical).
        * *u_extent* – physical wall width (edge length in metres).
        * *v_extent* – physical wall height (``self.height``).
        * *wall_index* – integer index of the wall.
        """
        h = self.height
        verts = self.vertices
        n = len(verts)
        planes = []
        for i in range(n):
            p0 = np.array(verts[i], dtype=np.float64)
            p1 = np.array(verts[(i + 1) % n], dtype=np.float64)
            edge = p1 - p0
            length = float(np.linalg.norm(edge))
            if length < 1e-12:
                continue

            # u-axis: unit direction along the edge
            u_dir = edge / length
            u_axis = np.array([u_dir[0], u_dir[1], 0.0])

            # v-axis: vertical
            v_axis = np.array([0.0, 0.0, 1.0])

            # Inward-facing normal for CCW polygon: rotate edge 90° clockwise
            # edge (dx, dy) → normal (dy, -dx), then negate for inward = (-dy, dx)
            # Actually for CCW winding, the inward normal is to the RIGHT of
            # the edge direction, which is (dy, -dx).  But let's verify:
            # For a CCW square with bottom edge from (0,0)->(1,0),
            # edge=(1,0), right normal=(0,-1) which points outward (down).
            # Left normal = (0,1) which points inward (up into the polygon).
            # So inward normal = (-dy, dx) for CCW winding.
            nx = -u_dir[1]
            ny = u_dir[0]
            normal = np.array([nx, ny, 0.0])

            # wall_origin: vertex i at z=0
            wall_origin = np.array([p0[0], p0[1], 0.0])

            planes.append((normal, wall_origin, u_axis, v_axis, length, h, i))

        return planes

    # -- point-in-polygon ----------------------------------------------------

    def point_in_polygon(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Test whether 2-D points lie inside the polygon (ray-casting algorithm).

        Args:
            x: Array of x-coordinates (any shape).
            y: Array of y-coordinates (same shape as *x*).

        Returns:
            Boolean array of the same shape, ``True`` where the point is
            inside the polygon.
        """
        verts = self.vertices
        n = len(verts)
        inside = np.zeros_like(x, dtype=bool)
        for i in range(n):
            x0, y0 = verts[i]
            x1, y1 = verts[(i + 1) % n]
            # Standard ray-casting parity test (ray along +x)
            cond = ((y0 <= y) & (y < y1)) | ((y1 <= y) & (y < y0))
            if not np.any(cond):
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                x_intersect = np.where(
                    cond,
                    x0 + (y - y0) * (x1 - x0) / (y1 - y0),
                    -np.inf,
                )
            inside ^= cond & (x < x_intersect)
        return inside


# ---------------------------------------------------------------------------
# Default polygon environment builder
# ---------------------------------------------------------------------------


def default_polygon_environment(
    vertices: List[List[float]],
    height: float = 0.5,
) -> PolygonEnvironment:
    """Create a polygon environment with real textures and landmarks.

    Wall textures, floor texture, and landmarks are taken from the
    ``ratvision.environments`` package data directory.  When the polygon has
    more than 4 edges, textures and landmarks are cycled with modulo
    indexing.

    Args:
        vertices: 2-D polygon vertices as ``[[x0, y0], [x1, y1], …]`` in
            counter-clockwise order, as produced by
            ``polygon_gen.generate_polygon()``.
        height: Wall height in metres (default 0.5).

    Returns:
        A fully configured :class:`PolygonEnvironment`.
    """
    verts = [(float(v[0]), float(v[1])) for v in vertices]
    n = len(verts)

    env_dir = files("ratvision.environments")

    # -- wall textures (cycled if n > 4) ------------------------------------
    wall_files = [
        "wall_concrete1_635x500.png",  # texture 0
        "wall_texture_635x500.png",    # texture 1
        "wall_concrete2_635x500.png",  # texture 2
        "wall_concrete3_635x500.png",  # texture 3
    ]
    base_textures = [_load_texture(env_dir.joinpath(f)) for f in wall_files]
    wall_textures = [base_textures[i % len(base_textures)] for i in range(n)]

    # -- floor texture ------------------------------------------------------
    floor_texture = _load_texture(env_dir.joinpath("floor_texture_635x635.png"))

    # -- compute edge lengths for landmark sizing ---------------------------
    edge_lengths = []
    for i in range(n):
        p0 = np.array(verts[i])
        p1 = np.array(verts[(i + 1) % n])
        edge_lengths.append(float(np.linalg.norm(p1 - p0)))

    # -- landmarks (cycled if n > 3) ----------------------------------------
    # Define landmark factories; each takes (wall_index, wall_width, wall_height)
    def _landmark_striped_rect(wall_index, wall_w, wall_h):
        return _make_striped_rect_landmark(
            wall_index=wall_index,
            wall_u_extent=wall_w,
            wall_v_extent=wall_h,
            rect_width=min(0.4, wall_w * 0.8),
            rect_height=min(0.3, wall_h * 0.8),
            stripe_width=0.1,
        )

    def _landmark_circle_ring(wall_index, wall_w, wall_h):
        return _make_circle_ring_landmark(
            wall_index=wall_index,
            wall_u_extent=wall_w,
            wall_v_extent=wall_h,
            diameter=min(0.4, wall_w * 0.8, wall_h * 0.8),
            stroke=0.07,
            color=1.0,
        )

    def _landmark_triangle(wall_index, wall_w, wall_h):
        return _make_triangle_landmark(
            wall_index=wall_index,
            wall_u_extent=wall_w,
            wall_v_extent=wall_h,
            tri_height=min(0.4, wall_h * 0.8),
            color=0.0,
            position="left",
        )

    landmark_factories = [
        _landmark_striped_rect,
        _landmark_circle_ring,
        _landmark_triangle,
    ]

    landmarks = []
    for i in range(n):
        factory = landmark_factories[i % len(landmark_factories)]
        landmarks.append(factory(i, edge_lengths[i], height))

    return PolygonEnvironment(
        vertices=verts,
        height=height,
        wall_textures=wall_textures,
        floor_texture=floor_texture,
        ceiling_color=0.0,
        landmarks=landmarks,
    )
