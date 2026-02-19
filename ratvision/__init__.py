"""ratvision — simulating rat vision through 3D rendering.

ratvision provides renderers that produce grayscale equirectangular images
of what a rat would see inside a box environment, given its position and
head direction.

Three rendering back-ends are available:

- `BlenderRenderer` — photorealistic rendering via Blender (requires an
  external Blender installation).
- `RaycastingRenderer` — fast CPU-based analytical raycasting (no external
  dependencies beyond NumPy).
- `TorchRenderer` — GPU-accelerated analytical raycasting as a
  `torch.nn.Module`, suitable for end-to-end differentiable pipelines.

The environment is described by a `BoxEnvironment` dataclass that holds
geometry, textures, and `Landmark` definitions.  A ready-made default
environment is available via `default_box_environment()`.

Quick start::

    from ratvision import RaycastingRenderer

    renderer = RaycastingRenderer()                   # default box
    frame    = renderer.render_frame(0.3, 0.3, 0.0)   # (H, W) numpy array

See the `examples/` directory for full demos with each back-end.
"""

from .blender_renderer import BlenderRenderer as BlenderRenderer
from .raycasting_renderer import RaycastingRenderer as RaycastingRenderer
from .box_environment import BoxEnvironment as BoxEnvironment
from .box_environment import Landmark as Landmark
from .box_environment import default_box_environment as default_box_environment
from .helper import get_video_animation as get_video_animation

try:
    from .torch_renderer import TorchRenderer as TorchRenderer
except ImportError:
    pass
