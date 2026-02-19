# ratvision

A Python library for simulating rat vision through 3D rendering.

ratvision provides a simple interface to render what a rat would see based on its position and head direction in a 3D environment. Three rendering back-ends are available: a **Blender-based** photorealistic renderer, a fast **CPU raycaster**, and a **GPU-accelerated PyTorch** renderer suitable for end-to-end differentiable pipelines.

### Documentation

Full API documentation is available at **[marcoabrate.github.io/ratvision](https://marcoabrate.github.io/ratvision/)**.

### Installation

```bash
pip install ratvision
```
Or by cloning this repository:
```bash
git clone git@github.com:marcoabrate/ratvision.git
cd ratvision
pip install .
```
For GPU-accelerated rendering with `TorchRenderer` (*Note:* it installs torch):
```bash
pip install ratvision[gpu]
```

### Quick start

**Raycasting renderer** (no external dependencies):
```python
from ratvision import RaycastingRenderer

renderer = RaycastingRenderer()                    # default box environment
frame    = renderer.render_frame(0.3, 0.3, 0.0)   # (H, W) numpy array
frames   = renderer.render_path(positions, head_directions)
```

**PyTorch renderer** (GPU-accelerated):
```python
from ratvision import TorchRenderer
import torch

renderer = TorchRenderer(config={'frame_dim': (64, 32)}).to('cuda')
frames   = renderer(positions_tensor, head_directions_tensor)  # (B, H, W) tensor
```

**Blender renderer** (photorealistic):
```python
from ratvision import BlenderRenderer

renderer = BlenderRenderer(blender_exec='/path/to/blender')
renderer.render(positions, head_directions)
```

See the `examples/` directory for full runnable demos with each back-end:
```bash
python examples/raycasting_render_demo.py
python examples/torch_render_demo.py
python examples/blender_render_demo.py --blender_exec "/path/to/Blender"
```

### Requirements

- Python 3.9+
- [Blender](https://www.blender.org/) (only required for `BlenderRenderer`, not included in the package)

The Blender renderer was tested with Blender 3.6 on Linux and macOS.

### Features

- Three rendering back-ends (Blender, CPU raycasting, GPU PyTorch)
- Generate rat-eye-view video animations from movement trajectories
- Easy to use Python API
- Compatible with custom 3D environments and procedural landmarks
- Built-in visualisation utilities (`get_video_animation`)
- GPU batch rendering for training loops (`TorchRenderer`)

### Configuration options

Each renderer can be configured with parameters such as:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `frame_dim` | Dimensions of the rendered frames (width, height) | `(128, 64)` |
| `camera_height` | Height of the camera from the ground in metres | `0.035` |
| `hfov` | Horizontal field of view in radians | `4π/3 (240°)` |
| `vfov` | Vertical field of view in radians | `2π/3 (120°)` |
| `output_dir` | Directory where rendered frames are saved | `./output` |

Additional Blender-specific options:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `env_file` | Path to Blender environment file | Built-in box |
| `camera_name` | Name of the camera in the Blender scene | `Camera_main` |
| `camera_vertical_angle` | Vertical inclination of the camera in radians | `π/2` |

You can view and update the configuration at runtime:
```python
renderer.config_description()          # print all available keys
renderer.update_config({'frame_dim': (64, 32)})
```

### Customisable 3D environment

While ratvision comes with a default 3D environment, you can create custom environments programmatically or use your own Blender files:

```python
from ratvision import BoxEnvironment, Landmark, RaycastingRenderer

env = BoxEnvironment(width=1.0, depth=1.0, height=0.8, wall_color=0.6)
renderer = RaycastingRenderer(env=env)
```

For Blender-based rendering with a custom `.blend` file:
```python
renderer.update_config({'env_file': '/path/to/environment.blend'})
```

> **Note:** All rendering and camera settings defined in the Blender file will be preserved. Only the parameters set through the config will be overwritten. For biologically-plausible rendering and camera settings, check the provided environment `environments/box_messy.blend`.

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Author

Marco P Abrate
[marcopietro.abrate@gmail.com](mailto:marcopietro.abrate@gmail.com)
