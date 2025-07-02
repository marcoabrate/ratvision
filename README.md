# ratvision

A Python library for simulating rat vision through 3D rendering.

ratvision provides a simple interface to render what a rat would see based on its position and head direction in a 3D environment. It uses Blender for the 3D rendering process, making it possible to create realistic simulations of a rat's visual perspective.

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

### Usage
```python
renderer = Renderer(blender_exec='/path/to/blender') # blender.exe for Win

# positions is List[Tuple[float, float]]
# head_directions is List[float]
renderer.render(positions, head_directions)
```

See `examples/render_demo.py` for a thorough example. After cloning this repository or downloading the `examples` folder, you can run the demo with:
```python
python examples/render_demo.py --blender_exec "/path/to/Blender"
```
Rendered frames and an animation `animation.mp4` will be saved to a new `output` directory.

#### Requirements

- Python 3.7+
- Blender (external dependency, not included in the package)

The code was tested with Blender 3.6 on Linux and MacOS machines.

### Features

- Generate rat-eye-view video animations from movement trajectories
- Easy to use Python API
- Compatible with custom 3D environments
- Built-in visualization function

### Configuration Options

The renderer can be configured with the following parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `env_file` | Path to Blender environment file | Built-in box |
| `output_dir` | Directory where rendered frames are saved | `./output` |
| `frame_dim` | Dimensions of the rendered frames (width, height) | `(120, 64)` |
| `camera_name` | Name of the camera in the Blender scene | `Camera_main` |
| `camera_height` | Height of the camera from the ground in meters | `0.035` |
| `camera_vertical_angle` | Vertical inclination of the camera in radians | `pi/2` |

You can view the configuration description by calling:

```python
Renderer.config_description()
```
### Customizable 3D environment

While ratvision comes with a default 3D environment, you can use your own Blender files:

```python
renderer.update_config({'env_file': '/path/to/environment.blend'})
```

__Note:__ All rendering and camera settings defined in the Blender environment will be preserved when rendering with ratvision. Only the parameters that can be set through the config will be overwritten. Be sure to change them to your preference before running ratvision. __For biologically-plausible rendering and camera settings__, you can check the provided environment `environments/box_messy.blend`. 

### License

This project is licensed under the MIT License - see the LICENSE file for details.

### Author

Marco P. Abrate
[marcopietro.abrate@gmail.com](mailto:marcopietro.abrate@gmail.com)
