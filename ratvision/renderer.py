import tempfile, os, shutil
import subprocess
import json
from importlib.resources import files
from typing import List, Dict, Tuple

class Renderer:
    DEFAULT_CONFIG = {
        "env_file": None, # None means the default box is used
        "render_output": "output",
        "blender_dir": None, # None means the alias "blender" is used
        "frame_dim": (120, 64),
        "camera_name": "Camera_main",
        "camera_height": 0.035, # meters
        "camera_vertical_angle": 90, # degrees
        "view_type": "panoramic",
        "fov_vertical": 120, # degrees
        "fov_horizontal": 240, # degrees
    }

    def __init__(self, config:Dict = None):
        """
        Initializes the Renderer with an optional configuration dictionary.
        """
        self.config = self.DEFAULT_CONFIG.copy()

        # Update the config with user-provided settings
        if config is not None and isinstance(config, dict):
            self.config.update(config)

    def _run_blender_command(
        self,
        blender_cmd: str,
        env_file: str,
        n_frames: int,
        positions_file: str,
        head_directions_file: str,
        config_file: str,
        log_file: str
    ) -> str:
        digits = len(str(n_frames))
        pad = ''.join(['#']*digits)

        blender_script_file = "blender_script.py"
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        blender_script_file = os.path.join(curr_dir, blender_script_file)

        cmd = (
            f"{blender_cmd} --background {env_file}"+
            f" --python {blender_script_file}"+
            f" --frame-start 1 --frame-end {n_frames}"+
            f" --render-output \"{os.path.join(self.config['render_output'], f'frame{pad}.png')}\""+
            f" --render-anim"+
            f" --log-level 0"+
            f" -- \"{positions_file}\" \"{head_directions_file}\" \"{config_file}\""+
            f" > {log_file} 2>&1;"
        )
        print(cmd)

        subprocess.run(
            cmd,
            # stdout=f,
            # stderr=subprocess.STDOUT,
            # text=True,
            shell=True,
            check=False
        )
        print("command executed")

        return None

    def render(self, positions: List[Tuple[float, float]], head_directions: List[float]) -> None:
        """
        Simulates rendering based on positions and head directions.
        """
        if len(positions) != len(head_directions):
            raise ValueError("positions and head_directions must have the same number of elements.")

        n_frames = len(head_directions)
        print(f"[*] rendering data for {n_frames} elements.")

        # Use a temporary directory for all temporary files
        with tempfile.TemporaryDirectory() as tmpdir:

            # Define paths for your temporary files within this directory
            positions_file = os.path.join(tmpdir, "positions.json")
            head_directions_file = os.path.join(tmpdir, "head_directions.json")
            config_file = os.path.join(tmpdir, "config.json")
            # log_file = os.path.join(tmpdir, "log.txt")
            log_file = "log.txt"
            try:
                with open(positions_file, 'w') as f:
                    json.dump(positions, f)
                with open(head_directions_file, 'w') as f:
                    json.dump(head_directions, f)
                with open(config_file, 'w') as f:
                    json.dump(self.config, f)

                if self.config['env_file'] is None:
                    # blender_env_source = files('ratvision.environments').joinpath('box_messy.blend')
                    # blender_env_dest_path = os.path.join(tmpdir, 'box_messy.blend')
                    # with blender_env_source.open('rb') as src, open(blender_env_dest_path, 'wb') as dst:
                    #     shutil.copyfileobj(src, dst)
                    # also need to copy the images
                    x = 0
                    env_file = 'something'
                else:
                    env_file = self.config['env_file']

                blender_cmd = (
                    "blender" if self.config['blender_dir'] is None
                    else self.config['blender_dir']
                )
                self._run_blender_command(
                    blender_cmd,
                    env_file,
                    n_frames,
                    positions_file,
                    head_directions_file,
                    config_file,
                    log_file
                )

            except Exception as e:
                print(f"An error occurred while handling temporary files: {e}")
                raise # Re-raise the exception after printing

        # After the 'with' block, tmpdir and its contents are automatically removed.
    
        # Example of a conceptual internal method to invoke Blender
        # def _invoke_blender_with_files(self, pos_file, head_dir_file):
        #     # This part depends on how you interface with Blender.
        #     # For instance, using subprocess to run Blender with a Python script:
        #     # import subprocess
        #     # blender_script = "path/to/your/blender_render_script.py"
        #     # command = ["blender", "--background", "--python", blender_script,
        #     #            "--", pos_file, head_dir_file]
        #     # subprocess.run(command, check=True)
        #     print(f"Simulating Blender invocation with: {pos_file}, {head_dir_file}")
            # Placeholder for actual rendering logic
