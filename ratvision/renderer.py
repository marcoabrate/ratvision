import json
import math
import os
from pathlib import Path
import platform
import subprocess
import tempfile
from typing import List, Dict, Tuple

import sys
if sys.version_info < (3, 9):
    import importlib_resources
    files = importlib_resources.files
else:
    from importlib.resources import files

import matplotlib.pyplot as plt
from matplotlib import animation

class Renderer:
    DEFAULT_CONFIG = {
        'env_file': None,
        'output_dir': None,
        'frame_dim': (120, 64),
        'camera_name': 'Camera_main',
        'camera_height': 0.035,
        'camera_vertical_angle': math.pi/2,
    }
    CONFIG_DESCRIPTION = {
        'env_file': 'String or Path to the Blender environment file (e.g., /your/path/box.blend). If None, the default environment is used.',
        'output_dir': 'Path where rendered images will be saved. If None, output will be saved to "./output".',
        'frame_dim': 'Dimensions of the rendered frames (width, height), in pixels.',
        'camera_name': 'Name of the camera in the Blender scene. Default is "Camera_main".',
        'camera_height': 'Height of the camera from the ground in meters. Default is 0.035 meters.',
        'camera_vertical_angle': 'Vertical inclination of the camera in radians. Default is pi/2, parallel to the ground.',
    }

    def __init__(self, blender_exec: str, config: Dict = None):
        '''
        This Renderer class is the main interface for rendering rat's vision using Blender.
        It allows you to configure the rendering settings, render the rat's vision based on positions and head directions,
        and retrieve the rendered video animation.

        Args:
            blender_exec (str): String containing path to the Blender executable. This is required to run the rendering process.
                Please be aware his may differ from machine to machine!
                Examples are "/usr/bin/blender" on Linux, or "/Applications/Blender.app/Contents/MacOS/Blender" on MacOS or
                "C:/Program Files/Blender Foundation/Blender/blender.exe" on Windows.
            config (Dict, optional): Configuration dictionary to override default settings. If None, default settings are used.
        '''

        # check if the path to the Blender executable was given by the user
        if blender_exec is None:
            raise ValueError(
                '"blender_exec" is not set, please provide the path to the Blender '+
                'executable in the config before calling the "render" function.'
            )
        self.blender_exec = Path(blender_exec)

        self.config = self.DEFAULT_CONFIG.copy()

        # update the config with user-provided configuration
        if config is not None:
            self.update_config(config)
        else:
            print('[*] no configuration provided, using default.')
            self._print_config_message()

        if self.config['output_dir'] is None:
            self.config['output_dir'] = os.path.join(os.getcwd(), 'output')

    @staticmethod
    def _print_config_message() -> None:
        '''
        Prints a message about the configuration and how to update it.
        '''
        print()
        print('you can check the configuration description by calling')
        print('the "config_description" function or by checking the documentation,')
        print('and update the configuration by calling the "update_config" function.')
        print()

    @staticmethod
    def config_description() -> None:
        '''
        Prints a description of each configuration key and its purpose.
        '''
        print('[*] configuration description:')
        for key, value in Renderer.CONFIG_DESCRIPTION.items():
            print(f'\t{key}: {value}')
        Renderer._print_config_message()

    def print_config(self) -> None:
        '''
        Prints the current configuration of the Renderer.
        '''
        print('[*] current configuration:')
        for key, value in self.config.items():
            print(f'\t{key}: {value}')
        self._print_config_message()

    def update_config(self, config: Dict) -> None:
        '''
        Updates the Renderer configuration with a new configuration dictionary.

        Args:
            config (Dict): Dictionary containing new configuration values. Only the provided keys will be updated.
        '''
        if not isinstance(config, dict):
            raise ValueError('config must be a dictionary.')
        
        for key, value in config.items():
            if key in self.config:
                self.config[key] = value
            else:
                print(f'[-] {key} is not a valid configuration key, skipping.')

    def _continue(self) -> bool:
        '''
        Asks the user if they wish to continue with the rendering process.
        '''
        while True:
            user_input = input(
                'do you wish to continue? (y/n): '
            )

            if user_input.strip().lower() == 'y':
                print()
                return True
            elif user_input.strip().lower() == 'n':
                print()
                return False

    def _run_blender_command(
        self,
        n_frames: int,
        positions_file: str,
        head_directions_file: str,
        config_file: str,
    ) -> str:
        # prepare the frame naming format, where "#"
        # is a placeholder for the frame number
        digits = len(str(n_frames))
        pad = ''.join(['#']*digits)

        # prepare the Blender script file path
        blender_script_file = 'blender_script.py'
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        blender_script_file = os.path.join(curr_dir, blender_script_file)

        # prepare the command to run Blender
        cmd = [
            str(self.blender_exec),
            '--background', self.config['env_file'],
            '--python', blender_script_file,
            '--frame-start', '1',
            '--frame-end', str(n_frames),
            '--render-output', os.path.join(self.config['output_dir'], f'frame{pad}.png'),
            '--render-anim',
            '--log-level', '0',
            '--', positions_file, head_directions_file, config_file,
        ]
        if platform.system() == 'Windows':
            cmd = ['cmd', '/C'] + cmd
        print(f'[+] ready to render rat\'s vision for {n_frames} samples')
        print()
        print('this might take several minutes, depending on your machine')

        if self._continue():
            print('[+] rendering...')
            subprocess.run(
                cmd,
                check=True
            )
        else:
            print('[-] rendering aborted, exiting.')
        
        return None

    def render(self, positions: List[Tuple[float, float]], head_directions: List[float]) -> None:
        '''
        Render rat's vision based on positions and head directions provided by the user.

        Args:
            positions (List[Tuple[float, float]]): List of tuples representing the x, y coordinates of the rat's position.
            head_directions (List[float]): List of head directions in radians.
        '''
        if not isinstance(positions, list) or not isinstance(head_directions, list):
            raise TypeError('positions and head_directions must be lists.')
        if len(positions) != len(head_directions):
            raise ValueError('positions and head_directions must have the same number of elements.')

        if any(hd < -2*math.pi or hd > 2*math.pi for hd in head_directions):
            print('[!!!] remember that head_directions should be in radians.')

        n_frames = len(head_directions)

        # use a temporary directory to pass
        # positions, head_directions and configurations to Blender
        with tempfile.TemporaryDirectory() as tmpdir:

            # define paths for temporary files
            positions_file = os.path.join(tmpdir, 'positions.json')
            head_directions_file = os.path.join(tmpdir, 'head_directions.json')
            config_file = os.path.join(tmpdir, 'config.json')

            try:
                # dump temporary files to temporary directory
                with open(positions_file, 'w') as f:
                    json.dump(positions, f)
                with open(head_directions_file, 'w') as f:
                    json.dump(head_directions, f)
                with open(config_file, 'w') as f:
                    json.dump(self.config, f)

                # check whether to use the default demo environment
                if self.config['env_file'] is None:
                    print('[*] no environment file provided, using default environment.\n')
                    blender_env_dir = files('ratvision.environments')
                    self.config['env_file'] = blender_env_dir.joinpath('box_messy.blend')
                elif isinstance(self.config['env_file'], Path):
                    self.config['env_file'] = str(self.config['env_file'])

                # print configuration to the user
                self.print_config()

                # render
                self._run_blender_command(
                    n_frames,
                    positions_file,
                    head_directions_file,
                    config_file,
                )

            except Exception as e:
                print(f'an error occurred while handling temporary files: {e}')
                raise

        return None

    def get_video_animation(self, fps: int = 10) -> animation.FuncAnimation:
        '''
        Opens the rendered video in the default video player.

        Args:
            fps (int, optional): Frames per second for the animation. Default is 10.

        Returns:
            animation.FuncAnimation: An animation object that can be used to display the rendered frames.
                The returned object can be saved with the "save" method (i.e. anim.save("filename.mp4")),
                or displayed in a Jupyter notebook with display.display(display.HTML(anim.to_html5_video())),
                where display is imported as "from IPython import display".
        '''
        from PIL import Image

        output_dir = self.config['output_dir']
        if not os.path.exists(output_dir):
            print(f'[-] output directory "{output_dir}" does not exist.')
            print('you first need to render the rat\'s vision, or specify the correct "output_dir" in the config.')
            return None
        
        # define the list of paths to the rendered frames
        frame_files = sorted([
            os.path.join(output_dir, f) for f in os.listdir(output_dir)
            if os.path.isfile(os.path.join(output_dir, f))
        ])

        if len(frame_files) == 0:
            print(f'[-] no frames found in the output directory "{output_dir}".')
            print('you first need to render the rat\'s vision, or specify the correct "output_dir" in the config.')
            return None

        # load the frames
        frames = [Image.open(ff) for ff in frame_files]

        print(f'[+] animating {len(frames)} frames at {fps} fps...')

        # initialize the animation's figure
        fig, ax = plt.subplots(1, 1, figsize=(11, 8))
        im = ax.imshow(frames[0])
        plt.axis('off')
        plt.close()

        def init():
            im.set_data(frames[0])

        def animate(i):
            im.set_data(frames[i])
            return im

        # animate
        anim = animation.FuncAnimation(
            fig,
            animate,
            init_func=init,
            frames=len(frames),
            interval=1_000/fps
        )

        return anim
