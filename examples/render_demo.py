from ratvision import Renderer

import os
import json
import argparse

def main(blender_exec: str):
    # load dummy positions and head_directions of simulated rat
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'positions.json'), 'r') as f:
        positions = json.load(f)
    with open(os.path.join(script_dir, 'head_directions.json'), 'r') as f:
        head_directions = json.load(f)

    # print the config description
    Renderer.config_description()

    # initialize the renderer with the provided Blender command
    r = Renderer(blender_exec, config={'output_dir': './output'})

    # example of updating the config
    r.update_config({'camera_name': 'Camera_main'})

    # start rendering the video
    r.render(positions, head_directions)

    # get the video animation and save it
    try:
        anim = r.get_video_animation()
        anim.save("./animation.mp4")
    except Exception as e:
        print(e)
        print("you probably refused to render, or you might have some issues with saving matplotlib animations.")

if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description='Render a video using ratvision.')
    argparser.add_argument(
        '--blender_exec', type=str, required=True,
        help='''
            Path to the Blender executable. This is required to run the rendering process.
            Please be aware his may differ from machine to machine!
            Examples are "/usr/bin/blender" on Linux, or "/Applications/Blender.app/Contents/MacOS/Blender" on MacOS.
        '''
    )
    args = argparser.parse_args()

    main(args.blender_exec)
