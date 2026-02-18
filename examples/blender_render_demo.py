import json
import argparse

from ratvision import BlenderRenderer, get_video_animation


def main(blender_exec: str):
    # load dummy positions and head_directions of simulated rat
    with open("./positions.json", "r") as f:
        positions = json.load(f)
    with open("./head_directions.json", "r") as f:
        head_directions = json.load(f)

    # print the config description
    BlenderRenderer.config_description()

    # initialize the renderer with the provided Blender command
    renderer = BlenderRenderer(
        blender_exec, config={"output_dir": "./output", "frame_dim": (64, 32)}
    )

    # example of updating the config
    renderer.update_config({"camera_name": "Camera_main"})

    # start rendering the video
    renderer.render(positions, head_directions)

    # get the video animation and save it
    try:
        anim = get_video_animation(frames=renderer.get_rendered_frames())
        anim.save("./animation_blender.mp4")
    except Exception as e:
        print(e)
        print(
            "you probably refused to render, or you might have some issues with saving matplotlib animations."
        )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Render a video using ratvision.")
    argparser.add_argument(
        "--blender_exec",
        type=str,
        required=True,
        help="""
            Path to the Blender executable. This is required to run the rendering process.
            Please be aware his may differ from machine to machine!
            Examples are "/usr/bin/blender" on Linux, or "/Applications/Blender.app/Contents/MacOS/Blender" on MacOS.
        """,
    )
    args = argparser.parse_args()

    main(args.blender_exec)
