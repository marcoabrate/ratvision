"""Visualisation helpers for ratvision.

Contains utility functions for displaying and exporting rendered frames,
such as `get_video_animation()` which produces a Matplotlib
``FuncAnimation`` from a sequence of frames.
"""

import matplotlib.pyplot as plt
from matplotlib import animation
from typing import List


def get_video_animation(frames: List, fps: int = 10) -> animation.FuncAnimation:
    """
    Opens the rendered video in the default video player.

    Args:
        frames (List): A list of frames to be animated.
        fps (int, optional): Frames per second for the animation. Default is 10.

    Returns:
        animation.FuncAnimation: An animation object that can be used to display the rendered frames.
            The returned object can be saved with the "save" method (i.e. anim.save("filename.mp4")),
            or displayed in a Jupyter notebook with display.display(display.HTML(anim.to_html5_video())),
            where display is imported as "from IPython import display".
    """

    print(f"[+] animating {len(frames)} frames at {fps} fps...")

    # initialize the animation's figure
    fig, ax = plt.subplots(1, 1, figsize=(11, 8))
    im = ax.imshow(frames[0], cmap="gray")
    plt.axis("off")
    plt.close()

    def init():
        im.set_data(frames[0])

    def animate(i):
        im.set_data(frames[i])
        return im

    # animate
    anim = animation.FuncAnimation(
        fig, animate, init_func=init, frames=len(frames), interval=1_000 / fps
    )

    return anim
