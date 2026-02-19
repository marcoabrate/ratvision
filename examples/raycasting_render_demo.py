import argparse
import json
import os
import time
import numpy as np

from ratvision import RaycastingRenderer, get_video_animation


def main(benchmark):
    # ---- Load example path data ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "positions.json")) as f:
        positions = json.load(f)
    with open(os.path.join(script_dir, "head_directions.json")) as f:
        head_directions = json.load(f)

    print(
        f"loaded {len(positions)} positions and {len(head_directions)} head directions"
    )

    # ---- Create renderer (default environment) ----
    renderer = RaycastingRenderer(config={"frame_dim": (64, 32)})

    # ---- Benchmark: render entire path ----
    n = len(positions)
    start = time.perf_counter()
    frames = renderer.render_path(positions, head_directions)
    elapsed = time.perf_counter() - start
    print(f"rendered {n} frames in {elapsed:.4f}s  ({n / elapsed:.0f} fps)")
    print(f"output shape: {frames.shape}, dtype: {frames.dtype}")
    print(f"pixel value range:  [{frames.min():.3f}, {frames.max():.3f}]")

    # get the video animation and save it
    try:
        anim = get_video_animation(frames)
        anim.save("./animation_raycasting.mp4")
    except Exception as e:
        print(e)
        print(
            "you probably refused to render, or you might have some issues with saving matplotlib animations."
        )

    if not benchmark:
        return

    # ---- Benchmark: single-frame rendering (training use-case) ----
    n_iter = 10_000
    xs = np.random.uniform(0.05, 0.585, n_iter)
    ys = np.random.uniform(0.05, 0.585, n_iter)
    thetas = np.random.uniform(-np.pi, np.pi, n_iter)

    start = time.perf_counter()
    for i in range(n_iter):
        _ = renderer.render_frame(xs[i], ys[i], thetas[i])
    elapsed = time.perf_counter() - start
    print(
        f"\nsingle-frame benchmark: {n_iter} frames in {elapsed:.2f}s  ({n_iter / elapsed:.0f} fps)"
    )


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Render a video using ratvision.")
    argparser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        help="""
            Whether to run a benchmark on 10,000 single-frame renderings after rendering the full path.
        """,
    )
    args = argparser.parse_args()

    main(args.benchmark)
