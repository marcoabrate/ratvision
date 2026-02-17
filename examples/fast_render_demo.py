"""
Fast rendering demo using the raycasting renderer.

This script demonstrates how to use RaycastingRenderer for fast,
Blender-free rendering of the default box environment.

Usage:
    python fast_render_demo.py
"""

import json
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from ratvision import RaycastingRenderer


def main():
    # ---- Load example path data ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, 'positions.json')) as f:
        positions = json.load(f)
    with open(os.path.join(script_dir, 'head_directions.json')) as f:
        head_directions = json.load(f)

    print(f'loaded {len(positions)} positions and {len(head_directions)} head directions')

    # ---- Create renderer (default environment) ----
    renderer = RaycastingRenderer(config={'frame_dim': (128, 64)})

    # ---- Benchmark: render entire path ----
    n = len(positions)
    start = time.perf_counter()
    frames = renderer.render_path(positions, head_directions)
    elapsed = time.perf_counter() - start
    print(f'rendered {n} frames in {elapsed:.4f}s  ({n / elapsed:.0f} fps)')
    print(f'output shape: {frames.shape}, dtype: {frames.dtype}')
    print(f'pixel value range:  [{frames.min():.3f}, {frames.max():.3f}]')

    # get the video animation and save it
    try:
        anim = renderer.get_video_animation(frames)
        anim.save("./animation_fast.mp4")
    except Exception as e:
        print(e)
        print("you probably refused to render, or you might have some issues with saving matplotlib animations.")


    # ---- Benchmark: single-frame rendering (training use-case) ----
    # n_iter = 10_000
    # xs = np.random.uniform(0.05, 0.585, n_iter)
    # ys = np.random.uniform(0.05, 0.585, n_iter)
    # thetas = np.random.uniform(-np.pi, np.pi, n_iter)

    # start = time.perf_counter()
    # for i in range(n_iter):
    #     _ = renderer.render_frame(xs[i], ys[i], thetas[i])
    # elapsed = time.perf_counter() - start
    # print(f'\nsingle-frame benchmark: {n_iter} frames in {elapsed:.2f}s  ({n_iter / elapsed:.0f} fps)')

    # ---- Visualise a few sample frames ----
    # fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    # sample_indices = np.linspace(0, n - 1, 8, dtype=int)
    # for ax, idx in zip(axes.flat, sample_indices):
    #     ax.imshow(frames[idx], cmap='gray', vmin=0, vmax=1)
    #     x, y = positions[idx]
    #     theta = head_directions[idx]
    #     ax.set_title(f'#{idx} ({x:.2f},{y:.2f},θ={theta:.2f})', fontsize=8)
    #     ax.axis('off')
    # plt.suptitle('Raycasting Renderer — Sample Frames', fontsize=13)
    # plt.tight_layout()
    # plt.savefig(os.path.join(script_dir, 'fast_render_samples.png'), dpi=150)
    # plt.show()


if __name__ == '__main__':
    main()
