import argparse
import json
import os
import time
import math
import torch

from ratvision import TorchRenderer, get_video_animation


def _sync_kernels(device):
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()
    else:
        pass  # No synchronization needed for CPU


def main(benchmark):
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"[*] using device: {device}\n")

    config = {"frame_dim": (64, 32)}

    # ---- Load example path data ----
    script_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(script_dir, "positions.json")) as f:
        positions = json.load(f)
    with open(os.path.join(script_dir, "head_directions.json")) as f:
        head_directions = json.load(f)

    n = len(positions)
    print(f"loaded {n} positions and {n} head directions\n")

    # ==================================================================
    # 1. PyTorch (TorchRenderer) — path rendering
    # ==================================================================
    torch_renderer = TorchRenderer(config=config).to(device)

    # Convert to tensors on device
    pos_t = torch.tensor(positions, dtype=torch.float32, device=device)
    hd_t = torch.tensor(head_directions, dtype=torch.float32, device=device)

    # Warm-up (JIT compilation, CUDA kernel loading)
    _ = torch_renderer(pos_t[:1], hd_t[:1])
    _sync_kernels(device)

    start = time.perf_counter()
    torch_frames = torch_renderer(pos_t, hd_t)
    _sync_kernels(device)
    torch_elapsed = time.perf_counter() - start

    print(f"[Torch]  {n} frames in {torch_elapsed:.4f}s  ({n / torch_elapsed:.0f} fps)")
    print(f"         shape={tuple(torch_frames.shape)}, dtype={torch_frames.dtype}")
    print(f"         device={torch_frames.device}")

    # get the video animation and save it
    try:
        anim = get_video_animation(torch_frames.cpu())
        anim.save("./animation_torch.mp4")
    except Exception as e:
        print(e)
        print(
            "you probably refused to render, or you might have some issues with saving matplotlib animations."
        )
    print()

    if not benchmark:
        return

    from ratvision import RaycastingRenderer

    # ==================================================================
    # 2. Numpy vs torch benchmark
    # ==================================================================
    n_iter = int(math.pow(2, 16))
    batch_size = int(math.pow(2, 13))

    xs = torch.rand(n_iter, device=device) * 0.585 + 0.025
    ys = torch.rand(n_iter, device=device) * 0.585 + 0.025
    thetas = torch.rand(n_iter, device=device) * 2 * math.pi - math.pi
    positions = torch.stack([xs, ys], dim=-1)  # (n_iter, 2)

    # Numpy renderer (CPU)
    np_renderer = RaycastingRenderer(config=config)
    start = time.perf_counter()
    np_frames = np_renderer.render_path(positions.tolist(), thetas.tolist())
    np_elapsed = time.perf_counter() - start
    print(
        f"[NumPy]  {n_iter} frames in {np_elapsed:.4f}s  ({n_iter / np_elapsed:.0f} fps)"
    )
    print(f"         shape={tuple(np_frames.shape)}, dtype={np_frames.dtype}")

    # Torch renderer (GPU, Metal, or CPU)
    # Warm-up
    _ = torch_renderer(positions[:1], thetas[:1])
    _sync_kernels(device)

    start = time.perf_counter()
    for i in range(0, n_iter, batch_size):
        torch_frames = torch_renderer(
            positions[i : i + batch_size], thetas[i : i + batch_size]
        )
        size = torch_frames.numel() * torch_frames.element_size() * 1e-6  # MB
    _sync_kernels(device)
    torch_elapsed = time.perf_counter() - start
    print(
        f"[Torch]  {n_iter} frames with batch {batch_size} in {torch_elapsed:.4f}s  ({n_iter / torch_elapsed:.0f} fps)"
    )
    print(
        f"         shape={tuple(torch_frames.shape)}, dtype={torch_frames.dtype}, memory={size:.1f} MB"
    )
    print(f"         device={device}")
    speedup = np_elapsed / torch_elapsed
    print(f"         speedup vs NumPy: {speedup:.1f}x\n")

    print("done.")


if __name__ == "__main__":
    argparser = argparse.ArgumentParser(description="Render a video using ratvision.")
    argparser.add_argument(
        "--benchmark",
        action=argparse.BooleanOptionalAction,
        help="""
            Whether to run the benchmark comparing TorchRenderer to RaycastingRenderer.
            This will render a large number of frames and print the timing results.
        """,
    )
    args = argparser.parse_args()

    main(args.benchmark)
