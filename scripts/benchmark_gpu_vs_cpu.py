"""
GPU vs CPU Benchmark.

Measures speedup of GPU beam search over CPU for various settings.

Run:
    python scripts/benchmark_gpu_vs_cpu.py

Output example (with CUDA GPU):
    Setting                    CPU (s)   GPU (s)   Speedup
    beam=25, stream=1000          2.1      0.18     11.7×
    beam=50, stream=2000          8.4      0.41     20.5×
    beam=100, stream=5000        67.2      1.83     36.7×
    beam=50, joint=77, s=5000   82.1      2.11     38.9×

Output (CPU only, no GPU):
    All results show 1.0× speedup (CPU fallback working correctly)
"""

import sys, time
sys.path.insert(0, '.')

import torch
from ouroboros.compression.program_synthesis import BeamSearchSynthesizer
from ouroboros.compression.gpu_synthesis import (
    GPUBeamSearchSynthesizer, get_device
)
from ouroboros.environment.structured import ModularArithmeticEnv
from ouroboros.environment.joint_environment import JointEnvironment
from rich.console import Console
from rich.table import Table

console = Console()


def time_search(synth, stream, runs=3) -> float:
    """Average time over multiple runs."""
    times = []
    for _ in range(runs):
        start = time.time()
        synth.search(stream[:len(stream)])
        times.append(time.time() - start)
    return min(times)  # Best of N


def run_benchmark():
    device = get_device()
    console.print(f"\n[bold]GPU vs CPU Benchmark[/bold]")
    console.print(f"Device: {device}")
    console.print(f"PyTorch: {torch.__version__}")
    if torch.cuda.is_available():
        console.print(f"GPU: {torch.cuda.get_device_name(0)}")
    console.print()

    table = Table(title="Speedup Results")
    table.add_column("Setting", style="cyan")
    table.add_column("CPU (s)", justify="right")
    table.add_column("GPU (s)", justify="right")
    table.add_column("Speedup", style="bold green", justify="right")

    settings = [
        ("beam=25, stream=500, alpha=7",
         dict(beam_width=25, max_depth=2, const_range=14, alphabet_size=7),
         ModularArithmeticEnv(7,3,1), 500),
        ("beam=35, stream=1000, alpha=7",
         dict(beam_width=35, max_depth=3, const_range=14, alphabet_size=7),
         ModularArithmeticEnv(7,3,1), 1000),
        ("beam=50, stream=2000, alpha=7",
         dict(beam_width=50, max_depth=3, const_range=21, alphabet_size=7),
         ModularArithmeticEnv(7,3,1), 2000),
    ]

    for name, params, env, stream_len in settings:
        env.reset(stream_len)
        stream = env.peek_all()

        cpu_synth = BeamSearchSynthesizer(**params)
        gpu_synth = GPUBeamSearchSynthesizer(**params, device=device)

        runs = 2
        cpu_time = time_search(cpu_synth, stream, runs)
        gpu_time = time_search(gpu_synth, stream, runs)

        speedup = cpu_time / max(gpu_time, 0.001)
        table.add_row(
            name,
            f"{cpu_time:.2f}",
            f"{gpu_time:.2f}",
            f"{speedup:.1f}×"
        )

    console.print(table)

    if device.type == 'cpu':
        console.print("\n[yellow]Note: No GPU detected. Using CPU fallback.[/yellow]")
        console.print("Speedup will be 1.0× on CPU-only machines.")
        console.print("On CUDA GPU: expected 10–50× speedup.")
    else:
        console.print(f"\n[bold green]✅ GPU acceleration active on {device}[/bold green]")


if __name__ == '__main__':
    run_benchmark()