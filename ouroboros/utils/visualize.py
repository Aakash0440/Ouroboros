"""
Minimal visualization helpers for Phase 1 experiments.

The plotting functions are intentionally lightweight so experiments can run
even when rich plotting dependencies or headless backends are constrained.
If matplotlib is installed (it is in requirements.txt), we generate simple
PNG files; otherwise we fall back to no-op stubs that still return the
expected save paths so callers don't crash.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, Optional

# Try to import matplotlib; degrade gracefully if unavailable
try:
    import matplotlib.pyplot as plt

    _HAVE_MPL = True
except Exception:  # pragma: no cover - best effort fallback
    _HAVE_MPL = False
    plt = None  # type: ignore


def _load_metrics(run_dir: str) -> Dict[int, Dict[str, list]]:
    """
    Load metrics.jsonl from a run directory and group by agent_id.

    Returns:
        dict: agent_id -> {'step': [...], 'compression_ratio': [...]}
    """
    path = Path(run_dir) / "metrics.jsonl"
    data: Dict[int, Dict[str, list]] = {}
    if not path.exists():
        return data

    with path.open() as f:
        for line in f:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("type") == "run_metadata":
                continue
            agent_id = record.get("agent_id")
            if agent_id is None:
                continue
            cr = record.get("compression_ratio")
            step = record.get("step")
            if cr is None or step is None:
                continue
            entry = data.setdefault(agent_id, {"step": [], "compression_ratio": []})
            entry["step"].append(step)
            entry["compression_ratio"].append(cr)
    return data


def _maybe_save(fig, save_path: str) -> str:
    """Save figure if matplotlib is available; otherwise just return path."""
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    if _HAVE_MPL:
        fig.savefig(save_path, bbox_inches="tight")
    return save_path


def plot_compression_curves(
    run_dir: str,
    title: str = "Compression Over Time",
    save_path: str = "compression_curves.png",
) -> str:
    """
    Plot compression_ratio versus step for each agent in a run directory.

    Returns:
        Path to the saved PNG (or intended path if matplotlib unavailable).
    """
    metrics = _load_metrics(run_dir)
    if not _HAVE_MPL or not metrics:
        return _maybe_save(None, save_path)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for agent_id, series in sorted(metrics.items()):
        ax.plot(series["step"], series["compression_ratio"], label=f"Agent {agent_id}")
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Compression Ratio")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.3)
    return _maybe_save(fig, save_path)


def plot_discovery_event(
    run_dir: str,
    agent_id: int,
    discovery_step: int,
    expression_found: str,
    save_path: str = "discovery_event.png",
) -> str:
    """
    Plot a single agent's compression curve with a vertical line marking discovery.
    """
    metrics = _load_metrics(run_dir)
    if not _HAVE_MPL or agent_id not in metrics:
        return _maybe_save(None, save_path)

    series = metrics[agent_id]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(series["step"], series["compression_ratio"], label=f"Agent {agent_id}", color="steelblue")
    ax.axvline(discovery_step, color="gold", linestyle="--", label="Discovery")
    ax.set_title("Discovery Event")
    ax.set_xlabel("Step")
    ax.set_ylabel("Compression Ratio")
    ax.legend(loc="upper right", fontsize="small")
    ax.text(discovery_step, min(series["compression_ratio"]), expression_found, rotation=90, va="bottom")
    ax.grid(True, alpha=0.3)
    return _maybe_save(fig, save_path)


def plot_environment_comparison(
    run_dirs: Iterable[str],
    labels: Optional[Iterable[str]] = None,
    save_path: str = "environment_comparison.png",
) -> str:
    """
    Placeholder for comparing multiple runs. Currently plots mean curve per run if
    matplotlib is available; otherwise returns the intended path.
    """
    if not _HAVE_MPL:
        return _maybe_save(None, save_path)

    labels = list(labels) if labels is not None else [f"Run {i}" for i, _ in enumerate(run_dirs)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for run_dir, label in zip(run_dirs, labels):
        metrics = _load_metrics(run_dir)
        if not metrics:
            continue
        # Compute mean compression per step across agents
        steps = {}
        for series in metrics.values():
            for s, r in zip(series["step"], series["compression_ratio"]):
                steps.setdefault(s, []).append(r)
        xs = sorted(steps)
        ys = [sum(steps[s]) / len(steps[s]) for s in xs]
        ax.plot(xs, ys, label=label)
    ax.set_title("Environment Comparison")
    ax.set_xlabel("Step")
    ax.set_ylabel("Mean Compression Ratio")
    ax.legend(loc="upper right", fontsize="small")
    ax.grid(True, alpha=0.3)
    return _maybe_save(fig, save_path)


__all__ = [
    "plot_compression_curves",
    "plot_discovery_event",
    "plot_environment_comparison",
]
