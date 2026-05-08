"""
Downloads and discretizes time series from DATASETS registry.
Returns List[int] ready for OUROBOROS.
"""
from __future__ import annotations
import math, requests
from typing import List, Optional, Dict, Any


def fetch(dataset: Dict[str, Any], max_points: int = 200, timeout: int = 15) -> Optional[List[int]]:
    parser = dataset.get("parser", "csv")
    try:
        r = requests.get(dataset["url"], timeout=timeout)
        r.raise_for_status()
    except Exception as e:
        print(f"  [fetch] {dataset['name']}: {e}")
        return None

    try:
        if parser == "usgs":
            floats = _parse_usgs_raw(r.text)
        else:
            floats = _parse_csv_raw(r.text, dataset["col"])

        if floats is None or len(floats) < 10:
            return None

        floats = floats[:max_points]
        return _discretize(floats)

    except Exception as e:
        print(f"  [parse] {dataset['name']}: {e}")
        return None


def _parse_csv_raw(text: str, col: int) -> Optional[List[float]]:
    floats = []
    for line in text.strip().splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.split(",")
        if len(parts) <= col:
            continue
        try:
            floats.append(float(parts[col].strip().strip('"')))
        except ValueError:
            continue
    return floats if floats else None


def _parse_usgs_raw(text: str) -> Optional[List[float]]:
    floats = []
    for line in text.splitlines():
        if line.startswith("#") or line.startswith("agency") or line.startswith("5s"):
            continue
        parts = line.split("\t")
        if len(parts) < 5:
            continue
        try:
            floats.append(float(parts[4]))
        except ValueError:
            continue
    return floats if floats else None


def _detrend(floats: List[float]) -> List[float]:
    """Remove linear trend so OUROBOROS sees residuals, not drift."""
    n = len(floats)
    if n < 4:
        return floats
    mx = (n - 1) / 2.0
    my = sum(floats) / n
    denom = sum((i - mx) ** 2 for i in range(n))
    if denom == 0:
        return floats
    slope = sum((i - mx) * (floats[i] - my) for i in range(n)) / denom
    intercept = my - slope * mx
    return [floats[i] - (slope * i + intercept) for i in range(n)]


def _discretize(floats: List[float], n_bins: int = 50) -> List[int]:
    """Map floats → integers in [0, n_bins] preserving relative ordering."""
    lo, hi = min(floats), max(floats)
    if hi == lo:
        return [0] * len(floats)
    return [int(round((v - lo) / (hi - lo) * n_bins)) for v in floats]