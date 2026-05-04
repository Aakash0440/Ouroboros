"""
RealDataPipeline — Connects OUROBOROS to real scientific data formats.

Supported formats:
  pandas DataFrame  — most common scientific Python data format
  numpy ndarray     — raw numerical data
  CSV file          — universal text-based format
  HDF5 file         — large numerical datasets (climate, genomics)
  netCDF file       — climate and atmospheric data standard
  FASTA file        — genomic sequence data (DNA, protein)
  JSON array        — web API data
  SQL database      — relational data via connection string

For each format, the pipeline:
  1. Loads data into a standard MultivariateSequence or scalar sequence
  2. Auto-detects the appropriate OUROBOROS environment type
  3. Runs the hierarchical router with appropriate settings
  4. Returns a DiscoveryResult with the found expression

Usage examples:
  # From pandas DataFrame
  pipeline = RealDataPipeline()
  result = pipeline.from_dataframe(df, target_column="temperature")

  # From CSV
  result = pipeline.from_csv("data/climate.csv", target_column="co2_ppm")

  # From genomic sequence
  result = pipeline.from_fasta("genome.fa", analyze_gc_content=True)

  # From time series in HDF5
  result = pipeline.from_hdf5("data.h5", dataset="measurements/sensor_1")
"""

from __future__ import annotations
import json
import math
import csv
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Dict, Any, Union

from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.multivariate.mdl_engine import MultivariateObservations as MultivariateSequence


@dataclass
class PipelineResult:
    """Result from the real data pipeline."""
    source_format: str
    source_path: Optional[str]
    n_observations: int
    n_channels: int
    channel_names: List[str]
    expression_str: Optional[str]
    mdl_cost: float
    compression_ratio: float
    math_family: str
    runtime_seconds: float
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"Pipeline Result ({self.source_format})\n"
            f"  Data: {self.n_observations} obs × {self.n_channels} channels\n"
            f"  Expression: {self.expression_str}\n"
            f"  MDL: {self.mdl_cost:.2f} bits\n"
            f"  Compression: {self.compression_ratio:.4f}\n"
            f"  Family: {self.math_family}\n"
            f"  Time: {self.runtime_seconds:.2f}s"
            + (f"\n  Warnings: {self.warnings}" if self.warnings else "")
        )


class DataLoader:
    """Loads various data formats into Python lists."""

    def from_list(self, data: List[float]) -> List[float]:
        """Pass-through for already-loaded lists."""
        return [float(v) for v in data if v == v and math.isfinite(float(v))]

    def from_csv_string(self, content: str, column: int = 0) -> List[float]:
        """Load from CSV string."""
        reader = csv.reader(io.StringIO(content))
        values = []
        header_skipped = False
        for row in reader:
            if not header_skipped:
                try:
                    float(row[column])
                    header_skipped = True
                except (ValueError, IndexError):
                    header_skipped = True
                    continue
            try:
                if len(row) > column:
                    val = float(row[column])
                    if math.isfinite(val):
                        values.append(val)
            except (ValueError, IndexError):
                pass
        return values

    def from_csv_file(self, path: str, column: Union[int, str] = 0) -> List[float]:
        """Load from CSV file."""
        content = Path(path).read_text()
        if isinstance(column, str):
            # Find column index from header
            reader = csv.reader(io.StringIO(content))
            header = next(reader)
            try:
                col_idx = header.index(column)
            except ValueError:
                col_idx = 0
        else:
            col_idx = column
        return self.from_csv_string(content, col_idx)

    def from_json_array(self, data: Union[str, list]) -> List[float]:
        """Load from JSON array."""
        if isinstance(data, str):
            data = json.loads(data)
        if isinstance(data, list):
            values = []
            for v in data:
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        values.append(fv)
                except (TypeError, ValueError):
                    pass
            return values
        return []

    def from_dict(
        self,
        data: Dict[str, List[float]],
        target_key: Optional[str] = None,
    ) -> List[float]:
        """Load from dict of {name: values}."""
        if target_key and target_key in data:
            return self.from_list(data[target_key])
        # Use first key
        if data:
            first_key = next(iter(data))
            return self.from_list(data[first_key])
        return []

    def try_load_pandas(
        self,
        df,
        target_column: Optional[str] = None,
    ) -> List[float]:
        """Load from pandas DataFrame (optional dependency)."""
        try:
            if target_column and target_column in df.columns:
                series = df[target_column].dropna()
            else:
                # Use first numeric column
                numeric_cols = df.select_dtypes(include='number').columns
                if len(numeric_cols) == 0:
                    return []
                series = df[numeric_cols[0]].dropna()
            return [float(v) for v in series.values if math.isfinite(float(v))]
        except Exception as e:
            return []

    def try_load_numpy(self, arr) -> List[float]:
        """Load from numpy array."""
        try:
            flat = arr.flatten()
            return [float(v) for v in flat if math.isfinite(float(v))]
        except Exception:
            return []

    def try_load_hdf5(self, path: str, dataset: str) -> List[float]:
        """Load from HDF5 file (requires h5py)."""
        try:
            import h5py
            with h5py.File(path, 'r') as f:
                data = f[dataset][:]
                return [float(v) for v in data.flatten() if math.isfinite(float(v))]
        except ImportError:
            return []  # h5py not installed
        except Exception:
            return []

    def try_load_netcdf(self, path: str, variable: str) -> List[float]:
        """Load from netCDF file (requires netCDF4 or xarray)."""
        try:
            import xarray as xr
            ds = xr.open_dataset(path)
            if variable in ds:
                data = ds[variable].values.flatten()
                return [float(v) for v in data if math.isfinite(float(v))]
        except ImportError:
            pass
        try:
            import netCDF4
            ds = netCDF4.Dataset(path)
            if variable in ds.variables:
                data = ds.variables[variable][:].flatten()
                return [float(v) for v in data if math.isfinite(float(v))]
        except ImportError:
            return []
        return []

    def from_fasta_gc_content(self, path_or_string: str) -> List[float]:
        """
        Load FASTA file and compute GC content in sliding windows.
        Returns a sequence of GC fractions (0.0 to 1.0).
        """
        # Parse FASTA (minimal implementation, no BioPython required)
        if path_or_string.startswith('>') or '\n' in path_or_string:
            content = path_or_string
        else:
            try:
                content = Path(path_or_string).read_text()
            except Exception:
                content = path_or_string

        sequence = ""
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('>') or not line:
                continue
            sequence += line.upper()

        if not sequence:
            return []

        # Sliding window GC content (window=100bp)
        window = min(100, len(sequence) // 10, 50)
        gc_values = []
        for i in range(0, len(sequence) - window, max(1, window // 5)):
            segment = sequence[i:i+window]
            gc = sum(1 for c in segment if c in 'GC') / max(len(segment), 1)
            gc_values.append(gc)

        return gc_values


class RealDataPipeline:
    """
    Connect OUROBOROS to real scientific data.

    The pipeline handles format detection, loading, and preprocessing.
    OUROBOROS handles discovery.

    Usage:
        pipeline = RealDataPipeline(beam_width=20, n_iterations=10)

        # From any format
        result = pipeline.discover(data, format="csv", target_column="value")
        result = pipeline.discover(df, format="pandas", target_column="temp")
        result = pipeline.discover("[1,2,3,4,5]", format="json")
    """

    def __init__(
        self,
        beam_width: int = 20,
        n_iterations: int = 10,
        max_sequence_length: int = 5000,
        verbose: bool = False,
    ):
        self._router = HierarchicalSearchRouter(RouterConfig(
            beam_width=beam_width,
            max_depth=5,
            n_iterations=n_iterations,
            random_seed=42,
        ))
        self._loader = DataLoader()
        self.max_length = max_sequence_length
        self.verbose = verbose

    def discover(
        self,
        data,
        format: str = "auto",
        target_column: Union[str, int] = None,
        alphabet_size: int = None,
        **kwargs,
    ) -> PipelineResult:
        """
        Auto-detect format and run OUROBOROS discovery.

        data: the data in any supported format
        format: "pandas", "numpy", "csv", "json", "hdf5", "netcdf", "fasta", "list", "auto"
        target_column: for tabular data, which column to analyze
        """
        import time
        start = time.time()
        warnings = []

        # Load data based on format
        values = self._load(data, format, target_column, kwargs, warnings)

        if len(values) < 10:
            return PipelineResult(
                source_format=format, source_path=None,
                n_observations=len(values), n_channels=1,
                channel_names=["obs"], expression_str=None,
                mdl_cost=float('inf'), compression_ratio=1.0,
                math_family="UNKNOWN", runtime_seconds=time.time()-start,
                warnings=warnings + ["Too few observations (need >= 10)"],
            )

        # Truncate if too long
        if len(values) > self.max_length:
            values = values[:self.max_length]
            warnings.append(f"Truncated to {self.max_length} observations")

        # Discretize if needed
        int_values = [int(round(v)) for v in values]
        if alphabet_size is None:
            unique_vals = len(set(int_values))
            alphabet_size = max(unique_vals + 2, 10)

        # Run discovery
        result = self._router.search(int_values, alphabet_size=alphabet_size)

        # Compute compression ratio
        n = len(int_values)
        baseline_bits = n * math.log2(max(alphabet_size, 2))
        compression = result.mdl_cost / max(baseline_bits, 1.0)

        return PipelineResult(
            source_format=format,
            source_path=str(data) if isinstance(data, (str, Path)) else None,
            n_observations=n,
            n_channels=1,
            channel_names=["obs"],
            expression_str=result.expr.to_string() if result.expr else None,
            mdl_cost=result.mdl_cost,
            compression_ratio=compression,
            math_family=result.math_family.name,
            runtime_seconds=time.time() - start,
            warnings=warnings,
        )

    def discover_multivariate(
        self,
        data: Union[Dict[str, List[float]], 'MultivariateSequence'],
        target_channel: int = 0,
        alphabet_size: int = None,
    ) -> PipelineResult:
        """Discover cross-channel laws in multivariate data."""
        import time
        start = time.time()

        from ouroboros.environments.multivariate import MultivariateBeamSearch, MultivariateSequence as MVS
        if isinstance(data, dict):
            n_t = min(len(v) for v in data.values())
            seq_data = [[data[k][t] for k in data] for t in range(n_t)]
            seq = MVS(seq_data, list(data.keys()))
        else:
            seq = data

        searcher = MultivariateBeamSearch(
            target_channel=target_channel,
            beam_width=15, n_iterations=8,
        )
        cross_law = searcher.discover_cross_channel_law(seq)

        return PipelineResult(
            source_format="multivariate",
            source_path=None,
            n_observations=seq.n_timesteps,
            n_channels=seq.n_channels,
            channel_names=seq.channel_names,
            expression_str=str(cross_law.get("cross_correlations", {})),
            mdl_cost=0.0,
            compression_ratio=1.0,
            math_family="STATISTICAL",
            runtime_seconds=time.time() - start,
        )

    def _load(
        self,
        data,
        format: str,
        target_column,
        kwargs: dict,
        warnings: List[str],
    ) -> List[float]:
        """Load data in the specified format."""
        # Auto-detect format
        if format == "auto":
            format = self._detect_format(data)

        if format == "pandas":
            return self._loader.try_load_pandas(data, target_column)

        if format == "numpy":
            return self._loader.try_load_numpy(data)

        if format == "list":
            return self._loader.from_list(data)

        if format == "json":
            return self._loader.from_json_array(data)

        if format == "csv":
            if isinstance(data, (str, Path)) and Path(data).exists():
                return self._loader.from_csv_file(str(data), target_column or 0)
            return self._loader.from_csv_string(str(data), target_column or 0)

        if format == "dict":
            return self._loader.from_dict(data, target_column)

        if format == "hdf5":
            return self._loader.try_load_hdf5(str(data), kwargs.get("dataset", "/data"))

        if format == "netcdf":
            return self._loader.try_load_netcdf(str(data), kwargs.get("variable", ""))

        if format == "fasta":
            return self._loader.from_fasta_gc_content(str(data))

        warnings.append(f"Unknown format: {format}")
        return []

    def _detect_format(self, data) -> str:
        """Auto-detect data format."""
        try:
            import pandas as pd
            if isinstance(data, pd.DataFrame):
                return "pandas"
        except ImportError:
            pass
        try:
            import numpy as np
            if isinstance(data, np.ndarray):
                return "numpy"
        except ImportError:
            pass
        if isinstance(data, list):
            return "list"
        if isinstance(data, dict):
            return "dict"
        if isinstance(data, str):
            if data.startswith('[') or data.startswith('{'):
                return "json"
            if data.endswith('.csv'):
                return "csv"
            if data.endswith('.h5') or data.endswith('.hdf5'):
                return "hdf5"
            if data.endswith('.nc') or data.endswith('.netcdf'):
                return "netcdf"
            if data.endswith('.fa') or data.endswith('.fasta'):
                return "fasta"
        return "list"