# ouroboros/core/config.py

"""
Central configuration for OUROBOROS.

Every hyperparameter lives here.
Never hardcode numbers in modules — always reference config.

Usage:
    from ouroboros.core.config import OuroborosConfig
    cfg = OuroborosConfig()                    # defaults
    cfg = OuroborosConfig.from_yaml('cfg.yml') # from file
    cfg.phase1.num_agents                      # 8
    cfg.compression.algorithm                  # 'zstd'
"""

from dataclasses import dataclass, field, asdict
from typing import Optional
import yaml


@dataclass
class Phase1Config:
    """Phase 1: MDL compression + proto-axiom emergence."""
    num_agents: int = 8
    stream_length: int = 10_000
    eval_interval: int = 200        # steps between compression logs
    max_program_length: int = 64    # bits — MDL search budget
    mdl_lambda: float = 1.0         # MDL regularization weight
    random_seed: int = 42


@dataclass
class Phase2Config:
    """Phase 2: Proof market + self-modification."""
    num_agents: int = 8
    starting_credit: float = 100.0
    bounty_amount: float = 10.0
    commit_window: int = 50         # steps agents have to commit
    reveal_window: int = 20         # steps agents have to reveal
    ood_test_envs: int = 8          # OOD environments to test each approved mod
    axiom_promotion_threshold: float = 0.95  # fraction of rounds survived


@dataclass
class Phase3Config:
    """Phase 3: Causal hierarchy + recursive ascent."""
    causal_scales: tuple = (1, 4, 16, 64)
    ood_probe_frequency: int = 500
    recursive_ascent_rounds: int = 200
    specialization_window: int = 50  # rounds to compute role from


@dataclass
class CompressionConfig:
    """Compression algorithm settings."""
    algorithm: str = 'zstd'         # zstd | entropy
    zstd_level: int = 3
    mdl_threshold: float = 0.01     # min improvement (bits/symbol) to be "real"


@dataclass
class SynthesisConfig:
    """Program synthesis settings."""
    beam_width: int = 25
    max_depth: int = 3              # max expression tree depth
    const_range: int = 20           # search constants 0..const_range
    mcmc_iterations: int = 200
    mcmc_temperature: float = 10.0
    mcmc_cooling: float = 0.98


@dataclass
class OuroborosConfig:
    """Master config — contains all sub-configs."""
    phase1: Phase1Config = field(default_factory=Phase1Config)
    phase2: Phase2Config = field(default_factory=Phase2Config)
    phase3: Phase3Config = field(default_factory=Phase3Config)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)
    log_dir: str = 'experiments/runs'
    device: str = 'cpu'

    @classmethod
    def from_yaml(cls, path: str) -> 'OuroborosConfig':
        with open(path) as f:
            d = yaml.safe_load(f) or {}
        cfg = cls()
        if 'phase1' in d:
            cfg.phase1 = Phase1Config(**d['phase1'])
        if 'phase2' in d:
            cfg.phase2 = Phase2Config(**d['phase2'])
        if 'compression' in d:
            cfg.compression = CompressionConfig(**d['compression'])
        if 'synthesis' in d:
            cfg.synthesis = SynthesisConfig(**d['synthesis'])
        return cfg

    def to_yaml(self, path: str) -> None:
        with open(path, 'w') as f:
            yaml.dump(asdict(self), f, default_flow_style=False)

    def save_run_config(self, run_dir: str) -> None:
        """Save config snapshot to a run directory for reproducibility."""
        import os
        os.makedirs(run_dir, exist_ok=True)
        self.to_yaml(f"{run_dir}/config.yml")