"""
ResultsTable and FigureGenerator for paper-ready output.

Generates:
  - Table 1: Compression ratios across moduli (LaTeX)
  - Table 2: Convergence statistics (LaTeX)
  - Table 3: CRT accuracy vs beam width (LaTeX)
  - Figure 1: Compression ratio trajectory (landmark experiment)
  - Figure 2: Convergence step distribution (violin plot)
  - Figure 3: CRT accuracy vs alphabet size
  - Figure 4: Self-improvement gain trajectory
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Dict, Any, List


def generate_latex_table_1(results: Dict[str, Any]) -> str:
    """Table 1: Compression ratio across moduli."""
    lines = [
        "% Table 1: Compression Ratio across Prime Moduli",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Compression ratio achieved by OUROBOROS agents on ModularArithmetic",
        r"environments with different prime moduli. Lower = better. Values are",
        r"mean $\pm$ 95\% CI over 10 random seeds.}",
        r"\label{tab:compression}",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Modulus & Compression Ratio & Success Rate & Mean Rounds & CI (95\%) \\",
        r"\midrule",
    ]

    for modulus in [5, 7, 11, 13]:
        key = f"moduli_generalization_mod{modulus}"
        if key in results:
            r = results[key]
            mean = r.get("mean", 0)
            ci = r.get("ci_95", 0)
            lines.append(
                f"    {modulus} & "
                f"{mean:.4f} & "
                f"$\\pm$ {ci:.4f} \\\\"
            )

    comp_key = "compression_landmark"
    if comp_key in results:
        r = results[comp_key]
        lines.extend([
            r"\midrule",
            f"    \\textbf{{Landmark (mod=7)}} & "
            f"\\textbf{{{r.get('mean', 0):.4f}}} & "
            f"$\\pm$ {r.get('ci_95', 0):.4f} \\\\"
        ])

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table}",
    ])
    return "\n".join(lines)


def generate_latex_table_2(results: Dict[str, Any]) -> str:
    """Table 2: Convergence statistics."""
    lines = [
        "% Table 2: Convergence Statistics",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{Rounds to consensus promotion across agents.",
        r"N=8 agents, ModularArithmetic(7). Mean $\pm$ 95\% CI over 10 seeds.}",
        r"\label{tab:convergence}",
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Metric & Mean & Std & 95\% CI \\",
        r"\midrule",
    ]
    key = "convergence_rounds"
    if key in results:
        r = results[key]
        lines.extend([
            f"    Rounds to convergence & {r.get('mean', 0):.2f} & "
            f"{r.get('std', 0):.2f} & $\\pm$ {r.get('ci_95', 0):.2f} \\\\",
            f"    Minimum & {r.get('min', 0):.0f} & — & — \\\\",
            f"    Maximum & {r.get('max', 0):.0f} & — & — \\\\",
        ])
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_latex_table_3(results: Dict[str, Any]) -> str:
    """Table 3: CRT accuracy."""
    lines = [
        "% Table 3: CRT Accuracy",
        r"\begin{table}[h]",
        r"\centering",
        r"\caption{CRT joint expression discovery rate.",
        r"JointEnvironment(mod1=7, mod2=11). Mean $\pm$ 95\% CI over 10 seeds.}",
        r"\label{tab:crt}",
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Method & CRT Success Rate \\",
        r"\midrule",
    ]
    key = "crt_accuracy"
    if key in results:
        r = results[key]
        lines.append(
            f"    SparseBeamSearch & {r.get('mean', 0):.3f} $\\pm$ {r.get('ci_95', 0):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def generate_figures(results: Dict[str, Any], output_dir: Path) -> List[Path]:
    """Generate all matplotlib figures. Returns list of figure paths."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # non-interactive backend
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping figure generation")
        return []

    figures = []

    # ── Figure 1: Compression ratio distribution ──────────────────────────────
    key = "compression_landmark"
    if key in results and results[key].get("values"):
        fig, ax = plt.subplots(figsize=(6, 4))
        values = results[key]["values"]
        ax.hist(values, bins=max(3, len(values)//2), color='steelblue',
                edgecolor='white', alpha=0.85)
        ax.axvline(results[key]["mean"], color='red', linewidth=2,
                   label=f'Mean = {results[key]["mean"]:.4f}')
        ax.set_xlabel("Compression Ratio (MDL / Naive)", fontsize=12)
        ax.set_ylabel("Count (seeds)", fontsize=12)
        ax.set_title("Figure 1: Compression Ratio at Discovery Step", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3)
        path = output_dir / "fig1_compression_ratio.pdf"
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        figures.append(path)

    # ── Figure 2: Convergence step distribution ───────────────────────────────
    key = "convergence_rounds"
    if key in results and results[key].get("values"):
        fig, ax = plt.subplots(figsize=(6, 4))
        values = results[key]["values"]
        ax.violinplot([values], positions=[1], showmeans=True, showmedians=True)
        ax.scatter([1]*len(values), values, color='steelblue', alpha=0.6, zorder=5)
        ax.set_xticks([1])
        ax.set_xticklabels(["ModArith(7)"])
        ax.set_ylabel("Rounds to Consensus", fontsize=12)
        ax.set_title("Figure 2: Convergence Step Distribution", fontsize=13)
        ax.grid(alpha=0.3, axis='y')
        path = output_dir / "fig2_convergence.pdf"
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        figures.append(path)

    # ── Figure 3: Success rate across moduli ─────────────────────────────────
    moduli = [5, 7, 11, 13]
    means = []
    cis = []
    for m in moduli:
        k = f"moduli_generalization_mod{m}"
        if k in results:
            means.append(results[k]["mean"])
            cis.append(results[k]["ci_95"])
        else:
            means.append(0.0)
            cis.append(0.0)

    if any(m > 0 for m in means):
        fig, ax = plt.subplots(figsize=(6, 4))
        x = np.arange(len(moduli))
        ax.bar(x, means, yerr=cis, capsize=6, color='steelblue', alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels([f"N={m}" for m in moduli])
        ax.set_xlabel("Modulus", fontsize=12)
        ax.set_ylabel("Discovery Success Rate", fontsize=12)
        ax.set_ylim(0, 1.1)
        ax.set_title("Figure 3: Moduli Generalization", fontsize=13)
        ax.grid(alpha=0.3, axis='y')
        path = output_dir / "fig3_moduli_generalization.pdf"
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        figures.append(path)

    # ── Figure 4: Self-improvement gain ──────────────────────────────────────
    key = "self_improvement_gain"
    if key in results and results[key].get("values"):
        fig, ax = plt.subplots(figsize=(6, 4))
        values = results[key]["values"]
        ax.bar(range(len(values)), values, color='steelblue', alpha=0.85)
        ax.axhline(results[key]["mean"], color='red', linewidth=2,
                   label=f'Mean = {results[key]["mean"]:.4f}')
        ax.set_xlabel("Seed Index", fontsize=12)
        ax.set_ylabel("Relative Cost Reduction", fontsize=12)
        ax.set_title("Figure 4: Self-Improvement Gain (Layer 2)", fontsize=13)
        ax.legend(fontsize=11)
        ax.grid(alpha=0.3, axis='y')
        path = output_dir / "fig4_self_improvement.pdf"
        fig.savefig(path, bbox_inches='tight', dpi=150)
        plt.close(fig)
        figures.append(path)

    return figures


class PaperNumbersReport:
    """
    Generates the complete set of numbers for Papers 1 and 2.
    
    Call generate() to produce a markdown document with all numbers
    filled in from actual experimental results.
    """

    def __init__(self, results: Dict[str, Any]):
        self.results = results

    def _get(self, key: str, field: str, default: float = 0.0) -> float:
        r = self.results.get(key, {})
        return r.get(field, default)

    def generate(self) -> str:
        lines = [
            "# OUROBOROS — Paper Numbers Report",
            f"Generated from benchmark results (n_seeds per experiment)",
            "",
            "## Paper 1: Mathematical Emergence (NeurIPS/ICLR Target)",
            "",
            "### Table 1: Compression Ratio",
            f"- **Compression ratio at discovery (mod=7):** "
            f"{self._get('compression_landmark', 'mean'):.4f} "
            f"± {self._get('compression_landmark', 'ci_95'):.4f} (95% CI)",
            f"- **Range:** [{self._get('compression_landmark', 'min'):.4f}, "
            f"{self._get('compression_landmark', 'max'):.4f}]",
            "",
            "### Table 2: Moduli Generalization",
        ]
        for mod in [5, 7, 11, 13]:
            k = f"moduli_generalization_mod{mod}"
            lines.append(
                f"- **Mod {mod}:** success rate = "
                f"{self._get(k, 'mean'):.3f} ± {self._get(k, 'ci_95'):.3f}"
            )
        lines.extend([
            "",
            "## Paper 2: Proof Market (ICML Target)",
            "",
            "### Table 3: Convergence",
            f"- **Rounds to consensus:** "
            f"{self._get('convergence_rounds', 'mean'):.2f} "
            f"± {self._get('convergence_rounds', 'ci_95'):.2f}",
            f"- **Min/Max:** "
            f"{self._get('convergence_rounds', 'min'):.0f} / "
            f"{self._get('convergence_rounds', 'max'):.0f}",
            "",
            "### Table 4: CRT Accuracy",
            f"- **CRT success rate:** "
            f"{self._get('crt_accuracy', 'mean'):.3f} "
            f"± {self._get('crt_accuracy', 'ci_95'):.3f}",
            "",
            "### Table 5: OOD Generalization",
            f"- **OOD pass fraction:** "
            f"{self._get('ood_generalization', 'mean'):.3f} "
            f"± {self._get('ood_generalization', 'ci_95'):.3f}",
            "",
            "### Table 6: Self-Improvement",
            f"- **Relative cost reduction (Layer 2):** "
            f"{self._get('self_improvement_gain', 'mean'):.4f} "
            f"± {self._get('self_improvement_gain', 'ci_95'):.4f}",
            "",
            "## LaTeX Tables",
            "",
        ])
        return "\n".join(lines)