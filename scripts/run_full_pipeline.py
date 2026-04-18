"""
OUROBOROS Full Pipeline — end-to-end from nothing to CRT.

Runs all three phases in sequence:
    Phase 1: Modular arithmetic emergence from MDL compression
    Phase 2: Self-modification under adversarial proof market
    Phase 3: CRT derived from joint environment compression

Usage:
    python scripts/run_full_pipeline.py              # Full run (~30 min)
    python scripts/run_full_pipeline.py --quick      # Quick test (~3 min)
    python scripts/run_full_pipeline.py --phase 1    # Phase 1 only

Outputs:
    experiments/pipeline_results/
        phase1_results.json
        phase2_results.json
        phase3_results.json
        crt_landmark_results.json
        pipeline_summary.json
        figures/
            figure1_discovery_event.png
            figure2_phase2_convergence.png
            figure3_crt_landmark.png
"""

import sys
import json
import time
import argparse
from pathlib import Path
sys.path.insert(0, '.')

from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()
RESULTS_DIR = Path('experiments/pipeline_results')
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
(RESULTS_DIR / 'figures').mkdir(exist_ok=True)


def run_phase1(quick: bool = False) -> dict:
    """Phase 1: Mathematical emergence from compression."""
    console.print("\n[bold cyan]═══ PHASE 1: MATHEMATICAL EMERGENCE ═══[/bold cyan]")

    from ouroboros.core.phase1_runner import Phase1Runner
    from ouroboros.core.config import OuroborosConfig

    cfg = OuroborosConfig()
    cfg.compression.beam_width = 15 if quick else 28
    cfg.compression.const_range = 14 if quick else 21

    runner = Phase1Runner.for_modular_arithmetic(
        7, 3, 1,
        num_agents=4 if quick else 8,
        config=cfg,
        run_dir=str(RESULTS_DIR / 'phase1_runs')
    )
    results = runner.run(
        stream_length=600 if quick else 1500,
        eval_interval=200 if quick else 300,
        consensus_threshold=0.40,
        verbose=True
    )
    runner.save_results(str(RESULTS_DIR / 'phase1_results.json'))

    # Generate Figure 1
    if results.discovery_step and runner._agents:
        best_agent_id = min(results.final_ratios, key=results.final_ratios.get)
        from ouroboros.utils.visualize import plot_discovery_event
        plot_discovery_event(
            run_dir=str(RESULTS_DIR / 'phase1_runs'),
            agent_id=best_agent_id,
            discovery_step=results.discovery_step,
            expression_found=results.discovery_expression or '',
            save_path=str(RESULTS_DIR / 'figures' / 'figure1_discovery_event.png')
        )

    return results.to_dict()


def run_phase2(quick: bool = False) -> dict:
    """Phase 2: Self-modification under adversarial proof market."""
    console.print("\n[bold cyan]═══ PHASE 2: PROOF MARKET SELF-MODIFICATION ═══[/bold cyan]")

    from ouroboros.core.phase2_runner import Phase2Runner

    runner = Phase2Runner.for_modular_arithmetic(
        7, 3, 1,
        num_agents=3 if quick else 6,
        run_dir=str(RESULTS_DIR / 'phase2_runs')
    )
    results = runner.run(
        num_rounds=5 if quick else 15,
        verbose=True
    )
    runner.save_results(str(RESULTS_DIR / 'phase2_results.json'))

    # Generate Figure 2
    try:
        from ouroboros.utils.visualize import plot_compression_curves
        plot_compression_curves(
            str(RESULTS_DIR / 'phase2_runs'),
            title='Phase 2: Compression Under Self-Modification',
            save_path=str(RESULTS_DIR / 'figures' / 'figure2_phase2_convergence.png')
        )
    except Exception:
        pass

    return results.to_dict()


def run_phase3(quick: bool = False) -> dict:
    """Phase 3: CRT from joint compression."""
    console.print("\n[bold cyan]═══ PHASE 3: CRT LANDMARK ═══[/bold cyan]")

    from ouroboros.environment.structured import ModularArithmeticEnv
    from ouroboros.environment.joint_environment import JointEnvironment
    from ouroboros.agents.synthesis_agent import SynthesisAgent
    from ouroboros.emergence.crt_detector import (
        check_behavioral_crt, gcd, crt_solution
    )
    from ouroboros.compression.program_synthesis import (
        build_linear_modular, BeamSearchSynthesizer
    )

    MOD1, SLOPE1, INT1 = 7, 3, 1
    MOD2, SLOPE2, INT2 = 11, 5, 2
    JOINT_MOD = MOD1 * MOD2

    # Build and search the joint stream
    e1 = ModularArithmeticEnv(MOD1, SLOPE1, INT1, seed=42)
    e2 = ModularArithmeticEnv(MOD2, SLOPE2, INT2, seed=42)
    joint_env = JointEnvironment(e1, e2)
    joint_env.reset(800 if quick else 2000)
    joint_stream = joint_env.peek_all()

    console.print(f"  Searching joint stream (alphabet={JOINT_MOD})...")

    num_agents = 3 if quick else 6
    agents = [
        SynthesisAgent(
            i, JOINT_MOD,
            beam_width=15 if quick else 25,
            max_depth=3,
            const_range=JOINT_MOD * 2,
            mcmc_iterations=50 if quick else 150,
            seed=42 + i * 11
        )
        for i in range(num_agents)
    ]

    found_crt = False
    best_acc = 0.0
    best_expr_str = None

    ref_e1 = build_linear_modular(SLOPE1, INT1, MOD1)
    ref_e2 = build_linear_modular(SLOPE2, INT2, MOD2)

    for agent in agents:
        agent.observe(joint_stream)
        agent.search_and_update()
        ratio = agent.measure_compression_ratio()
        if agent.best_expression:
            is_crt, acc = check_behavioral_crt(
                ref_e1, ref_e2, agent.best_expression, MOD1, MOD2, 100
            )
            if acc > best_acc:
                best_acc = acc
                best_expr_str = agent.best_expression.to_string()
                if is_crt:
                    found_crt = True
            console.print(
                f"  Agent {agent.agent_id}: ratio={ratio:.4f}  "
                f"crt_acc={acc:.1%}  "
                + ("← [gold1]CRT![/gold1]" if is_crt else "")
            )

    results = {
        'mod1': MOD1, 'mod2': MOD2, 'joint_mod': JOINT_MOD,
        'found_crt': found_crt,
        'best_crt_accuracy': round(best_acc, 4),
        'best_expression': best_expr_str,
    }
    with open(RESULTS_DIR / 'phase3_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    return results


def main():
    parser = argparse.ArgumentParser(description='OUROBOROS full pipeline')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test run (smaller parameters)')
    parser.add_argument('--phase', type=int, default=0,
                        help='Run only phase N (0=all)')
    args = parser.parse_args()

    console.print(Panel.fit(
        "[bold gold1]OUROBOROS FULL PIPELINE[/bold gold1]\n"
        "From void to Chinese Remainder Theorem\n"
        f"Mode: {'QUICK' if args.quick else 'FULL'}",
        border_style="gold1"
    ))

    start = time.time()
    pipeline_results = {}

    if args.phase in (0, 1):
        p1 = run_phase1(args.quick)
        pipeline_results['phase1'] = p1

    if args.phase in (0, 2):
        p2 = run_phase2(args.quick)
        pipeline_results['phase2'] = p2

    if args.phase in (0, 3):
        p3 = run_phase3(args.quick)
        pipeline_results['phase3'] = p3

    elapsed = time.time() - start
    pipeline_results['elapsed_seconds'] = round(elapsed, 2)

    with open(RESULTS_DIR / 'pipeline_summary.json', 'w') as f:
        json.dump(pipeline_results, f, indent=2, default=str)

    # Final summary
    console.print()
    p1_disc = pipeline_results.get('phase1', {}).get('discovery_expression', 'N/A')
    p2_conv = pipeline_results.get('phase2', {}).get('converged', False)
    p3_crt = pipeline_results.get('phase3', {}).get('found_crt', False)
    p3_acc = pipeline_results.get('phase3', {}).get('best_crt_accuracy', 0)

    console.print(Panel.fit(
        f"[bold]OUROBOROS PIPELINE COMPLETE[/bold]\n\n"
        f"Phase 1 — Mathematical Emergence:\n"
        f"  Discovered: {p1_disc}\n\n"
        f"Phase 2 — Proof Market:\n"
        f"  Converged: {'✅' if p2_conv else '⚠️ partial'}\n\n"
        f"Phase 3 — CRT:\n"
        f"  CRT Found: {'✅' if p3_crt else '⚠️ partial'}  "
        f"Accuracy: {p3_acc:.1%}\n\n"
        f"Total time: {elapsed:.0f}s\n"
        f"Results: {RESULTS_DIR}/",
        border_style="bright_green" if (p3_crt) else "yellow"
    ))


if __name__ == '__main__':
    main()