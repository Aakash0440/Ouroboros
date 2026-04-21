from pathlib import Path

for fname in ["ouroboros/synthesis/long_range_beam.py",
              "ouroboros/environments/modular.py",
              "ouroboros/environments/noise.py",
              "ouroboros/environments/multi_scale.py",
              "ouroboros/environments/fibonacci_mod.py"]:
    p = Path(fname)
    src = p.read_text()
    # Make these classes not inherit from ObservationEnvironment ABC
    # Instead use a simple base
    src = src.replace("(ObservationEnvironment):", ":")
    p.write_text(src)
    print(f"Patched {fname}")
