import urllib.request
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig

url = "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.csv"
try:
    with urllib.request.urlopen(url) as response:
        content = response.read().decode()

    lines = [l for l in content.split('\n') if not l.startswith('#') and l.strip()]
    co2_values = []
    for line in lines[1:]:
        parts = line.split(',')
        if len(parts) >= 4:
            try:
                val = float(parts[3])
                if val > 0:
                    co2_values.append(val)
            except ValueError:
                pass

    print(f"Loaded {len(co2_values)} monthly CO2 measurements")
    print(f"Range: {min(co2_values):.2f} to {max(co2_values):.2f} ppm")

    co2_int = [int(round(v)) for v in co2_values[:300]]

    router = HierarchicalSearchRouter(RouterConfig(
        beam_width=20, max_depth=5, n_iterations=12
    ))
    result = router.search(co2_int, alphabet_size=100)

    print(f"Expression: {result.expr.to_string() if result.expr else 'None'}")
    print(f"MDL cost:   {result.mdl_cost:.2f}")
    print(f"Family:     {result.math_family.name}")

except Exception as e:
    print(f"Download failed: {e} — using cached data")