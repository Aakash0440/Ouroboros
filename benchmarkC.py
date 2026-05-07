import urllib.request
import ssl
from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
from ouroboros.search.fft_period import FFTPeriodFinder

ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://www.sidc.be/SILSO/DATA/SN_m_tot_V2.0.csv"
try:
    with urllib.request.urlopen(url, context=ctx) as response:
        content = response.read().decode()

    values = []
    for line in content.split('\n'):
        parts = line.strip().split(';')
        if len(parts) >= 4:
            try:
                val = float(parts[3])
                if val >= 0:
                    values.append(int(round(val)))
            except ValueError:
                pass

    print(f"Loaded {len(values)} monthly sunspot counts")

    finder = FFTPeriodFinder()
    periods = finder.find_periods(values[:300])
    print(f"Detected periods: {periods}")
    # Expected: period near 132 months (11 years × 12 months)

    # NEW
    max_v = max(values[:300])
    values_norm = [int(round(v / max_v * 99)) for v in values[:300]]
    one_cycle = values_norm[100:232]

    router = HierarchicalSearchRouter(RouterConfig(
        beam_width=30,
        n_iterations=25,
        max_depth=6,
        time_budget_seconds=45.0
    ))
    result = router.search(one_cycle, alphabet_size=101)

    print(f"Expression: {result.expr.to_string() if result.expr else 'None'}")
    print(f"MDL cost:   {result.mdl_cost:.2f}")
    print(f"Family:     {result.math_family.name}")

except Exception as e:
    print(f"Download failed: {e}")