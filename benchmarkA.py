# URL: https://archive.ics.uci.edu/ml/machine-learning-databases/00312/
# Or use: pip install yfinance
# This is real financial data with known structure

import yfinance as yf
import pandas as pd

# 5 years of daily SPY returns
spy = yf.download("SPY", start="2018-01-01", end="2023-12-31", progress=False)
returns = spy["Close"].pct_change().dropna().values * 100  # in percent

# Test 1: Returns themselves should have near-zero autocorrelation
# Test 2: Squared returns (volatility proxy) should have positive autocorrelation
# Test 3: OUROBOROS compression ratio on returns vs squared returns
# Expected: squared returns compress better than raw returns (GARCH structure)

sq_returns = [float(r)**2 for r in returns[:500]]
raw_returns_int = [int(round(float(r) * 10)) for r in returns[:500]]
sq_returns_int  = [int(round(float(r) * 10)) for r in sq_returns[:500]]

from ouroboros.search.hierarchical_router import HierarchicalSearchRouter, RouterConfig
router = HierarchicalSearchRouter(RouterConfig(beam_width=15, n_iterations=10))
raw_result = router.search(raw_returns_int, alphabet_size=50)
sq_result  = router.search(sq_returns_int,  alphabet_size=50)

print(f"Raw returns MDL:     {raw_result.mdl_cost:.2f}")
print(f"Squared returns MDL: {sq_result.mdl_cost:.2f}")
# Expected: sq_result.mdl_cost < raw_result.mdl_cost
# (squared returns have more structure — autocorrelated volatility clusters)