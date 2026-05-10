"""
Run this first: python diagnose_sources.py
Tells you exactly which data sources work from your machine.
"""
import urllib.request
import json
import time

TESTS = [
    ("FRED",        "https://fred.stlouisfed.org/graph/fredgraph.csv?id=UNRATE"),
    ("FRED2",       "https://api.stlouisfed.org/fred/series/observations?series_id=UNRATE&api_key=&file_type=json"),
    ("World Bank",  "https://api.worldbank.org/v2/country/WLD/indicator/NY.GDP.MKTP.KD.ZG?format=json&per_page=10"),
    ("OWID CO2",    "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"),
    ("OWID COVID",  "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"),
    ("USGS",        "https://waterservices.usgs.gov/nwis/iv/?sites=09380000&parameterCd=00060&startDT=2020-01-01&endDT=2020-12-31&format=json"),
    ("Open-Meteo",  "https://archive-api.open-meteo.com/v1/era5?latitude=52.52&longitude=13.41&start_date=2020-01-01&end_date=2020-12-31&daily=temperature_2m_mean"),
    ("ECB Stats",   "https://data-api.ecb.europa.eu/service/data/ICP/M.DE.N.000000.4.ANR?format=csvdata"),
    ("WHO GHO",     "https://ghoapi.azureedge.net/api/WHOSIS_000001?$filter=SpatialDim eq 'GLOBAL'"),
    ("NASDAQ",      "https://data.nasdaq.com/api/v3/datasets/FRED/GDP.csv?rows=10"),
    ("GitHub",      "https://raw.githubusercontent.com/jbrownlee/Datasets/master/daily-min-temperatures.csv"),
    ("Pomber COVID","https://pomber.github.io/covid19/timeseries.json"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

print("Testing data sources...\n")
working = []
for name, url in TESTS:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as r:
            data = r.read(500)
            print(f"  ✓ {name:<20} status={r.status}, preview: {data[:60]!r}")
            working.append(name)
    except Exception as e:
        print(f"  ✗ {name:<20} {str(e)[:60]}")
    time.sleep(0.3)

print(f"\nWorking: {working}")
print("\n--- statsmodels ---")
try:
    import statsmodels.api as sm
    print("  ✓ statsmodels available")
    datasets = ['sunspots', 'co2', 'elnino', 'macrodata', 'modechoice', 'fair']
    for d in datasets:
        try:
            ds = getattr(sm.datasets, d).load_pandas().data
            print(f"    {d}: {len(ds)} rows, cols={list(ds.columns)[:4]}")
        except: pass
except ImportError:
    print("  ✗ not installed — run: pip install statsmodels")

print("\n--- sklearn ---")
try:
    from sklearn import datasets as skds
    print("  ✓ sklearn available")
except ImportError:
    print("  ✗ not installed")

print("\n--- yfinance ---")
try:
    import yfinance as yf
    print("  ✓ yfinance available")
except ImportError:
    print("  ✗ not installed — run: pip install yfinance")