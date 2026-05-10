"""
extra_crawler.py
Supplements the main crawler with 60+ additional datasets from sources
that don't require authentication and are more resilient.

Run AFTER dataset_crawler.py:
    python extra_crawler.py

Adds datasets to the same data/raw_datasets/ folder.
Sources:
  statsmodels built-ins   ~12 datasets  (zero network, instant)
  Jason Brownlee GitHub   ~25 datasets  (raw GitHub CSVs, very stable)
  Our World in Data       ~10 datasets  (GitHub-hosted, no auth)
  Open-Meteo Weather      ~8 datasets   (free API, no auth)
  World Bank              ~10 datasets  (public API)
  FRED (fixed headers)    ~12 datasets  (retry with browser UA)
"""

import json, math, time, csv, io, os, urllib.request
from pathlib import Path
from typing import List, Optional

OUTPUT_DIR = Path("data/raw_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/json,*/*",
}

added = 0
failed = 0


def fetch(url: str, timeout: float = 20.0) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read().decode("utf-8", errors="replace")
    except Exception as e:
        return None


def save(ds_id, name, domain, source, url, values, unit="", description="", metadata=None):
    global added, failed
    if not values:
        print(f"  ✗ {name[:50]} — no values")
        failed += 1
        return

    record = {
        "dataset_id": ds_id, "name": name, "domain": domain,
        "source": source, "url": url, "unit": unit,
        "description": description, "n_observations": len(values),
        "values": [float(v) for v in values],
        "metadata": metadata or {},
    }
    with open(OUTPUT_DIR / f"{ds_id}.json", "w") as f:
        json.dump(record, f, indent=2)

    # Update manifest entry
    manifest_file = OUTPUT_DIR / "manifest.json"
    manifest = []
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
    # Remove old entry with same id
    manifest = [m for m in manifest if m["dataset_id"] != ds_id]
    manifest.append({
        "dataset_id": ds_id, "name": name, "domain": domain,
        "source": source, "n_observations": len(values),
        "download_success": True, "unit": unit,
    })
    with open(manifest_file, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"  ✓ {name[:50]} — {len(values)} obs")
    added += 1


def parse_csv_col(content: str, col: int = 1, sep: str = ",",
                  skip_header: bool = True) -> List[float]:
    vals = []
    lines = content.strip().split("\n")
    if skip_header and lines:
        lines = lines[1:]
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split(sep)
        if col < len(parts):
            try:
                v = float(parts[col].strip().replace(",", ""))
                if math.isfinite(v):
                    vals.append(v)
            except ValueError:
                pass
    return vals[:2000]


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 1: statsmodels built-ins (ZERO network, guaranteed to work)
# ═══════════════════════════════════════════════════════════════════════

def fetch_statsmodels():
    print("\n" + "="*55)
    print("  statsmodels built-in datasets")
    print("="*55)
    try:
        import statsmodels.api as sm
        import numpy as np
    except ImportError:
        print("  ✗ statsmodels not installed. Run: pip install statsmodels")
        return

    # sunspots
    ds = sm.datasets.sunspots.load_pandas().data
    save("sm_sunspots", "Sunspot Activity (Annual)", "astronomy",
         "statsmodels", "", ds["SUNACTIVITY"].tolist(), "count",
         "Annual sunspot counts 1700-2008")

    # CO2
    ds = sm.datasets.co2.load_pandas().data
    vals = ds["co2"].dropna().tolist()
    save("sm_co2_weekly", "Mauna Loa CO2 Weekly", "climate",
         "statsmodels", "", vals, "ppm", "Weekly CO2 at Mauna Loa 1974-1998")

    # elnino — sea surface temperature
    ds = sm.datasets.elnino.load_pandas().data
    # elnino has monthly columns (Jan-Dec) for each year — flatten into time series
    cols = [c for c in ds.columns if c != 'YEAR']
    flat = []
    for _, row in ds.iterrows():
        for c in cols:
            try:
                v = float(row[c])
                if math.isfinite(v) and v > -99:
                    flat.append(v)
            except: pass
    save("sm_elnino", "El Nino Sea Surface Temperature", "climate",
         "statsmodels", "", flat, "°C", "Monthly Pacific SST 1950-2010")

    # macrodata — multiple economic series
    ds = sm.datasets.macrodata.load_pandas().data
    for col, name, unit, domain in [
        ("realgdp",     "US Real GDP (statsmodels)",          "B$",     "economics"),
        ("realcons",    "US Real Consumption",                "B$",     "economics"),
        ("realinv",     "US Real Investment",                 "B$",     "economics"),
        ("realgovt",    "US Real Government Spending",        "B$",     "economics"),
        ("realdpi",     "US Real Disposable Income",          "B$",     "economics"),
        ("cpi",         "US CPI (statsmodels)",               "index",  "economics"),
        ("m1",          "US M1 Money Stock",                  "B$",     "economics"),
        ("tbilrate",    "US T-Bill Rate (statsmodels)",       "%",      "finance"),
        ("unemp",       "US Unemployment (statsmodels)",      "%",      "economics"),
        ("infl",        "US Inflation Rate",                  "%",      "economics"),
        ("realint",     "US Real Interest Rate",              "%",      "finance"),
    ]:
        vals = ds[col].dropna().tolist()
        save(f"sm_macro_{col}", name, domain, "statsmodels", "", vals, unit)

    # Longley dataset (multicollinear economic data)
    ds = sm.datasets.longley.load_pandas().data
    save("sm_longley_employ", "US Total Employment (Longley)", "economics",
         "statsmodels", "", ds["TOTEMP"].tolist(), "persons")

    # fair — marriage data (life satisfaction over time)
    try:
        ds = sm.datasets.fair.load_pandas().data
        save("sm_fair_affair", "Extramarital Affairs Rate (Fair)", "social_science",
             "statsmodels", "", ds["affairs"].tolist(), "count")
    except: pass

    # Interest rates
    try:
        ds = sm.datasets.interest_inflation.load_pandas().data
        save("sm_interest_rate", "Interest Rate Series", "finance",
             "statsmodels", "", ds.iloc[:, 1].dropna().tolist(), "%")
    except: pass


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 2: Jason Brownlee's curated time series (raw GitHub)
# Very stable, no auth, classic ML benchmark datasets
# ═══════════════════════════════════════════════════════════════════════

def fetch_brownlee_github():
    print("\n" + "="*55)
    print("  Jason Brownlee GitHub (classic TS benchmarks)")
    print("="*55)

    BASE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
    datasets = [
        ("jb_daily_temp",      "Daily Min Temperature Melbourne",
         "daily-min-temperatures.csv", 1, "°C", "climate"),
        ("jb_monthly_sales",   "Monthly Champagne Sales",
         "monthly-champagne.csv", 1, "units", "economics"),
        ("jb_airline",         "Monthly Airline Passengers",
         "airline-passengers.csv", 1, "passengers", "transport"),
        ("jb_shampoo_sales",   "Monthly Shampoo Sales",
         "shampoo.csv", 1, "units", "economics"),
        ("jb_monthly_births",  "Daily Total Female Births CA",
         "daily-total-female-births.csv", 1, "count", "public_health"),
        ("jb_sunspots",        "Monthly Sunspots (Brownlee)",
         "monthly-sunspots.csv", 1, "count", "astronomy"),
        ("jb_electric_prod",   "Monthly Electric Production",
         "monthly_electric.csv", 1, "GWh", "energy"),
        ("jb_gold",            "Daily Gold Price",
         "gold.csv", 1, "USD", "finance"),
        ("jb_pollution",       "Beijing PM2.5 Pollution (Hourly)",
         "pollution.csv", 2, "µg/m³", "environment"),
        ("jb_exchange_rate",   "AUD-USD Exchange Rate (Daily)",
         "daily-currency-exchange-rates.csv", 1, "rate", "finance"),
        ("jb_hotel_demand",    "Hotel Demand (Monthly)",
         "monthly-hotel-demand.csv", 1, "rooms", "economics"),
        ("jb_tractor_sales",   "Monthly Tractor Sales",
         "tractor-sales.csv", 1, "units", "economics"),
        ("jb_us_births",       "Daily US Births",
         "daily-births.csv", 1, "count", "public_health"),
        ("jb_passengers2",     "Monthly Int'l Airline Pass (2)",
         "international-airline-passengers.csv", 1, "passengers", "transport"),
        ("jb_motor_vehicle",   "Monthly Motor Vehicle Sales",
         "monthly-car-sales.csv", 1, "units", "transport"),
    ]

    for ds_id, name, filename, col, unit, domain in datasets:
        url = f"{BASE}/{filename}"
        content = fetch(url)
        if content:
            vals = parse_csv_col(content, col=col)
            save(ds_id, name, domain, "Brownlee-GitHub", url, vals, unit)
        else:
            print(f"  ✗ {name[:50]} — download failed")
            global failed
            failed += 1
        time.sleep(0.2)


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 3: Our World in Data (GitHub raw CSV — no auth)
# ═══════════════════════════════════════════════════════════════════════

def fetch_owid():
    print("\n" + "="*55)
    print("  Our World in Data (GitHub raw)")
    print("="*55)

    # CO2 and energy — subset to world aggregate
    url = "https://raw.githubusercontent.com/owid/co2-data/master/owid-co2-data.csv"
    print(f"  Fetching OWID CO2 dataset...")
    content = fetch(url, timeout=30)
    if content:
        reader = csv.DictReader(io.StringIO(content))
        rows_world = [r for r in reader if r.get("country") == "World"]
        rows_world.sort(key=lambda r: int(r.get("year", 0) or 0))

        for col, name, unit, domain in [
            ("co2",                  "World CO2 Emissions (Annual)",       "MtCO2",    "climate"),
            ("co2_per_capita",       "World CO2 per Capita",               "tCO2",     "climate"),
            ("methane",              "World Methane Emissions",             "MtCO2eq",  "climate"),
            ("nitrous_oxide",        "World Nitrous Oxide Emissions",      "MtCO2eq",  "climate"),
            ("primary_energy_cons",  "World Primary Energy Consumption",   "TWh",      "energy"),
            ("coal_co2",             "World Coal CO2 Emissions",           "MtCO2",    "climate"),
            ("oil_co2",              "World Oil CO2 Emissions",            "MtCO2",    "climate"),
            ("gas_co2",              "World Gas CO2 Emissions",            "MtCO2",    "climate"),
            ("cement_co2",           "World Cement CO2 Emissions",         "MtCO2",    "climate"),
        ]:
            vals = []
            for r in rows_world:
                v = r.get(col, "")
                try:
                    fv = float(v)
                    if math.isfinite(fv):
                        vals.append(fv)
                except: pass
            save(f"owid_{col}", name, domain, "OWID", url, vals, unit)
    else:
        print("  ✗ OWID CO2 dataset — download failed")

    time.sleep(1.0)

    # WHO/OWID vaccination data
    url2 = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/vaccinations/vaccinations.csv"
    content2 = fetch(url2, timeout=30)
    if content2:
        reader = csv.DictReader(io.StringIO(content2))
        rows_world = [r for r in reader if r.get("location") == "World"]
        rows_world.sort(key=lambda r: r.get("date", ""))
        vals = []
        for r in rows_world:
            v = r.get("total_vaccinations_per_hundred", "")
            try:
                fv = float(v)
                if math.isfinite(fv):
                    vals.append(fv)
            except: pass
        save("owid_vaccinations", "World COVID Vaccinations per 100", "public_health",
             "OWID", url2, vals, "per 100 people")
    else:
        print("  ✗ OWID vaccinations — download failed")


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 4: Open-Meteo historical weather (free, no auth)
# ═══════════════════════════════════════════════════════════════════════

def fetch_open_meteo():
    print("\n" + "="*55)
    print("  Open-Meteo Historical Weather")
    print("="*55)

    import json as jsonlib

    cities = [
        ("openmeteo_london_temp",    "London Daily Temperature",     51.51, -0.13),
        ("openmeteo_newyork_temp",   "New York Daily Temperature",   40.71, -74.01),
        ("openmeteo_tokyo_temp",     "Tokyo Daily Temperature",      35.68, 139.69),
        ("openmeteo_sydney_temp",    "Sydney Daily Temperature",     -33.87, 151.21),
        ("openmeteo_dubai_temp",     "Dubai Daily Temperature",      25.20, 55.27),
        ("openmeteo_moscow_temp",    "Moscow Daily Temperature",     55.75, 37.62),
        ("openmeteo_cairo_temp",     "Cairo Daily Temperature",      30.06, 31.25),
        ("openmeteo_chicago_temp",   "Chicago Daily Temperature",    41.88, -87.63),
    ]

    for ds_id, name, lat, lon in cities:
        url = (
            f"https://archive-api.open-meteo.com/v1/era5"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date=2015-01-01&end_date=2023-12-31"
            f"&daily=temperature_2m_mean"
            f"&timezone=UTC"
        )
        content = fetch(url, timeout=20)
        if content:
            try:
                data = jsonlib.loads(content)
                vals = [v for v in data.get("daily", {}).get("temperature_2m_mean", [])
                        if v is not None and math.isfinite(v)]
                save(ds_id, name, "climate", "Open-Meteo", url, vals, "°C")
            except Exception as e:
                print(f"  ✗ {name} — parse error: {e}")
                global failed
                failed += 1
        else:
            print(f"  ✗ {name} — download failed")
            failed += 1
        time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 5: FRED with browser User-Agent (retry)
# ═══════════════════════════════════════════════════════════════════════

def fetch_fred_retry():
    print("\n" + "="*55)
    print("  FRED (retry with browser headers)")
    print("="*55)

    fred_series = [
        ("fred_gdp",        "US Real GDP",                  "GDPC1",           "B$",    "economics"),
        ("fred_unrate",     "US Unemployment Rate",         "UNRATE",          "%",     "economics"),
        ("fred_cpi",        "US CPI",                       "CPIAUCSL",        "index", "economics"),
        ("fred_fedfunds",   "Federal Funds Rate",           "FEDFUNDS",        "%",     "finance"),
        ("fred_sp500",      "S&P 500",                      "SP500",           "index", "finance"),
        ("fred_m2",         "M2 Money Supply",              "M2SL",            "B$",    "economics"),
        ("fred_10yr",       "10-Year Treasury Rate",        "GS10",            "%",     "finance"),
        ("fred_housing",    "US Housing Starts",            "HOUST",           "K",     "economics"),
        ("fred_indpro",     "US Industrial Production",     "INDPRO",          "index", "economics"),
        ("fred_wti",        "WTI Crude Oil Price",          "MCOILWTICO",      "USD",   "commodities"),
        ("fred_gold",       "Gold Price",                   "GOLDAMGBD228NLBM","USD",   "commodities"),
        ("fred_vix",        "CBOE Volatility Index",        "VIXCLS",          "index", "finance"),
    ]

    for ds_id, name, series_id, unit, domain in fred_series:
        url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}"
        content = fetch(url)
        if content and "DATE" in content[:100]:
            vals = parse_csv_col(content, col=1, sep=",")
            save(ds_id, name, domain, "FRED", url, vals, unit)
        else:
            # Try alternate FRED endpoint
            url2 = f"https://api.stlouisfed.org/fred/series/observations?series_id={series_id}&file_type=json"
            content2 = fetch(url2)
            if content2:
                try:
                    import json as j
                    data = j.loads(content2)
                    vals = []
                    for obs in data.get("observations", []):
                        v = obs.get("value", ".")
                        if v != ".":
                            try:
                                fv = float(v)
                                if math.isfinite(fv):
                                    vals.append(fv)
                            except: pass
                    save(ds_id, name, domain, "FRED", url2, vals, unit)
                except:
                    print(f"  ✗ {name} — both endpoints failed")
                    global failed
                    failed += 1
            else:
                print(f"  ✗ {name} — both endpoints failed")
                failed += 1
        time.sleep(0.4)


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 6: World Bank Open Data API
# ═══════════════════════════════════════════════════════════════════════

def fetch_world_bank():
    print("\n" + "="*55)
    print("  World Bank Open Data")
    print("="*55)

    import json as j

    indicators = [
        ("wb_world_gdp_growth",   "World GDP Growth Rate",         "NY.GDP.MKTP.KD.ZG", "WLD", "%",     "economics"),
        ("wb_world_co2",          "World CO2 (World Bank)",         "EN.ATM.CO2E.KT",    "WLD", "kt",    "climate"),
        ("wb_world_pop",          "World Population",               "SP.POP.TOTL",        "WLD", "count", "social_science"),
        ("wb_world_lifeexp",      "World Life Expectancy",          "SP.DYN.LE00.IN",     "WLD", "years", "public_health"),
        ("wb_world_fertility",    "World Fertility Rate",           "SP.DYN.TFRT.IN",     "WLD", "births","public_health"),
        ("wb_world_inflation",    "World Inflation Rate",           "FP.CPI.TOTL.ZG",     "WLD", "%",     "economics"),
        ("wb_world_trade",        "World Trade (% of GDP)",         "NE.TRD.GNFS.ZS",     "WLD", "%",     "economics"),
        ("wb_world_electricity",  "World Electricity Access",       "EG.ELC.ACCS.ZS",     "WLD", "%",     "energy"),
        ("wb_world_internet",     "World Internet Users",           "IT.NET.USER.ZS",     "WLD", "%",     "technology"),
        ("wb_world_urban",        "World Urban Population %",       "SP.URB.TOTL.IN.ZS",  "WLD", "%",     "social_science"),
    ]

    for ds_id, name, indicator, country, unit, domain in indicators:
        url = (f"https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
               f"?format=json&per_page=100&mrv=60")
        content = fetch(url, timeout=15)
        if content:
            try:
                data = j.loads(content)
                if len(data) >= 2:
                    records = data[1] or []
                    records.sort(key=lambda r: r.get("date", "0"))
                    vals = []
                    for r in records:
                        v = r.get("value")
                        if v is not None:
                            try:
                                fv = float(v)
                                if math.isfinite(fv):
                                    vals.append(fv)
                            except: pass
                    save(ds_id, name, domain, "World Bank", url, vals, unit)
                else:
                    print(f"  ✗ {name} — empty response")
                    global failed
                    failed += 1
            except Exception as e:
                print(f"  ✗ {name} — parse error: {e}")
                failed += 1
        else:
            print(f"  ✗ {name} — download failed")
            failed += 1
        time.sleep(0.4)


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 7: yfinance (financial time series, if installed)
# ═══════════════════════════════════════════════════════════════════════

def fetch_yfinance():
    print("\n" + "="*55)
    print("  yfinance (financial data)")
    print("="*55)
    try:
        import yfinance as yf
    except ImportError:
        print("  ✗ yfinance not installed. Run: pip install yfinance")
        return

    tickers = [
        ("yf_sp500",    "S&P 500 (yfinance)",         "^GSPC",   "index",   "finance"),
        ("yf_nasdaq",   "NASDAQ Composite",            "^IXIC",   "index",   "finance"),
        ("yf_gold",     "Gold Futures (yfinance)",     "GC=F",    "USD/oz",  "commodities"),
        ("yf_oil",      "Crude Oil Futures",           "CL=F",    "USD/bbl", "commodities"),
        ("yf_btc",      "Bitcoin/USD",                 "BTC-USD", "USD",     "finance"),
        ("yf_aapl",     "Apple Stock Price",           "AAPL",    "USD",     "finance"),
        ("yf_msft",     "Microsoft Stock Price",       "MSFT",    "USD",     "finance"),
        ("yf_dxy",      "US Dollar Index",             "DX-Y.NYB","index",   "finance"),
        ("yf_vix",      "VIX Fear Index (yf)",         "^VIX",    "index",   "finance"),
        ("yf_tnx",      "10Y Treasury Yield (yf)",     "^TNX",    "%",       "finance"),
    ]

    for ds_id, name, ticker, unit, domain in tickers:
        try:
            t = yf.Ticker(ticker)
            hist = t.history(period="10y", interval="1mo")
            if hist.empty:
                hist = t.history(period="5y", interval="1mo")
            vals = hist["Close"].dropna().tolist()
            save(ds_id, name, domain, "yfinance", f"yfinance:{ticker}", vals, unit)
        except Exception as e:
            print(f"  ✗ {name} — {str(e)[:60]}")
            global failed
            failed += 1
        time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════
# SOURCE 8: More NOAA endpoints that weren't in the original crawler
# ═══════════════════════════════════════════════════════════════════════

def fetch_noaa_extra():
    print("\n" + "="*55)
    print("  NOAA Extra Endpoints")
    print("="*55)

    sources = [
        {
            "id": "noaa_sf6_monthly",
            "name": "Global Mean SF6 Monthly",
            "url": "https://gml.noaa.gov/webdata/ccgg/trends/sf6/sf6_mm_gl.txt",
            "unit": "ppt",
        },
        {
            "id": "noaa_co2_daily",
            "name": "Mauna Loa CO2 Daily",
            "url": "https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_daily_mlo.txt",
            "unit": "ppm",
        },
        {
            "id": "noaa_arctic_ice",
            "name": "Arctic Sea Ice Extent (Monthly)",
            "url": "https://masie_web.apps.nsidc.org/pub/DATASETS/NOAA/G02135/north/monthly/data/N_seaice_extent_monthly_v3.0.csv",
            "unit": "million km²",
        },
    ]

    for source in sources:
        content = fetch(source["url"])
        if content:
            vals = []
            for line in content.split("\n"):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                for col_idx in [3, 2, 1]:
                    if col_idx < len(parts):
                        try:
                            v = float(parts[col_idx])
                            if math.isfinite(v) and v > -999:
                                vals.append(v)
                                break
                        except: pass
            save(source["id"], source["name"], "climate", "NOAA",
                 source["url"], vals[:2000], source["unit"])
        else:
            print(f"  ✗ {source['name']} — download failed")
            global failed
            failed += 1
        time.sleep(0.5)


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("EXTRA CRAWLER — supplementing main dataset collection")
    print("Output:", OUTPUT_DIR.absolute())

    fetch_statsmodels()
    fetch_brownlee_github()
    fetch_owid()
    fetch_open_meteo()
    fetch_fred_retry()
    fetch_world_bank()
    fetch_yfinance()
    fetch_noaa_extra()

    # Final count from manifest
    manifest_file = OUTPUT_DIR / "manifest.json"
    if manifest_file.exists():
        with open(manifest_file) as f:
            manifest = json.load(f)
        total = len([m for m in manifest if m["download_success"]])
        print(f"\n{'='*55}")
        print(f"EXTRA CRAWLER DONE")
        print(f"  Added: {added} datasets")
        print(f"  Failed: {failed}")
        print(f"  Total in manifest (all sources): {total}")
        print(f"{'='*55}")