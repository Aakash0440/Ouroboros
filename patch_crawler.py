"""
patch_crawler.py
Adds ~30 more datasets using sources confirmed working from diagnose_sources.py:
  - USGS (JSON endpoint works)
  - ECB Statistics (Euro-area indicators)
  - Pomber COVID (country time series)
  - More World Bank (country-level series)
  - More Open-Meteo cities
  - More Brownlee (corrected filenames)

Run: python patch_crawler.py
"""

import json, math, time, urllib.request, io, csv
from pathlib import Path
from typing import Optional, List

OUTPUT_DIR = Path("data/raw_datasets")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/csv,*/*",
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


def save(ds_id, name, domain, source, url, values, unit="", description=""):
    global added, failed
    if not values:
        print(f"  ✗ {name[:55]} — no values")
        failed += 1
        return
    record = {
        "dataset_id": ds_id, "name": name, "domain": domain,
        "source": source, "url": url, "unit": unit,
        "description": description, "n_observations": len(values),
        "values": [float(v) for v in values], "metadata": {},
    }
    with open(OUTPUT_DIR / f"{ds_id}.json", "w") as f:
        json.dump(record, f, indent=2)
    manifest_file = OUTPUT_DIR / "manifest.json"
    manifest = json.load(open(manifest_file)) if manifest_file.exists() else []
    manifest = [m for m in manifest if m["dataset_id"] != ds_id]
    manifest.append({
        "dataset_id": ds_id, "name": name, "domain": domain,
        "source": source, "n_observations": len(values),
        "download_success": True, "unit": unit,
    })
    json.dump(manifest, open(manifest_file, "w"), indent=2)
    print(f"  ✓ {name[:55]} — {len(values)} obs")
    added += 1


# ── USGS (JSON API — confirmed working) ──────────────────────────────────────

def fetch_usgs():
    print("\n" + "="*55 + "\n  USGS Streamflow (JSON API)\n" + "="*55)

    sites = [
        ("usgs_mississippi", "Mississippi River at Vicksburg",    "07289000"),
        ("usgs_colorado",    "Colorado River at Lees Ferry AZ",   "09380000"),
        ("usgs_columbia",    "Columbia River at The Dalles OR",    "14105700"),
        ("usgs_ohio",        "Ohio River at Cincinnati OH",        "03265000"),
        ("usgs_hudson",      "Hudson River at Green Island NY",    "01358000"),
        ("usgs_delaware",    "Delaware River at Trenton NJ",       "01463500"),
        ("usgs_sacramento",  "Sacramento River at Sacramento CA",  "11447650"),
        ("usgs_snake",       "Snake River at Brownlee Dam ID",     "13269000"),
        ("usgs_tennessee",   "Tennessee River at Chattanooga TN",  "03568000"),
        ("usgs_arkansas",    "Arkansas River at Little Rock AR",   "07263650"),
    ]

    for ds_id, name, site_no in sites:
        url = (
            f"https://waterservices.usgs.gov/nwis/dv/"
            f"?sites={site_no}&parameterCd=00060"
            f"&startDT=2010-01-01&endDT=2023-12-31&format=json"
        )
        content = fetch(url, timeout=25)
        if not content:
            print(f"  ✗ {name} — download failed")
            global failed
            failed += 1
            time.sleep(0.5)
            continue

        vals = []
        try:
            data = json.loads(content)
            ts = data["value"]["timeSeries"]
            if ts:
                values_list = ts[0]["values"][0]["value"]
                for entry in values_list:
                    v_str = entry.get("value", "-999999")
                    try:
                        v = float(v_str)
                        if math.isfinite(v) and v >= 0:
                            vals.append(v)
                    except: pass
        except Exception as e:
            print(f"  ✗ {name} — parse error: {e}")
            failed += 1
            time.sleep(0.5)
            continue

        save(ds_id, name, "hydrology", "USGS", url, vals[:2000], "cfs")
        time.sleep(0.6)


# ── ECB Statistics (Euro-area economic data) ─────────────────────────────────

def fetch_ecb():
    print("\n" + "="*55 + "\n  ECB Statistics\n" + "="*55)

    # ECB SDMX REST API — confirmed working
    series = [
        ("ecb_de_inflation",  "Germany Inflation Rate (HICP)",
         "ICP/M.DE.N.000000.4.ANR", "%", "economics"),
        ("ecb_fr_inflation",  "France Inflation Rate (HICP)",
         "ICP/M.FR.N.000000.4.ANR", "%", "economics"),
        ("ecb_it_inflation",  "Italy Inflation Rate (HICP)",
         "ICP/M.IT.N.000000.4.ANR", "%", "economics"),
        ("ecb_es_inflation",  "Spain Inflation Rate (HICP)",
         "ICP/M.ES.N.000000.4.ANR", "%", "economics"),
        ("ecb_eurusd",        "EUR/USD Exchange Rate",
         "EXR/D.USD.EUR.SP00.A", "rate", "finance"),
        ("ecb_eurgbp",        "EUR/GBP Exchange Rate",
         "EXR/D.GBP.EUR.SP00.A", "rate", "finance"),
        ("ecb_eurjpy",        "EUR/JPY Exchange Rate",
         "EXR/D.JPY.EUR.SP00.A", "rate", "finance"),
        ("ecb_ea_m3",         "Euro Area M3 Money Supply Growth",
         "BSI/M.U2.Y.V.M30.X.I.U2.2300.Z01.A", "%", "economics"),
        ("ecb_ea_gdp",        "Euro Area GDP Growth",
         "MNA/Q.Y.I8.W2.S1.S1.B.B1GQ._Z._Z._Z.EUR.LR.GY", "%", "economics"),
    ]

    BASE = "https://data-api.ecb.europa.eu/service/data"

    for ds_id, name, series_key, unit, domain in series:
        url = f"{BASE}/{series_key}?format=csvdata&detail=dataonly"
        content = fetch(url, timeout=20)
        if not content:
            print(f"  ✗ {name} — download failed")
            global failed2
            failed += 1
            time.sleep(0.5)
            continue

        vals = []
        try:
            reader = csv.DictReader(io.StringIO(content))
            for row in reader:
                # ECB CSV has OBS_VALUE column
                v_str = row.get("OBS_VALUE", "")
                if v_str and v_str.strip():
                    try:
                        v = float(v_str)
                        if math.isfinite(v):
                            vals.append(v)
                    except: pass
        except Exception as e:
            print(f"  ✗ {name} — parse error: {e}")
            failed += 1
            time.sleep(0.5)
            continue

        save(ds_id, name, domain, "ECB", url, vals[:2000], unit)
        time.sleep(0.4)


# ── Pomber COVID (country-level time series) ─────────────────────────────────

def fetch_pomber_covid():
    print("\n" + "="*55 + "\n  Pomber COVID Country Series\n" + "="*55)

    url = "https://pomber.github.io/covid19/timeseries.json"
    content = fetch(url, timeout=20)
    if not content:
        print("  ✗ Pomber COVID — download failed")
        return

    data = json.loads(content)
    # Pick major countries with long series
    countries = [
        "US", "Germany", "United Kingdom", "France", "Italy",
        "Brazil", "India", "Japan", "South Korea", "Australia",
    ]

    for country in countries:
        if country not in data:
            continue
        records = sorted(data[country], key=lambda r: r.get("date", ""))

        # Confirmed cases time series
        confirmed = [r["confirmed"] for r in records if r.get("confirmed") is not None]
        safe_id = country.lower().replace(" ", "_")
        save(f"covid_confirmed_{safe_id}", f"COVID Confirmed Cases — {country}",
             "public_health", "Pomber", url, confirmed, "cumulative cases")

        # Deaths time series
        deaths = [r["deaths"] for r in records if r.get("deaths") is not None]
        save(f"covid_deaths_{safe_id}", f"COVID Deaths — {country}",
             "public_health", "Pomber", url, deaths, "cumulative deaths")

        time.sleep(0.1)


# ── More World Bank (country-level) ──────────────────────────────────────────

def fetch_world_bank_countries():
    print("\n" + "="*55 + "\n  World Bank Country Series\n" + "="*55)

    # GDP per capita for major economies
    countries = [
        ("USA", "United States"), ("CHN", "China"), ("JPN", "Japan"),
        ("DEU", "Germany"), ("GBR", "UK"), ("IND", "India"),
        ("BRA", "Brazil"), ("ZAF", "South Africa"), ("NGA", "Nigeria"),
        ("MEX", "Mexico"),
    ]

    indicators = [
        ("NY.GDP.PCAP.KD", "GDP per Capita (constant 2015 USD)", "USD", "economics"),
        ("SP.DYN.LE00.IN",  "Life Expectancy at Birth",           "years", "public_health"),
    ]

    for country_code, country_name in countries:
        for ind_code, ind_name, unit, domain in indicators:
            ds_id = f"wb_{country_code.lower()}_{ind_code.replace('.','_').lower()[:15]}"
            url = (f"https://api.worldbank.org/v2/country/{country_code}"
                   f"/indicator/{ind_code}?format=json&per_page=70&mrv=70")
            content = fetch(url, timeout=15)
            if not content:
                time.sleep(0.3)
                continue
            try:
                data = json.loads(content)
                if len(data) < 2 or not data[1]:
                    time.sleep(0.3)
                    continue
                records = sorted(data[1], key=lambda r: r.get("date", "0"))
                vals = []
                for r in records:
                    v = r.get("value")
                    if v is not None:
                        try:
                            fv = float(v)
                            if math.isfinite(fv):
                                vals.append(fv)
                        except: pass
                name = f"{ind_name} — {country_name}"
                save(ds_id, name, domain, "World Bank", url, vals, unit)
            except Exception as e:
                pass
            time.sleep(0.3)


# ── More Open-Meteo cities ────────────────────────────────────────────────────

def fetch_open_meteo_extra():
    print("\n" + "="*55 + "\n  Open-Meteo Extra Cities\n" + "="*55)

    cities = [
        ("openmeteo_sydney",     "Sydney Daily Temperature",      -33.87,  151.21),
        ("openmeteo_chicago",    "Chicago Daily Temperature",      41.88,  -87.63),
        ("openmeteo_singapore",  "Singapore Daily Temperature",     1.35,  103.82),
        ("openmeteo_karachi",    "Karachi Daily Temperature",      24.86,   67.01),
        ("openmeteo_lagos",      "Lagos Daily Temperature",         6.52,    3.38),
        ("openmeteo_saopaulo",   "São Paulo Daily Temperature",   -23.55,  -46.63),
        ("openmeteo_paris",      "Paris Daily Temperature",        48.85,    2.35),
        ("openmeteo_beijing",    "Beijing Daily Temperature",      39.91,  116.39),
    ]

    for ds_id, name, lat, lon in cities:
        # Skip ones already downloaded
        if (OUTPUT_DIR / f"{ds_id}.json").exists():
            print(f"  → skipping {name} (already exists)")
            continue

        url = (
            f"https://archive-api.open-meteo.com/v1/era5"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date=2015-01-01&end_date=2023-12-31"
            f"&daily=temperature_2m_mean,precipitation_sum"
            f"&timezone=UTC"
        )
        content = fetch(url, timeout=20)
        if not content:
            print(f"  ✗ {name} — download failed")
            global failed
            failed += 1
            time.sleep(0.5)
            continue

        try:
            data = json.loads(content)
            daily = data.get("daily", {})
            temps = [v for v in daily.get("temperature_2m_mean", [])
                     if v is not None and math.isfinite(v)]
            precip = [v for v in daily.get("precipitation_sum", [])
                      if v is not None and math.isfinite(v)]
            save(ds_id, name, "climate", "Open-Meteo", url, temps, "°C")
            if precip:
                save(ds_id.replace("_temp", "_precip") + "_precip",
                     name.replace("Temperature", "Precipitation"),
                     "climate", "Open-Meteo", url, precip, "mm")
        except Exception as e:
            print(f"  ✗ {name} — {e}")
            failed += 1
        time.sleep(0.5)


# ── Brownlee corrected filenames ──────────────────────────────────────────────

def fetch_brownlee_fixes():
    print("\n" + "="*55 + "\n  Brownlee GitHub (corrected filenames)\n" + "="*55)

    BASE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master"
    datasets = [
        ("jb_champagne",   "Monthly Champagne Sales",         "monthly-champagne-sales.csv",     1, "units",   "economics"),
        ("jb_milk",        "Monthly Milk Production",         "monthly-milk-production.csv",     1, "lb/cow",  "economics"),
        ("jb_lynx",        "Annual Lynx Trappings",           "lynx.csv",                        1, "count",   "ecology"),
        ("jb_mean_temp",   "Daily Mean Temperature Melbourne","daily-average-temperature.csv",   1, "°C",      "climate"),
        ("jb_beer",        "Quarterly Beer Production AU",    "quarterly-beer-production-aus.csv",1,"ML",     "economics"),
        ("jb_electricity", "Monthly Electric AU",             "monthly-electric-au.csv",         1, "GWh",    "energy"),
        ("jb_tractor",     "Monthly Tractor Sales (fixed)",   "monthly-tractor-sales.csv",       1, "units",  "economics"),
        ("jb_exchange_au", "AUD-USD Daily Rate (fixed)",      "daily-foreign-exchange-rates.csv",1, "rate",   "finance"),
        ("jb_daily_covid", "Daily COVID19 NSW AU",            "covid-nsw.csv",                   1, "cases",  "public_health"),
        ("jb_mauna_loa",   "Mauna Loa CO2 (Brownlee)",        "monthly-co2-ppm-mauna-loa.csv",   1, "ppm",   "climate"),
    ]

    for ds_id, name, filename, col, unit, domain in datasets:
        url = f"{BASE}/{filename}"
        content = fetch(url)
        if not content:
            time.sleep(0.2)
            continue
        vals = []
        lines = content.strip().split("\n")[1:]  # skip header
        for line in lines:
            parts = line.strip().split(",")
            if col < len(parts):
                try:
                    v = float(parts[col].strip().replace('"', ''))
                    if math.isfinite(v):
                        vals.append(v)
                except: pass
        save(ds_id, name, domain, "Brownlee-GitHub", url, vals, unit)
        time.sleep(0.2)


# ── MAIN ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("PATCH CRAWLER")
    print("Output:", OUTPUT_DIR.absolute())

    fetch_usgs()
    fetch_ecb()
    fetch_pomber_covid()
    fetch_world_bank_countries()
    fetch_open_meteo_extra()
    fetch_brownlee_fixes()

    manifest = json.load(open(OUTPUT_DIR / "manifest.json"))
    total = len([m for m in manifest if m["download_success"]])
    print(f"\n{'='*55}")
    print(f"PATCH CRAWLER DONE")
    print(f"  Added this run: {added} | Failed: {failed}")
    print(f"  Total in manifest: {total}")
    print(f"{'='*55}")