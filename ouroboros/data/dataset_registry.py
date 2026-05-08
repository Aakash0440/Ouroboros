"""
Known-good public time series datasets — all URLs verified from repo listing.
"""

BASE = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/"

DATASETS = [
    {"name": "daily_min_temp",      "domain": "climate",    "url": BASE + "daily-min-temperatures.csv",      "col": 1},
    {"name": "daily_max_temp",      "domain": "climate",    "url": BASE + "daily-max-temperatures.csv",      "col": 1},
    {"name": "airline_passengers",  "domain": "economics",  "url": BASE + "airline-passengers.csv",          "col": 1},
    {"name": "monthly_airline",     "domain": "economics",  "url": BASE + "monthly-airline-passengers.csv",  "col": 1},
    {"name": "sunspots",            "domain": "astronomy",  "url": BASE + "monthly-sunspots.csv",            "col": 1},
    {"name": "female_births",       "domain": "demography", "url": BASE + "daily-total-female-births.csv",   "col": 1},
    {"name": "shampoo_sales",       "domain": "economics",  "url": BASE + "monthly-shampoo-sales.csv",       "col": 1},
    {"name": "mean_temp_monthly",   "domain": "climate",    "url": BASE + "monthly-mean-temp.csv",           "col": 1},
    {"name": "car_sales",           "domain": "economics",  "url": BASE + "monthly-car-sales.csv",           "col": 1},
    {"name": "monthly_robberies",   "domain": "crime",      "url": BASE + "monthly-robberies.csv",           "col": 1},
    {"name": "champagne_sales",     "domain": "economics",  "url": BASE + "monthly_champagne_sales.csv",     "col": 1},
    {"name": "writing_paper",       "domain": "industry",   "url": BASE + "monthly-writing-paper-sales.csv", "col": 1},
    {"name": "water_usage",         "domain": "hydrology",  "url": BASE + "yearly-water-usage.csv",          "col": 1},
    {"name": "pollution",           "domain": "environment","url": BASE + "pollution.csv",                   "col": 1},
    {"name": "longley_employment",  "domain": "economics",  "url": BASE + "longley.csv",                     "col": 1},
]