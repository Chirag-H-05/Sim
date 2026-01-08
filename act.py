import requests
import pandas as pd
import time
from pathlib import Path

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
URL = "https://facilities.aicte-india.org/dashboard/pages/php/approvedinstituteserver.php"

YEARS = [
    "2012-2013","2013-2014","2014-2015","2015-2016",
    "2016-2017","2017-2018","2018-2019","2019-2020",
    "2020-2021","2021-2022","2022-2023","2023-2024",
    "2024-2025","2025-2026"
]

STATES = pd.read_csv("data/metadata/aicte_states.csv")["state"].tolist()

HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://facilities.aicte-india.org/dashboard/pages/angulardashboard.php",
    "X-Requested-With": "XMLHttpRequest"
}

COLUMNS = [
    "institute_id",
    "institute_name",
    "address",
    "district",
    "institution_type",
    "women_only",
    "minority",
    "parent_id"
]

RAW_DIR = Path("data/raw/aicte")
RAW_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE = Path("data/processed/aicte_approved_institutes_master.csv")
MASTER_FILE.parent.mkdir(parents=True, exist_ok=True)

# -------------------------------------------------
# EXTRACTION LOOP
# -------------------------------------------------
for year in YEARS:
    for state in STATES:
        print(f"▶ Fetching | Year={year} | State={state}")

        params = {
            "method": "fetchdata",
            "year": year,
            "program": "Engineering and Technology",
            "level": "1",
            "institutiontype": "1",
            "Women": "1",
            "Minority": "1",
            "state": state,
            "course": "1"
        }

        try:
            r = requests.get(URL, params=params, headers=HEADERS, timeout=30)
            r.raise_for_status()

            data = r.json()

            if not data:
                print(f"⚠ No data for {year} | {state}")
                time.sleep(1)
                continue

            df = pd.DataFrame(data, columns=COLUMNS)

            # attach metadata (CRITICAL for research)
            for k, v in params.items():
                df[k] = v

            # ---------------- RAW STORAGE ----------------
            year_dir = RAW_DIR / f"year={year}"
            year_dir.mkdir(exist_ok=True)

            raw_file = year_dir / f"state={state.replace(' ', '_')}.csv"
            df.to_csv(raw_file, index=False)

            # ---------------- MASTER APPEND ----------------
            df.to_csv(
                MASTER_FILE,
                mode="a",
                header=not MASTER_FILE.exists(),
                index=False
            )

            print(f"✅ {len(df)} rows saved")

            # polite delay (DO NOT REMOVE)
            time.sleep(1.5)

        except Exception as e:
            print(f"❌ ERROR | {year} | {state} → {e}")
            time.sleep(3)

print("🎉 AICTE extraction COMPLETE")
