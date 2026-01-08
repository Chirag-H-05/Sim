import requests
import pandas as pd

URL = "https://facilities.aicte-india.org/dashboard/pages/php/approvedintakeserver.php"

params = {
    "method": "fetchdata",
    "year": "2024-2025",
    "program": "Engineering and Technology",
    "level": "1",
    "institutiontype": "1",
    "state": "Karnataka",
}

headers = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://facilities.aicte-india.org/dashboard/pages/angulardashboard.php",
    "X-Requested-With": "XMLHttpRequest"
}

r = requests.get(URL, params=params, headers=headers)
r.raise_for_status()

print(r.text[:500])      # inspect
data = r.json()          # usually works
print(type(data), len(data))

df = pd.DataFrame(data)
print(df.head())
