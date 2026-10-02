"""Download England MSOA11 boundaries (ONS BGC, December 2011) for the spatial
diagnostics (neighbour structure, Moran's I, spillover variable).

Source: ONS Open Geography Portal, ArcGIS FeatureServer
MSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3_2022. Paginated 2000 rows at
a time, England only (MSOA11CD starting 'E02'). Raw file is written once and never
modified; a manifest row is appended.
"""

from __future__ import annotations

import csv
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, get_logger  # noqa: E402

log = get_logger("eda_00_download_msoa11_boundaries")

URL = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
       "MSOA_Dec_2011_Boundaries_Generalised_Clipped_BGC_EW_V3_2022/FeatureServer/0")
OUT = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa11_boundaries" / "MSOA11_BGC_England.geojson"
MANIFEST = PROJECT_ROOT / "config" / "source_manifest.csv"
EXPECTED_ENGLAND_MSOA11 = 6791


def main() -> int:
    if OUT.exists():
        log.info(f"{OUT} already exists - raw files are never overwritten.")
        return 0
    features, offset = [], 0
    while True:
        r = requests.get(f"{URL}/query", params={
            "where": "MSOA11CD LIKE 'E02%'", "outFields": "MSOA11CD,MSOA11NM", "f": "geojson",
            "outSR": 4326, "resultOffset": offset, "resultRecordCount": 2000,
            "orderByFields": "MSOA11CD"}, timeout=120)
        r.raise_for_status()
        page = r.json().get("features", [])
        if not page:
            break
        features += page
        offset += len(page)
        log.info(f"Fetched {offset} features")
    codes = {f["properties"]["MSOA11CD"] for f in features}
    if len(codes) != EXPECTED_ENGLAND_MSOA11 or len(features) != EXPECTED_ENGLAND_MSOA11:
        log.error(f"Expected {EXPECTED_ENGLAND_MSOA11} unique England MSOA11s, got {len(codes)} / {len(features)} features.")
        return 1
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
    sha = hashlib.sha256(OUT.read_bytes()).hexdigest()
    with open(MANIFEST, "a", newline="") as f:
        csv.writer(f).writerow([
            "ONS Middle Layer Super Output Areas (December 2011) Boundaries BGC (generalised clipped), England subset",
            URL, date.today().isoformat(), OUT.name, OUT.stat().st_size, sha, "2011 boundaries",
            "Used only for neighbour structure / spatial diagnostics (eda scripts); England only (E02)."])
    log.info(f"Wrote {OUT} ({OUT.stat().st_size:,} bytes, sha256 {sha[:12]}...), {len(features)} MSOA11s.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
