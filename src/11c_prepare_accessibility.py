"""Baseline accessibility: DfT Journey Time Statistics 2014 (LSOA11) aggregated to MSOA11, plus a
straight-line distance to the nearest 2004 town centre.

Variables (all fixed baseline, one value per MSOA11; higher = LESS accessible):
  access_{service}_{pt|car}_min_2014   population-weighted mean of LSOA minutes to the nearest service
                                       (8 services x 2 modes). Weights = the table's own service-user
                                       population (working-age for employment, age groups for schools,
                                       households for GP / hospital / food / town centre).
  access_keyservices_{pt|car}_min_2014 DfT's "key services average": equal-weighted mean of the eight
                                       MSOA-level service times.
  dist_town_centre_km                  straight-line km from the MSOA11 population-weighted centroid to the
                                       nearest English Town Centres 2004 polygon (0 if inside one).
LSOA11 nests exactly in MSOA11, so aggregation uses the existing lookup with no crosswalk, and never
averages ranks. Times are capped by DfT at 120 minutes; the share capped is reported.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import xlrd

from utils import PROJECT_ROOT, get_logger, load_config

log = get_logger("11c_prepare_accessibility")

LSOA_MSOA_CSV = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa_crosswalks" / "LSOA11_MSOA11_LAD11_EW_distinct.csv"
DENSITY = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_census2011_density.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa" / "accessibility"


def fail(msg: str) -> int:
    log.error(f"STOP: {msg}")
    return 1


def read_jts_table(path, sheet_title_contains: str) -> pd.DataFrame:
    """Read a DfT JTS .xls table: title in row 3, header in row 6, data from row 7."""
    wb = xlrd.open_workbook(path)
    sheet = next(s for s in wb.sheets() if s.name.upper().startswith("JTS"))
    title = str(sheet.row_values(3)[0])
    if sheet_title_contains.lower() not in title.lower():
        raise ValueError(f"{path.name}: title '{title}' does not mention '{sheet_title_contains}'")
    header = [str(c).strip() for c in sheet.row_values(6)]
    rows = [sheet.row_values(r) for r in range(7, sheet.nrows)]
    df = pd.DataFrame(rows, columns=header)
    return df.rename(columns={"LSOA_code": "lsoa11cd"}).dropna(subset=["lsoa11cd"])


def weighted_mean(df: pd.DataFrame, value: str, weight: str, by: str) -> pd.Series:
    w = df[weight]
    num = (df[value] * w).groupby(df[by]).sum()
    den = w.groupby(df[by]).sum()
    return num / den


def main() -> int:
    cfg = load_config()["accessibility"]
    QA_DIR.mkdir(parents=True, exist_ok=True)
    data_dir = PROJECT_ROOT / cfg["data_dir"]
    cap = cfg["time_cap_minutes"]

    lk = pd.read_csv(LSOA_MSOA_CSV, dtype=str)
    lk.columns = ["lsoa11cd", "msoa11cd", "lad11cd", "lad11nm"]
    universe = set(pd.read_parquet(DENSITY)["msoa11cd"])

    # ---- 1. read the eight LSOA tables, check structure, build one LSOA table ----
    lsoa = None
    qa_rows = []
    for svc, sc in cfg["services"].items():
        try:
            df = read_jts_table(data_dir / sc["file"], sc["title_contains"])
        except ValueError as e:
            return fail(str(e))
        log.info(f"{sc['file']} [{svc}]: {len(df):,} rows; columns used: "
                 f"{sc['pop']}, " + ", ".join(f"{sc['prefix']}{m}" for m in cfg["modes"].values()))
        need = [sc["pop"]] + [f"{sc['prefix']}{m}" for m in cfg["modes"].values()]
        missing = [c for c in need if c not in df.columns]
        if missing:
            return fail(f"{sc['file']}: expected columns missing: {missing}")
        n_dup = int(df["lsoa11cd"].duplicated().sum())
        if n_dup:
            d = df[df["lsoa11cd"].duplicated(keep=False)]
            if len(d.drop_duplicates()) != d["lsoa11cd"].nunique():
                return fail(f"{sc['file']}: duplicate LSOA rows that are not identical.")
            log.warning(f"{sc['file']}: {n_dup} exact duplicate LSOA row(s) dropped (identical values).")
            df = df.drop_duplicates()
        if len(df) != cfg["expected_n_lsoa"] or df["lsoa11cd"].duplicated().any():
            return fail(f"{sc['file']}: expected {cfg['expected_n_lsoa']:,} unique LSOA11s, found {len(df):,}.")
        if not df["lsoa11cd"].str.startswith("E01").all():
            return fail(f"{sc['file']}: non-England LSOA codes present.")
        part = df[["lsoa11cd", sc["pop"]]].rename(columns={sc["pop"]: f"{svc}_pop"})
        for mode, suffix in cfg["modes"].items():
            col = f"{sc['prefix']}{suffix}"
            t = pd.to_numeric(df[col], errors="raise")
            if t.isna().any() or (t < 0).any() or (t > cap).any():
                return fail(f"{sc['file']} {col}: missing or out-of-range times (valid range 0-{cap}).")
            part[f"{svc}_{mode}"] = t.values
            qa_rows.append({"service": svc, "mode": mode, "n_lsoa": len(t), "mean": t.mean(), "median": t.median(),
                            "max": t.max(), "share_at_cap": float((t >= cap).mean())})
        if (part[f"{svc}_pop"] < 0).any():
            return fail(f"{sc['file']}: negative service-user population.")
        n_zero = int((part[f"{svc}_pop"] == 0).sum())
        if n_zero:
            log.info(f"{sc['file']}: {n_zero} LSOAs have zero service users ({sc['users']}); they get zero weight.")
        lsoa = part if lsoa is None else lsoa.merge(part, on="lsoa11cd", how="outer", validate="1:1")
    if len(lsoa) != cfg["expected_n_lsoa"] or lsoa.isna().any().any():
        return fail("LSOA sets differ across the eight tables (or missing values after merge).")
    log.info("Assert OK: 32,844 LSOA11s in all eight tables, no missing values, times within 0-120.")

    # nesting and coverage
    lsoa = lsoa.merge(lk[["lsoa11cd", "msoa11cd"]], on="lsoa11cd", how="left", validate="1:1")
    if lsoa["msoa11cd"].isna().any():
        return fail(f"{int(lsoa['msoa11cd'].isna().sum())} LSOA11s have no MSOA11 in the lookup.")
    if set(lsoa["msoa11cd"]) != universe or len(universe) != cfg["expected_n_msoa11"]:
        return fail("MSOA11 coverage differs from the panel universe (6,791).")
    log.info("Assert OK: each LSOA11 maps to exactly one MSOA11; all 6,791 MSOA11s covered.")
    qa = pd.DataFrame(qa_rows)
    qa.to_csv(QA_DIR / "jts2014_lsoa_time_summary.csv", index=False)
    log.info(f"LSOA-level summary of times and share at the 120-minute cap:\n{qa.round(3).to_string(index=False)}")

    # ---- 2. aggregate to MSOA11 (population-weighted means of LSOA minutes) ----
    out = pd.DataFrame(index=sorted(universe))
    out.index.name = "msoa11cd"
    for svc in cfg["services"]:
        if (lsoa.groupby("msoa11cd")[f"{svc}_pop"].sum() <= 0).any():
            return fail(f"some MSOA11s have zero total service-user population for {svc}.")
    diffs = []
    for svc in cfg["services"]:
        for mode in cfg["modes"]:
            pw = weighted_mean(lsoa, f"{svc}_{mode}", f"{svc}_pop", "msoa11cd")
            out[f"access_{svc}_{mode}_min_2014"] = pw
            unw = lsoa.groupby("msoa11cd")[f"{svc}_{mode}"].mean()
            diffs.append({"service": svc, "mode": mode, "max_abs_diff_vs_unweighted": float((pw - unw).abs().max()),
                          "corr_with_unweighted": float(pw.corr(unw))})
    pd.DataFrame(diffs).to_csv(QA_DIR / "weighting_sensitivity_weighted_vs_unweighted.csv", index=False)
    # independent SQL-style check of one aggregation
    chk = lsoa.assign(x=lambda d: d["employment_pt"] * d["employment_pop"]).groupby("msoa11cd")[["x", "employment_pop"]].sum()
    if not np.allclose(chk["x"] / chk["employment_pop"], out["access_employment_pt_min_2014"]):
        return fail("independent re-computation of the employment aggregation disagrees.")

    # ---- 3. composites (DfT key services average = mean of the eight service times) ----
    for mode in cfg["modes"]:
        cols = [f"access_{svc}_{mode}_min_2014" for svc in cfg["services"]]
        out[f"access_keyservices_{mode}_min_2014"] = out[cols].mean(axis=1)
    # sensitivity: composite at LSOA level (equal mean of eight services) then weighted by employment population
    for mode in cfg["modes"]:
        l = lsoa[[f"{s}_{mode}" for s in cfg["services"]]].mean(axis=1)
        alt = (l * lsoa["employment_pop"]).groupby(lsoa["msoa11cd"]).sum() / lsoa.groupby("msoa11cd")["employment_pop"].sum()
        log.info(f"Composite ({mode}): corr of MSOA-level equal-weighted vs LSOA-level/working-age-weighted = "
                 f"{out[f'access_keyservices_{mode}_min_2014'].corr(alt):.5f}; max abs diff {float((out[f'access_keyservices_{mode}_min_2014'] - alt).abs().max()):.3f} min")
    nat = {m: float(np.average(lsoa[[f"{s}_{m}" for s in cfg['services']]].mean(axis=1), weights=lsoa["employment_pop"])) for m in cfg["modes"]}
    log.info(f"England working-age-weighted key-services average, from these LSOA tables: {nat} minutes "
             "(DfT published 2014: about 17 min public transport/walk, 10 min car).")

    # ---- 4. straight-line distance to nearest town centre (British National Grid, metres -> km) ----
    import geopandas as gpd
    pwc = gpd.read_file(PROJECT_ROOT / cfg["msoa11_pwc_geojson"]).set_crs(27700, allow_override=True)
    tc = gpd.read_file(PROJECT_ROOT / cfg["town_centres_geojson"]).set_crs(4326, allow_override=True).to_crs(27700)
    if len(pwc) != cfg["expected_n_msoa11"] or set(pwc["msoa11cd"]) != universe:
        return fail("population-weighted centroid file does not cover the 6,791 MSOA11s.")
    near = gpd.sjoin_nearest(pwc[["msoa11cd", "geometry"]], tc[["ID", "NAME", "geometry"]], how="left", distance_col="d_m")
    near = near.sort_values(["msoa11cd", "d_m", "ID"]).drop_duplicates("msoa11cd")
    cen = gpd.GeoDataFrame(tc[["ID"]].assign(cx=tc["CENTROIDX"], cy=tc["CENTROIDY"]),
                           geometry=gpd.points_from_xy(tc["CENTROIDX"], tc["CENTROIDY"]), crs=27700)
    near_c = gpd.sjoin_nearest(pwc[["msoa11cd", "geometry"]], cen[["ID", "geometry"]], how="left", distance_col="d_m")
    near_c = near_c.sort_values(["msoa11cd", "d_m", "ID"]).drop_duplicates("msoa11cd").set_index("msoa11cd")["d_m"]
    d = near.set_index("msoa11cd")
    out["dist_town_centre_km"] = (d["d_m"] / 1000).reindex(out.index)
    out["dist_town_centre_nearest_name"] = d["NAME"].reindex(out.index)
    out["dist_town_centre_centroid_km"] = (near_c / 1000).reindex(out.index)      # sensitivity: to polygon centroid
    inside = float((out["dist_town_centre_km"] == 0).mean())
    log.info(f"Town centres: {len(tc)} polygons (DfT cites 1,211). Distance km: mean {out['dist_town_centre_km'].mean():.2f}, "
             f"median {out['dist_town_centre_km'].median():.2f}, max {out['dist_town_centre_km'].max():.1f}; "
             f"{inside:.1%} of MSOA11 centroids lie inside a town-centre polygon. Corr(polygon, centroid distance) = "
             f"{out['dist_town_centre_km'].corr(out['dist_town_centre_centroid_km']):.4f}")
    if out["dist_town_centre_km"].isna().any():
        return fail("missing town-centre distances.")

    # ---- 5. transforms: England-only z-scores (higher = less accessible) and logs for the headline measures ----
    heads = ["access_keyservices_pt_min_2014", "access_keyservices_car_min_2014", "dist_town_centre_km"]
    for c in heads:
        out[c.replace("_2014", "").replace("_min", "") + "_z" if c.startswith("access") else "dist_town_centre_z"] = \
            (out[c] - out[c].mean()) / out[c].std(ddof=0)
        out["log_" + c] = np.log1p(out[c]) if c == "dist_town_centre_km" else np.log(out[c])
    out = out.reset_index()
    if out.isna().drop(columns=["dist_town_centre_nearest_name"]).any().any():
        return fail("missing values in the accessibility table.")
    out.to_parquet(OUT_DIR / "msoa11_accessibility2014.parquet", compression="zstd", index=False)
    log.info(f"Wrote {OUT_DIR / 'msoa11_accessibility2014.parquet'} ({len(out):,} rows x {out.shape[1]} cols)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
