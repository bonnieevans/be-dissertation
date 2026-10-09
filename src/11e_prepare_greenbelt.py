"""Baseline land-use measure: share of each MSOA11's area inside designated green belt, as at 31 March 2011.

Source: MHCLG national green belt polygons (England_Green_Belt_2010_11 = as at 31 March 2011), MSOA11 generalised clipped (BGC)
boundaries. Both are reprojected to British National Grid; green belt polygons from different local authorities are dissolved so
overlaps are not double counted; invalid geometries are repaired with make_valid. share = area(MSOA ∩ green belt) / area(MSOA).
Stability checks use the 2011/12 and 2014/15 layers. BGC boundaries are generalised, so shares are approximate near boundaries.

Variables: greenbelt_share_2011 (0-1), greenbelt_share_z (England-only unweighted z-score), greenbelt_any_2011 (1 if share >= 1%).
"""

from __future__ import annotations

import sys

import geopandas as gpd
import numpy as np
import pandas as pd

from utils import PROJECT_ROOT, get_logger

log = get_logger("11e_prepare_greenbelt")
GB_DIR = PROJECT_ROOT / "data" / "raw" / "landuse_access" / "green_belt"
BOUND = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa11_boundaries" / "MSOA11_BGC_England.geojson"
DENSITY = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_census2011_density.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa" / "greenbelt"
LAYERS = {"2011": "England_Green_Belt_2010_11_WGS84", "2012": "England_Green_Belt_2011_12_WGS84", "2015": "England_Green_Belt_2014_15_WGS84"}
OFFICIAL_HA = {"2011": 1639530.0, "2012": 1639480.0}     # MHCLG green belt statistics annex 2 (as at 31 March)


def fail(msg: str) -> int:
    log.error(f"STOP: {msg}")
    return 1


def tile_polygons(geom, cell: float = 10_000.0):
    """Cut a (multi)polygon into pieces on a regular grid so later intersections work with small, simple geometries.
    Pieces are disjoint, so areas add up exactly."""
    import shapely
    minx, miny, maxx, maxy = geom.bounds
    prep = shapely.prepared.prep(geom)
    pieces = []
    for x in np.arange(np.floor(minx / cell) * cell, maxx, cell):
        for y in np.arange(np.floor(miny / cell) * cell, maxy, cell):
            box = shapely.box(x, y, x + cell, y + cell)
            if not prep.intersects(box):
                continue
            c = shapely.clip_by_rect(geom, x, y, x + cell, y + cell)
            if not c.is_empty and c.area > 0:
                pieces.append(c)
    return pieces


def shares(msoa: gpd.GeoDataFrame, layer: str) -> tuple[pd.Series, dict]:
    import shapely
    import shapely.prepared  # noqa: F401
    g = gpd.read_file(GB_DIR / f"{layer}.geojson").to_crs(27700)
    g["geometry"] = g.geometry.make_valid()
    sum_ha = g.geometry.area.sum() / 1e4
    union = g.geometry.union_all()
    union_ha = union.area / 1e4
    log.info(f"{layer}: {len(g)} polygons dissolved ({sum_ha:,.0f} ha summed, {union_ha:,.0f} ha dissolved); tiling ...")
    tiles = tile_polygons(union)
    pieces = gpd.GeoDataFrame({"gb": np.ones(len(tiles), dtype=int)}, geometry=gpd.GeoSeries(tiles, crs=27700), crs=27700)
    log.info(f"{layer}: {len(pieces):,} tiles; intersecting with MSOA11 boundaries ...")
    inter = gpd.overlay(msoa[["msoa11cd", "geometry"]], pieces, how="intersection", keep_geom_type=True)
    inter["a"] = inter.geometry.area
    a = inter.groupby("msoa11cd")["a"].sum().reindex(msoa.msoa11cd).fillna(0.0).values
    s = pd.Series(np.clip(a / msoa.geometry.area.values, 0, 1), index=msoa.msoa11cd.values)
    info = {"layer": layer, "n_polygons": len(g), "sum_polygon_area_ha": sum_ha, "dissolved_area_ha": union_ha,
            "overlap_double_count_ha": sum_ha - union_ha, "area_inside_msoa_boundaries_ha": a.sum() / 1e4}
    return s, info


def main() -> int:
    QA_DIR.mkdir(parents=True, exist_ok=True)
    msoa = gpd.read_file(BOUND).rename(columns={"MSOA11CD": "msoa11cd"}).to_crs(27700)
    msoa["geometry"] = msoa.geometry.make_valid()
    universe = set(pd.read_parquet(DENSITY)["msoa11cd"])
    if len(msoa) != 6791 or set(msoa.msoa11cd) != universe:
        return fail("MSOA11 boundaries do not match the 6,791 MSOA11 universe.")
    res, infos = {}, []
    for yr, layer in LAYERS.items():
        s, info = shares(msoa, layer)
        res[yr] = s
        info["year_as_at_31_march"] = int(yr)
        info["official_national_ha"] = OFFICIAL_HA.get(yr, np.nan)
        info["polygon_vs_official"] = info["sum_polygon_area_ha"] / OFFICIAL_HA[yr] if yr in OFFICIAL_HA else np.nan
        info["share_of_greenbelt_area_captured_by_msoa_boundaries"] = info["area_inside_msoa_boundaries_ha"] / info["dissolved_area_ha"]
        infos.append(info)
        log.info(f"{layer}: {info}")
    pd.DataFrame(infos).to_csv(QA_DIR / "greenbelt_layer_checks.csv", index=False)

    out = pd.DataFrame({"msoa11cd": res["2011"].index, "greenbelt_share_2011": res["2011"].values})
    if out.greenbelt_share_2011.isna().any():
        return fail("missing green belt shares.")
    out["greenbelt_share_z"] = (out.greenbelt_share_2011 - out.greenbelt_share_2011.mean()) / out.greenbelt_share_2011.std(ddof=0)
    out["greenbelt_any_2011"] = (out.greenbelt_share_2011 >= 0.01).astype(int)
    d = out.greenbelt_share_2011
    log.info(f"Green belt share (as at 31 March 2011): mean {d.mean():.3f}; share of MSOA11s with any (>=1%): {out.greenbelt_any_2011.mean():.1%}; "
             f"exactly zero: {(d == 0).mean():.1%}; fully inside (>=99%): {(d >= .99).mean():.1%}; median among positive {d[d > 0].median():.3f}")
    # stability across layers and size-weighted total
    stab = pd.DataFrame({"msoa11cd": out.msoa11cd, "s2011": d.values, "s2012": res["2012"].reindex(out.msoa11cd).values, "s2015": res["2015"].reindex(out.msoa11cd).values})
    stab["abs_diff_2011_2012"] = (stab.s2011 - stab.s2012).abs(); stab["abs_diff_2011_2015"] = (stab.s2011 - stab.s2015).abs()
    stab.to_csv(QA_DIR / "greenbelt_share_2011_2012_2015_by_msoa.csv", index=False)
    summ = pd.DataFrame({"comparison": ["2011 vs 2012 layer", "2011 vs 2015 layer"],
                         "pearson": [stab.s2011.corr(stab.s2012), stab.s2011.corr(stab.s2015)],
                         "spearman": [stab.s2011.corr(stab.s2012, method="spearman"), stab.s2011.corr(stab.s2015, method="spearman")],
                         "n_msoa_with_abs_diff_over_0.05": [int((stab.abs_diff_2011_2012 > .05).sum()), int((stab.abs_diff_2011_2015 > .05).sum())],
                         "max_abs_diff": [stab.abs_diff_2011_2012.max(), stab.abs_diff_2011_2015.max()]})
    summ.to_csv(QA_DIR / "greenbelt_stability.csv", index=False)
    log.info(f"Stability:\n{summ.round(4).to_string(index=False)}")
    out.to_parquet(OUT_DIR / "msoa11_greenbelt2011.parquet", compression="zstd", index=False)
    log.info(f"Wrote msoa11_greenbelt2011.parquet ({len(out):,} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
