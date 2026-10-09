"""Accessibility diagnostics: is one measure representative of the others?

Writes to outputs/qa/accessibility/. Everything is MSOA11 level (n = 6,791), baseline 2014 (DfT) and
2004 town centres. Higher = LESS accessible for every variable here.
"""

from __future__ import annotations

import sys
import warnings

warnings.filterwarnings("ignore")
import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from utils import PROJECT_ROOT, get_logger

log = get_logger("15c_accessibility_diagnostics")
BASE = PROJECT_ROOT / "data" / "processed" / "msoa_baseline_characteristics.parquet"
BOUND = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa11_boundaries" / "MSOA11_BGC_England.geojson"
OUT = PROJECT_ROOT / "outputs" / "qa" / "accessibility"
SVCS = ["employment", "primary_school", "secondary_school", "further_education", "gp", "hospital", "food_store", "town_centre"]
MODERATORS = {"income": "log_baseline_income", "deprivation": "deprivation_moderator_value",
              "social_rent": "social_rent_share_2011", "log_density": "log_population_density_2011",
              "age_65plus": "age_share_65plus_2011", "degree_share": "degree_share_2011"}


def md(df: pd.DataFrame, nd: int = 3) -> str:
    cols = [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.{nd}f}" if isinstance(v, (float, np.floating)) else str(v) for v in r.values) + " |")
    return "\n".join(out)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    b = pd.read_parquet(BASE)
    full = pd.read_parquet(PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_accessibility2014.parquet")
    extra = [c for c in full.columns if c not in b.columns]
    b = b.merge(full[["msoa11cd"] + extra], on="msoa11cd", how="left")     # individual services live in the interim table only
    pt = {s: f"access_{s}_pt_min_2014" for s in SVCS}
    car = {s: f"access_{s}_car_min_2014" for s in SVCS}
    comp = ["access_keyservices_pt_min_2014", "access_keyservices_car_min_2014", "dist_town_centre_km", "dist_town_centre_centroid_km"]
    allv = list(pt.values()) + list(car.values()) + comp
    short = {v: v.replace("access_", "").replace("_min_2014", "") for v in allv}
    D = b[allv].rename(columns=short)

    # 1. correlations among all accessibility measures
    pear, spear = D.corr(), D.corr(method="spearman")
    pear.to_csv(OUT / "corr_pearson_all_access.csv"); spear.to_csv(OUT / "corr_spearman_all_access.csv")
    for name, m in (("pearson", pear), ("spearman", spear)):
        fig, ax = plt.subplots(figsize=(13, 11))
        sns.heatmap(m, cmap="RdBu_r", vmin=-1, vmax=1, center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, square=True, ax=ax)
        ax.set_title(f"Accessibility measures, {name} (MSOA11, higher = less accessible)")
        fig.savefig(OUT / f"corr_{name}_heatmap.png", dpi=130, bbox_inches="tight"); plt.close(fig)

    # 2. how well does each measure track the composite? (the representativeness question)
    rows = []
    for mode, d_, c in (("pt", pt, "keyservices_pt"), ("car", car, "keyservices_car")):
        for s in SVCS:
            v = short[d_[s]]
            rows.append({"mode": mode, "measure": s, "pearson_with_composite": D[v].corr(D[c]), "spearman_with_composite": D[v].corr(D[c], method="spearman"),
                         "pearson_with_dist_town_centre": D[v].corr(D["dist_town_centre_km"]),
                         "spearman_with_dist_town_centre": D[v].corr(D["dist_town_centre_km"], method="spearman")})
        rows.append({"mode": mode, "measure": "composite", "pearson_with_composite": 1.0, "spearman_with_composite": 1.0,
                     "pearson_with_dist_town_centre": D[c].corr(D["dist_town_centre_km"]),
                     "spearman_with_dist_town_centre": D[c].corr(D["dist_town_centre_km"], method="spearman")})
    rep = pd.DataFrame(rows)
    rep.to_csv(OUT / "representativeness_each_measure_vs_composite_and_distance.csv", index=False)

    # 3. PCA of the eight service times (z-scored, then logged to limit the effect of remote outliers), per mode
    pca_rows, load_rows = [], []
    for mode, d_ in (("pt", pt), ("car", car)):
        for tr in ("minutes", "log_minutes"):
            X = b[[d_[s] for s in SVCS]].copy()
            X = np.log(X) if tr == "log_minutes" else X
            Z = StandardScaler().fit_transform(X)
            p = PCA().fit(Z)
            for i, (ev, sh) in enumerate(zip(p.explained_variance_, p.explained_variance_ratio_)):
                pca_rows.append({"mode": mode, "transform": tr, "component": i + 1, "eigenvalue": ev, "share_variance": sh})
            for s, l in zip(SVCS, p.components_[0]):
                load_rows.append({"mode": mode, "transform": tr, "service": s, "PC1_loading": l * np.sign(p.components_[0].sum())})
            pc1 = p.transform(Z)[:, 0] * np.sign(p.components_[0].sum())
            comp_v = b[f"access_keyservices_{mode}_min_2014"]
            pca_rows.append({"mode": mode, "transform": tr, "component": "corr(PC1, composite) / corr(PC1, dist_town_centre)",
                             "eigenvalue": np.corrcoef(pc1, comp_v)[0, 1], "share_variance": np.corrcoef(pc1, b["dist_town_centre_km"])[0, 1]})
    pd.DataFrame(pca_rows).to_csv(OUT / "pca_eight_services.csv", index=False)
    pd.DataFrame(load_rows).to_csv(OUT / "pca_eight_services_pc1_loadings.csv", index=False)

    # 4. public transport vs car
    modecmp = pd.DataFrame([{"measure": s, "pearson_pt_vs_car": b[pt[s]].corr(b[car[s]]), "spearman_pt_vs_car": b[pt[s]].corr(b[car[s]], method="spearman"),
                             "mean_pt_min": b[pt[s]].mean(), "mean_car_min": b[car[s]].mean()} for s in SVCS] +
                           [{"measure": "keyservices composite", "pearson_pt_vs_car": b[comp[0]].corr(b[comp[1]]),
                             "spearman_pt_vs_car": b[comp[0]].corr(b[comp[1]], method="spearman"),
                             "mean_pt_min": b[comp[0]].mean(), "mean_car_min": b[comp[1]].mean()}])
    modecmp.to_csv(OUT / "public_transport_vs_car.csv", index=False)

    # 5. overlap with the existing moderators
    mm = pd.DataFrame({k: b[v] for k, v in MODERATORS.items()})
    ov = pd.concat([mm, D[["keyservices_pt", "keyservices_car", "dist_town_centre_km"]]], axis=1).corr().loc[
        ["keyservices_pt", "keyservices_car", "dist_town_centre_km"], list(MODERATORS)]
    ov.to_csv(OUT / "correlation_with_existing_moderators_pearson.csv")
    ovs = pd.concat([mm, D[["keyservices_pt", "keyservices_car", "dist_town_centre_km"]]], axis=1).corr(method="spearman").loc[
        ["keyservices_pt", "keyservices_car", "dist_town_centre_km"], list(MODERATORS)]
    ovs.to_csv(OUT / "correlation_with_existing_moderators_spearman.csv")

    # 6. skewness / extremes
    cs = [short[c] for c in comp]
    ext = pd.DataFrame({"mean": D[cs].mean(), "median": D[cs].median(), "p99": D[cs].quantile(.99), "max": D[cs].max(), "skew": D[cs].skew()})
    ext["n_msoa_above_5x_median"] = [(D[c] > 5 * D[c].median()).sum() for c in cs]
    ext.to_csv(OUT / "composite_and_distance_distribution.csv")
    top = b.loc[b["access_keyservices_car_min_2014"].nlargest(8).index, ["msoa11cd", "msoa11nm", "access_keyservices_car_min_2014", "access_keyservices_pt_min_2014", "dist_town_centre_km"]]
    top.to_csv(OUT / "least_accessible_msoas.csv", index=False)

    # 7. scatter + maps
    fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
    ax[0].scatter(D["dist_town_centre_km"], D["keyservices_pt"], s=3, alpha=.3); ax[0].set_xlabel("distance to town centre (km)"); ax[0].set_ylabel("key-services average, public transport (min)")
    ax[1].scatter(D["dist_town_centre_km"], D["keyservices_car"], s=3, alpha=.3); ax[1].set_xlabel("distance to town centre (km)"); ax[1].set_ylabel("key-services average, car (min)")
    ax[2].scatter(D["keyservices_car"], D["keyservices_pt"], s=3, alpha=.3); ax[2].set_xlabel("car (min)"); ax[2].set_ylabel("public transport (min)")
    for a in ax:
        a.grid(alpha=.25)
    fig.suptitle("Accessibility measures, MSOA11 (higher = less accessible)")
    fig.savefig(OUT / "scatter_composite_vs_distance_and_modes.png", dpi=130, bbox_inches="tight"); plt.close(fig)
    g = gpd.read_file(BOUND).rename(columns={"MSOA11CD": "msoa11cd"}).merge(b[["msoa11cd"] + comp[:3]], on="msoa11cd")
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))
    for a, (c, t) in zip(axes, [(comp[0], "Key-services average, public transport/walk (min)"), (comp[1], "Key-services average, car (min)"), (comp[2], "Distance to nearest town centre (km)")]):
        g.plot(column=c, ax=a, cmap="viridis_r", legend=True, linewidth=0, vmax=g[c].quantile(.98), legend_kwds={"shrink": .5}); a.set_title(t, fontsize=10); a.set_axis_off()
    fig.savefig(OUT / "maps_accessibility.png", dpi=90, bbox_inches="tight"); plt.close(fig)

    # 8. report
    pcs = pd.read_csv(OUT / "pca_eight_services.csv")
    first = pcs[pcs["component"].astype(str).isin(["1", "2", "3"])]
    L = ["# Accessibility measures: representativeness diagnostics\n",
         "Generated by `src/15c_accessibility_diagnostics.py`. MSOA11 level, n = 6,791. DfT Journey Time Statistics 2014 (LSOA11, population-weighted to MSOA11) and "
         "English Town Centres 2004. **Higher = less accessible for every variable** (minutes or km).\n",
         "## 1. Each measure against the composite and against distance to town centre\n", md(rep), "",
         "## 2. PCA of the eight service times (share of variance in components 1-3)\n", md(first[["mode", "transform", "component", "eigenvalue", "share_variance"]]),
         "\nCorrelation of PC1 with the composite, and with the straight-line distance:\n",
         md(pcs[pcs["component"].astype(str).str.startswith("corr")][["mode", "transform", "eigenvalue", "share_variance"]].rename(
             columns={"eigenvalue": "corr(PC1, composite)", "share_variance": "corr(PC1, dist_town_centre)"})), "",
         "## 3. Public transport vs car\n", md(modecmp), "",
         "## 4. Correlation with the existing moderators (Pearson; Spearman in the CSV)\n", md(ov.reset_index().rename(columns={"index": "measure"})), "",
         "## 5. Distribution of the composites and distance\n", md(ext.reset_index().rename(columns={"index": "measure"})),
         "\nMost remote MSOA11s (these drive the long right tail; z-scores of the raw variables are large for them):\n", md(top), "",
         "Figures: `scatter_composite_vs_distance_and_modes.png`, `maps_accessibility.png`, `corr_*_heatmap.png`. Full matrices: `corr_pearson_all_access.csv`, `corr_spearman_all_access.csv`.\n"]
    (OUT / "ACCESSIBILITY_REPORT.md").write_text("\n".join(L))
    log.info("Wrote ACCESSIBILITY_REPORT.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
