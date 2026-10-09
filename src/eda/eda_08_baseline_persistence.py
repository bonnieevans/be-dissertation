"""EDA 08 - persistence of the baseline (2011) characteristics: how much did places move by 2021?

Why: the model fixes each moderator at its 2011 value. If MSOAs keep their relative position between 2011
and 2021, the 2011 value is a good stand-in for the neighbourhood type throughout the 2016-2023 window; if
places re-order, the baseline measures the moderator with error (attenuation of the interaction terms).

Method: 2011 values (this project's MSOA11 baseline table) are compared with the SAME definition built from the
2021 Census at MSOA21, linked through the ONS best-fit MSOA11 -> MSOA21 lookup. Two samples:
  (A) "unchanged": MSOA11 code equals its MSOA21 code and no other MSOA11 maps to that MSOA21;
  (B) "all linked": every MSOA11 with its best-fit MSOA21 (boundaries may differ; merged/split areas included).
Rank-based measures are the relevant ones because the model uses England-wide z-scores/quartiles.
Variables with a 2021 Census equivalent: social-rent share, density, age groups, degree share, unemployment.
Income (SAIE FYE2012) and IMD 2015 have no 2021 equivalent here and are not covered.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eda_common import *  # noqa: F401,F403
import seaborn as sns
from report_utils import md_table

log = get_logger("eda_08_baseline_persistence")
style()
OUT = out_dir(EDA_DIR, "persistence")
RAW21 = PROJECT_ROOT / "data" / "raw" / "census2021"
TEN21 = PROJECT_ROOT / "data" / "raw" / "census" / "tenure_ts054" / "ts054_hh_tenure_9a_msoa_2021.json"
LOOKUP = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa_crosswalks" / "MSOA11_MSOA21_LAD22_EW_LU_v2.csv"
AGE11 = PROJECT_ROOT / "data" / "raw" / "census2011" / "agestructure_ks102ew.xlsx"

# Age groups whose edges are available in BOTH censuses (2011 bands: 0-4,5-7,8-9,10-14,15,16-17,...; 2021: 5-year bands)
AGE_GROUPS_2011 = {
    "age_0_14": ["Age 0 to 4", "Age 5 to 7", "Age 8 to 9", "Age 10 to 14"],
    "age_15_24": ["Age 15", "Age 16 to 17", "Age 18 to 19", "Age 20 to 24"],
    "age_25_44": ["Age 25 to 29", "Age 30 to 44"],
    "age_45_64": ["Age 45 to 59", "Age 60 to 64"],
    "age_65plus": ["Age 65 to 74", "Age 75 to 84", "Age 85 to 89", "Age 90 and over"],
}
AGE_GROUPS_2021 = {
    "age_0_14": ["Aged 4 years and under", "Aged 5 to 9 years", "Aged 10 to 14 years"],
    "age_15_24": ["Aged 15 to 19 years", "Aged 20 to 24 years"],
    "age_25_44": ["Aged 25 to 29 years", "Aged 30 to 34 years", "Aged 35 to 39 years", "Aged 40 to 44 years"],
    "age_45_64": ["Aged 45 to 49 years", "Aged 50 to 54 years", "Aged 55 to 59 years", "Aged 60 to 64 years"],
    "age_65plus": ["Aged 65 to 69 years", "Aged 70 to 74 years", "Aged 75 to 79 years", "Aged 80 to 84 years", "Aged 85 years and over"],
}
LABEL = {"social_rent_share": "Social-rented share of households", "log_density": "Log population density",
         "degree_share": "Level 4+ qualification share (16+)", "unemployment_rate": "Unemployed / economically active",
         "age_0_14": "Age 0-14 share", "age_15_24": "Age 15-24 share", "age_25_44": "Age 25-44 share",
         "age_45_64": "Age 45-64 share", "age_65plus": "Age 65+ share"}


def build_2011() -> pd.DataFrame:
    b = pd.read_parquet(BASELINE)
    d = pd.DataFrame({"msoa11cd": b.msoa11cd, "social_rent_share": b.social_rent_share_2011, "log_density": b.log_population_density_2011,
                      "degree_share": b.degree_share_2011, "unemployment_rate": b.unemployment_rate_2011}).set_index("msoa11cd")
    spec = importlib.util.spec_from_file_location("age11", PROJECT_ROOT / "src" / "11b_prepare_census2011_age.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    raw = mod.read_nomis_sheet()
    raw = raw[raw.msoa11cd.str.startswith("E02")].set_index("msoa11cd")
    for g, cols in AGE_GROUPS_2011.items():
        d[g] = raw[cols].astype(float).sum(axis=1) / raw["All usual residents"].astype(float)
    return d


def read_nomis(name: str, cat_col: str) -> pd.DataFrame:
    x = pd.read_csv(RAW21 / f"{name}.csv", usecols=["GEOGRAPHY_CODE", cat_col, "OBS_VALUE"])
    x = x[x.GEOGRAPHY_CODE.str.startswith("E02")]
    return x.pivot_table(index="GEOGRAPHY_CODE", columns=cat_col, values="OBS_VALUE", aggfunc="sum")


def build_2021() -> pd.DataFrame:
    age = read_nomis("age_ts007a_msoa21", "C2021_AGE_19_NAME")
    ea = read_nomis("economic_activity_ts066_msoa21", "C2021_EASTAT_20_NAME")
    qu = read_nomis("qualification_ts067_msoa21", "C2021_HIQUAL_8_NAME")
    dn = pd.read_csv(RAW21 / "density_ts006_msoa21.csv", usecols=["GEOGRAPHY_CODE", "OBS_VALUE"])
    dn = dn[dn.GEOGRAPHY_CODE.str.startswith("E02")].set_index("GEOGRAPHY_CODE")["OBS_VALUE"]
    obs = json.load(open(TEN21))["observations"]
    rows = [(o["dimensions"][0]["option_id"], o["dimensions"][1]["option_id"], o["observation"]) for o in obs]
    t = pd.DataFrame(rows, columns=["msoa", "cat", "n"])
    t = t[t.msoa.str.startswith("E02")].pivot_table(index="msoa", columns="cat", values="n", aggfunc="sum")
    d = pd.DataFrame(index=age.index)
    hh = t[["0", "1", "2", "3", "4", "5", "6", "7"]].sum(axis=1)
    d["social_rent_share"] = (t["3"] + t["4"]).reindex(d.index) / hh.reindex(d.index)
    d["log_density"] = np.log(dn.reindex(d.index))
    d["degree_share"] = qu["Level 4 qualifications or above"] / qu["Total: All usual residents aged 16 years and over"]
    active = ea["Economically active (excluding full-time students)"] + ea["Economically active and a full-time student"]
    d["unemployment_rate"] = ea["Economically active (excluding full-time students): Unemployed"] / active
    for g, cols in AGE_GROUPS_2021.items():
        d[g] = age[cols].sum(axis=1) / age["Total"]
    return d


def quart(s: pd.Series) -> pd.Series:
    return pd.qcut(s.rank(method="first"), 4, labels=[1, 2, 3, 4]).astype(int)


def main() -> int:
    d11, d21 = build_2011(), build_2021()
    assert len(d11) == 6791 and len(d21) == 6856, (len(d11), len(d21))
    lk = pd.read_csv(LOOKUP, dtype=str)
    lk = lk[lk.MSOA11CD.str.startswith("E02")][["MSOA11CD", "MSOA21CD"]].rename(columns={"MSOA11CD": "msoa11cd", "MSOA21CD": "msoa21cd"})
    n_to = lk.groupby("msoa21cd")["msoa11cd"].transform("nunique")
    lk["same_code"] = lk.msoa11cd == lk.msoa21cd
    lk["unchanged"] = lk.same_code & (n_to == 1)
    structure = pd.DataFrame({
        "msoa11_total": [len(lk)], "msoa21_total_england": [len(d21)],
        "msoa11_code_equals_msoa21_code": [int(lk.same_code.sum())], "unchanged_sample_A": [int(lk.unchanged.sum())],
        "msoa21_receiving_2plus_msoa11": [int((lk.groupby("msoa21cd").size() > 1).sum())],
        "msoa11_in_a_merged_msoa21": [int((n_to > 1).sum())],
        "msoa21_not_a_best_fit_of_any_msoa11": [int((~d21.index.isin(lk.msoa21cd)).sum())]})
    structure.to_csv(OUT / "boundary_linkage_structure.csv", index=False)
    log.info(f"Linkage structure:\n{structure.T}")

    # Quartiles are formed within each census year's own England distribution (what the model's z-score/quartile does)
    vars_ = list(LABEL)
    q11 = d11[vars_].apply(quart)
    q21 = d21[vars_].apply(quart)
    rows, trans = [], {}
    for sample, mask in (("A_unchanged_codes", lk.unchanged), ("B_all_linked_best_fit", pd.Series(True, index=lk.index))):
        L = lk[mask]
        for v in vars_:
            a = d11.loc[L.msoa11cd, v].values
            b = d21.loc[L.msoa21cd, v].values
            s1, s2 = pd.Series(a), pd.Series(b)
            ra, rb = s1.rank(pct=True), s2.rank(pct=True)
            qa = q11.loc[L.msoa11cd, v].values
            qb = q21.loc[L.msoa21cd, v].values
            rows.append({"sample": sample, "variable": v, "label": LABEL[v], "n": len(L),
                         "mean_2011": s1.mean(), "mean_2021": s2.mean(),
                         "pearson": s1.corr(s2), "spearman": s1.corr(s2, method="spearman"),
                         "mean_abs_rank_percentile_change": float((ra - rb).abs().mean()),
                         "share_same_quartile": float((qa == qb).mean()),
                         "share_moving_2plus_quartiles": float((abs(qa - qb) >= 2).mean()),
                         "top_quartile_2011_still_top": float((qb[qa == 4] == 4).mean()),
                         "bottom_quartile_2011_still_bottom": float((qb[qa == 1] == 1).mean())})
            if sample.startswith("A"):
                trans[v] = pd.crosstab(pd.Series(qa, name="quartile 2011"), pd.Series(qb, name="quartile 2021"), normalize="index")
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "persistence_summary.csv", index=False)
    pd.concat({k: v for k, v in trans.items()}).to_csv(OUT / "quartile_transition_matrices_sample_A.csv")
    log.info("Persistence (sample A, unchanged codes):\n" + res[res["sample"].str.startswith("A")][
        ["variable", "n", "pearson", "spearman", "share_same_quartile", "share_moving_2plus_quartiles"]].round(3).to_string(index=False))

    # persistence by region and by 2011 density quartile (sample A): does stability differ by area type?
    base = load_baseline().set_index("msoa11cd")
    A = lk[lk.unchanged]
    rr = []
    for v in vars_:
        a = d11.loc[A.msoa11cd, v]
        b = pd.Series(d21.loc[A.msoa21cd, v].values, index=A.msoa11cd)
        dq = base.loc[A.msoa11cd, "density_quartile"] if "density_quartile" in base else None
        grp = pd.qcut(base.loc[A.msoa11cd, "population_density_2011"].rank(method="first"), 4, labels=[1, 2, 3, 4])
        for qv in (1, 2, 3, 4):
            m = (grp == qv).values
            rr.append({"variable": v, "density_quartile_2011": qv, "n": int(m.sum()), "spearman": a[m].corr(b[m], method="spearman")})
    pd.DataFrame(rr).to_csv(OUT / "persistence_by_density_quartile_sample_A.csv", index=False)

    # figure: rank-rank scatter, sample A
    fig, axes = plt.subplots(3, 3, figsize=(14, 13))
    for ax, v in zip(axes.ravel(), vars_):
        a = d11.loc[A.msoa11cd, v].rank(pct=True).values
        b = d21.loc[A.msoa21cd, v].rank(pct=True).values
        ax.hexbin(a, b, gridsize=40, cmap="Blues", mincnt=1)
        ax.set_title(f"{LABEL[v]}\nSpearman {res[(res.variable == v) & res['sample'].str.startswith('A')].spearman.iloc[0]:.2f}", fontsize=9)
        ax.set_xlabel("rank percentile 2011")
        ax.set_ylabel("rank percentile 2021")
        ax.plot([0, 1], [0, 1], color="r", lw=.7)
    savefig(fig, OUT / "rank_rank_2011_vs_2021_sample_A.png")

    # report
    A_res = res[res["sample"].str.startswith("A")][["label", "n", "mean_2011", "mean_2021", "pearson", "spearman", "mean_abs_rank_percentile_change",
                                                   "share_same_quartile", "share_moving_2plus_quartiles", "top_quartile_2011_still_top", "bottom_quartile_2011_still_bottom"]]
    B_res = res[res["sample"].str.startswith("B")][["label", "n", "pearson", "spearman", "share_same_quartile", "share_moving_2plus_quartiles"]]
    byq = pd.DataFrame(rr).pivot_table(index="variable", columns="density_quartile_2011", values="spearman").reset_index()
    L = ["# Persistence of the 2011 baseline characteristics (2011 vs 2021)\n",
         "Generated by `src/eda/eda_08_baseline_persistence.py`. 2011 = this project's MSOA11 baseline values; 2021 = the same definition built from the 2021 Census "
         "(Nomis TS007A, TS066, TS067, TS006; ONS TS054) at MSOA21, linked by the ONS best-fit lookup. Rank-based measures are the relevant ones because the model "
         "uses England-wide z-scores. Quartiles are formed within each census year's own England distribution. Income (SAIE) and IMD 2015 have no 2021 equivalent here.\n",
         "## Boundary linkage\n", md_table(structure.T.reset_index().rename(columns={"index": "item", 0: "count"}), nd=0), "",
         "## Sample A: MSOAs whose code is unchanged (n shown)\n", md_table(A_res, nd=3), "",
         "## Sample B: all MSOA11s via best-fit MSOA21 (boundaries may differ)\n", md_table(B_res, nd=3), "",
         "## Spearman by 2011 density quartile (sample A; 1 = least dense)\n", md_table(byq, nd=3), "",
         "Figures: `rank_rank_2011_vs_2021_sample_A.png`. Transition matrices: `quartile_transition_matrices_sample_A.csv`.\n",
         "Notes: the 2021 Census was taken during COVID-19 restrictions (March 2021), which affects some measures (for example student numbers and economic-activity categories). "
         "The 2011 and 2021 unemployment and qualification definitions are aligned as closely as the published categories allow but are not identical (2021 economic-activity "
         "categories changed; 2021 includes some changes in how students are classified). Differences in means between years are not corrected, because the model uses ranks and z-scores.\n"]
    (OUT / "PERSISTENCE_REPORT.md").write_text("\n".join(L))
    log.info("EDA 08 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
