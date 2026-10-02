"""IMD 2015 baseline deprivation measures, revised (see docs in
DATA_PROCESSING_LOG.md section E).

Why revised: the overall IMD includes the Barriers to Housing and Services
domain, whose indicators include housing affordability (house prices
relative to incomes). The outcome here is house prices, so the overall IMD
cannot be the primary moderator, and averaging LSOA *ranks* is not valid.

What this script does
  1. Loads IMD 2015 File 7 using the real column names mapped in config.yaml
     (printed first, never guessed) and asserts 32,844 England LSOA11s, that
     each nests in exactly one MSOA11, and that all 6,791 MSOA11s are covered.
  2. Rebuilds the overall IMD from the seven domains using the Technical
     Report method (rank -> R -> exponential transform -> published weights)
     and HARD-STOPS unless it reproduces the published score
     (Spearman > 0.999; max absolute difference reported).
  3. Builds reduced indices at LSOA level: imd_ex_housing (6 domains),
     imd_ex_housing_living (5), imd_ex_income_housing (5).
  4. Aggregates to MSOA11 as population-weighted means of LSOA VALUES (never
     ranks/deciles). Income rate weighted by total population (exact MSOA
     rate); Employment rate by working-age population; everything else by
     total population.
  5. Recomputes the bottom-20 share from national deciles of imd_ex_housing
     (secondary/descriptive) and keeps the old overall-IMD share as a
     legacy column that is NOT for estimation.
The z-score / quartile of the *selected* moderator are built downstream in
14_merge_final_panel.py from the config switches.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

import imd_methods as im
from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("07_prepare_imd2015")

IMD_CSV = PROJECT_ROOT / "data" / "raw" / "imd" / "imd2015" / "File_7_ID2015_ranks_deciles_scores_population.csv"
LSOA_MSOA_CSV = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa_crosswalks" / "LSOA11_MSOA11_LAD11_EW_distinct.csv"
INCOME_PARQUET = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "msoa11_baseline_income.parquet"
OUT_DIR = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks"
QA_DIR = PROJECT_ROOT / "outputs" / "qa" / "imd_revision"


def fail(msg: str) -> int:
    log.error(f"STOP: {msg}")
    return 1


def main() -> int:
    cfg = load_config()
    imd_cfg = cfg["imd2015"]
    QA_DIR.mkdir(parents=True, exist_ok=True)

    for p in (IMD_CSV, LSOA_MSOA_CSV, INCOME_PARQUET):
        if not p.exists():
            return fail(f"{p} not found (run 06_prepare_income.py first for the MSOA11 universe).")

    # ------------------------------------------------------------------
    # 1. Load + header mapping + structural assertions
    # ------------------------------------------------------------------
    raw = im.strip_columns(pd.read_csv(IMD_CSV))
    log.info("File 7 header (stripped):")
    for i, c in enumerate(raw.columns, 1):
        log.info(f"  {i:2d} {c}")
    try:
        im.assert_columns_present(raw, imd_cfg)
    except KeyError as e:
        return fail(str(e))

    cols = imd_cfg["columns"]
    lsoa_col = cols["lsoa"]
    n_lsoa = len(raw)
    if n_lsoa != imd_cfg["expected_n_lsoa"] or raw[lsoa_col].nunique() != imd_cfg["expected_n_lsoa"]:
        return fail(f"expected {imd_cfg['expected_n_lsoa']} unique England LSOA11s, found "
                    f"{n_lsoa} rows / {raw[lsoa_col].nunique()} unique.")
    log.info(f"Assert OK: {n_lsoa:,} England LSOA11s.")

    lk = pd.read_csv(LSOA_MSOA_CSV, dtype=str)
    lk = lk[lk["LSOA11CD"].isin(raw[lsoa_col])]
    if lk["LSOA11CD"].duplicated().any():
        return fail("an LSOA11 maps to more than one MSOA11 in the lookup (nesting violated).")
    if len(lk) != n_lsoa:
        return fail(f"{n_lsoa - len(lk)} LSOA11s have no MSOA11 in the lookup.")
    if not lk["MSOA11CD"].str.startswith("E02").all():
        return fail("an England LSOA11 maps to a non-England MSOA11.")
    n_msoa_lookup = lk["MSOA11CD"].nunique()
    universe = set(pd.read_parquet(INCOME_PARQUET, columns=["msoa11cd"])["msoa11cd"])
    if n_msoa_lookup != imd_cfg["expected_n_msoa11"] or set(lk["MSOA11CD"]) != universe:
        return fail(f"MSOA11 coverage mismatch: lookup has {n_msoa_lookup} MSOA11s, expected "
                    f"{imd_cfg['expected_n_msoa11']}; {len(universe ^ set(lk['MSOA11CD']))} codes differ "
                    "from the England MSOA11 universe used by the panel.")
    log.info(f"Assert OK: each LSOA11 nests in exactly one MSOA11; {n_msoa_lookup:,} MSOA11s fully covered, "
             "identical to the panel's MSOA11 universe.")

    lsoa = raw.set_index(lsoa_col)
    lsoa["msoa11cd"] = lk.set_index("LSOA11CD")["MSOA11CD"].reindex(lsoa.index)
    lsoa["pop_total"] = lsoa[cols["population_total"]].astype(float)
    lsoa["pop_working"] = lsoa[cols["population_working_age"]].astype(float)
    if lsoa[["pop_total", "pop_working"]].isna().any().any() or (lsoa["pop_total"] <= 0).any():
        return fail("missing or non-positive LSOA total population.")

    # ------------------------------------------------------------------
    # 2. Reproduce the published overall IMD (hard stop)
    # ------------------------------------------------------------------
    weights = imd_cfg["domain_weights"]
    published = lsoa[cols["overall_score"]].astype(float)
    validation_rows = []
    transformed_by_source = {}
    for source in ("published", "score"):
        tx = im.transformed_domains(lsoa, imd_cfg, source)
        transformed_by_source[source] = tx
        recombined = im.combine(tx, im.OVERALL_DOMAINS, weights, rescale=False)   # published weights AS PUBLISHED
        diff = (recombined - published).abs()
        validation_rows.append({
            "rank_source": source,
            "n_lsoa": n_lsoa,
            "weights_sum": round(sum(weights.values()), 6),
            "spearman": spearmanr(recombined, published)[0],
            "pearson": pearsonr(recombined, published)[0],
            "max_abs_diff": diff.max(),
            "mean_abs_diff": diff.mean(),
            "is_configured_source": source == imd_cfg["rank_source"],
        })
    validation = pd.DataFrame(validation_rows)
    validation.to_csv(QA_DIR / "imd_reproduction_validation.csv", index=False)
    log.info(f"Reproduction of the published IMD score from 7 recombined domains:\n{validation.to_string(index=False)}")
    chosen = validation[validation["is_configured_source"]].iloc[0]
    if chosen["spearman"] <= imd_cfg["min_spearman_reproduction"]:
        return fail(f"recombined 7-domain index does not reproduce the published IMD score "
                    f"(Spearman {chosen['spearman']:.6f} <= {imd_cfg['min_spearman_reproduction']}).")
    log.info(f"Reproduction PASSED: Spearman {chosen['spearman']:.6f}, max |diff| {chosen['max_abs_diff']:.4f} "
             f"(rank_source={imd_cfg['rank_source']}).")

    # Independent check of the transform against the official File 9 values
    file9 = PROJECT_ROOT / imd_cfg["file9_xlsx"]
    if file9.exists():
        import warnings
        import openpyxl
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            wb = openpyxl.load_workbook(file9, read_only=True)
            rows = list(wb["ID 2015 Transformed scores"].iter_rows(values_only=True))
        f9 = pd.DataFrame(rows[1:], columns=rows[0]).dropna(subset=[lsoa_col]).set_index(lsoa_col)
        f9_map = {"income": "Income", "employment": "Employment", "education": "Education, Skills and Training",
                  "health": "Health Deprivation and Disability", "crime": "Crime",
                  "barriers": "Barriers to Housing and Services", "living": "Living Environment"}
        f9_rows = []
        for d, label in f9_map.items():
            off = f9[f"{label} Score - exponentially transformed"].astype(float).reindex(lsoa.index)
            for source, tx in transformed_by_source.items():
                dd = (tx[d] - off).abs()
                f9_rows.append({"domain": d, "rank_source": source, "max_abs_diff_vs_file9": dd.max(),
                                "mean_abs_diff_vs_file9": dd.mean()})
        f9_df = pd.DataFrame(f9_rows)
        f9_df.to_csv(QA_DIR / "imd_transform_vs_file9.csv", index=False)
        log.info(f"Transform vs official File 9 (published-rank source max diff "
                 f"{f9_df[f9_df.rank_source == 'published'].max_abs_diff_vs_file9.max():.4f}; "
                 f"score-rank source {f9_df[f9_df.rank_source == 'score'].max_abs_diff_vs_file9.max():.4f}).")
    else:
        log.warning(f"{file9} not found; skipping the File 9 cross-check.")

    # Agreement of re-ranked rounded scores with the published ranks (why 'published' is the default)
    agree = []
    for d, dc in cols["domains"].items():
        r_score = lsoa[dc["score"]].rank(ascending=False, method="min")
        agree.append({"domain": d, "share_ranks_identical": float((r_score == lsoa[dc["rank"]]).mean()),
                      "max_rank_gap": float((r_score - lsoa[dc["rank"]]).abs().max()),
                      "published_ranks_unique": bool(lsoa[dc["rank"]].is_unique)})
    pd.DataFrame(agree).to_csv(QA_DIR / "imd_rank_source_agreement.csv", index=False)

    # ------------------------------------------------------------------
    # 3. LSOA-level recombined indices
    # ------------------------------------------------------------------
    tx = transformed_by_source[imd_cfg["rank_source"]]
    lsoa_out = pd.DataFrame(index=lsoa.index)
    lsoa_out["msoa11cd"] = lsoa["msoa11cd"]
    lsoa_out["pop_total"] = lsoa["pop_total"]
    lsoa_out["pop_working"] = lsoa["pop_working"]
    for d in im.OVERALL_DOMAINS:
        lsoa_out[f"x_{d}"] = tx[d]
    lsoa_out["imd_overall_recombined"] = im.combine(tx, im.OVERALL_DOMAINS, weights, rescale=False)
    for name, doms in imd_cfg["variants"].items():
        if "barriers" in doms and name != "imd_ex_income_housing":
            return fail(f"{name} must exclude the Barriers to Housing and Services domain.")
        lsoa_out[name] = im.combine(tx, doms, weights, rescale=True)
        log.info(f"{name}: domains {doms}; rescaled weights "
                 f"{ {d: round(weights[d] / sum(weights[x] for x in doms), 4) for d in doms} }")
    for name in imd_cfg["variants"]:
        if lsoa_out[name].isna().any() or not lsoa_out[name].between(0, 100).all():
            return fail(f"{name} has missing or out-of-range LSOA values.")

    # ------------------------------------------------------------------
    # 4. Bottom-20 flags (LSOA level)
    # ------------------------------------------------------------------
    dep_deciles = imd_cfg["deprived_deciles"]
    lsoa_out["imd_ex_housing_decile"] = im.decile_from_values(lsoa_out["imd_ex_housing"], n_lsoa)
    lsoa_out["imd_ex_housing_bottom20"] = lsoa_out["imd_ex_housing_decile"].isin(dep_deciles).astype(int)
    lsoa_out["overall_published_decile"] = lsoa[cols["overall_decile"]].astype(int)
    lsoa_out["overall_bottom20_legacy"] = lsoa_out["overall_published_decile"].isin(dep_deciles).astype(int)
    # sanity: decile function reproduces the published overall deciles
    dec_check = (im.decile_from_values(published, n_lsoa) == lsoa_out["overall_published_decile"]).mean()
    log.info(f"Decile function applied to the published overall IMD score matches the published deciles for {dec_check:.4%} of LSOAs.")

    # raw (untransformed) domain scores and the published overall score, for aggregation
    raw_scores = {d: lsoa[dc["score"]].astype(float) for d, dc in cols["domains"].items()}
    lsoa_out["overall_score_published"] = published
    for d, s in raw_scores.items():
        lsoa_out[f"raw_{d}"] = s
    lsoa_out.index.name = "lsoa11cd"
    lsoa_out.reset_index().to_parquet(OUT_DIR / "lsoa11_imd2015_indices.parquet", compression="zstd", index=False)

    # ------------------------------------------------------------------
    # 5. Aggregate to MSOA11 (population-weighted means of LSOA values)
    # ------------------------------------------------------------------
    g = "msoa11cd"
    agg = pd.DataFrame({
        "imd2015_total_population": lsoa_out.groupby(g)["pop_total"].sum(),
        "imd2015_working_age_population": lsoa_out.groupby(g)["pop_working"].sum(),
        "n_lsoa": lsoa_out.groupby(g).size(),
    })
    # exact MSOA income-deprivation rate = sum(rate*pop)/sum(pop)
    agg["imd2015_income_rate_msoa"] = im.pop_weighted_mean(lsoa_out, g, "raw_income", "pop_total")
    agg["imd2015_employment_rate_msoa"] = im.pop_weighted_mean(lsoa_out, g, "raw_employment", "pop_working")
    for d, label in [("education", "education"), ("health", "health"), ("crime", "crime"),
                     ("barriers", "barriers"), ("living", "living")]:
        agg[f"imd2015_{label}_score_pw"] = im.pop_weighted_mean(lsoa_out, g, f"raw_{d}", "pop_total")
    for name in imd_cfg["variants"]:
        agg[name] = im.pop_weighted_mean(lsoa_out, g, name, "pop_total")
    agg["imd2015_overall_score_pw"] = im.pop_weighted_mean(lsoa_out, g, "overall_score_published", "pop_total")
    agg["imd_ex_housing_bottom20_popshare"] = im.pop_share(lsoa_out, g, "imd_ex_housing_bottom20", "pop_total")
    agg["imd2015_overall_bottom20_popshare_legacy"] = im.pop_share(lsoa_out, g, "overall_bottom20_legacy", "pop_total")
    agg = agg.reset_index()

    # ------------------------------------------------------------------
    # 6. QA on the aggregation
    # ------------------------------------------------------------------
    if len(agg) != imd_cfg["expected_n_msoa11"]:
        return fail(f"aggregated {len(agg)} MSOA11s, expected {imd_cfg['expected_n_msoa11']}.")
    value_cols = [c for c in agg.columns if c not in ("msoa11cd",)]
    if agg[value_cols].isna().any().any():
        return fail(f"missing values after aggregation: {agg[value_cols].isna().sum()[lambda s: s > 0].to_dict()}")

    # Independent cross-check in SQL: weights sum to the MSOA population, and the
    # income rate / ex-housing index recomputed from scratch agree.
    con = get_duckdb_connection()
    con.register("lsoa_t", lsoa_out.reset_index()[["lsoa11cd", "msoa11cd", "pop_total", "pop_working",
                                                     "raw_income", "raw_employment", "imd_ex_housing"]])
    sql = con.sql("""
        SELECT msoa11cd, sum(pop_total) AS w_total, sum(pop_working) AS w_working,
               sum(raw_income * pop_total) / sum(pop_total) AS income_rate,
               sum(raw_employment * pop_working) / sum(pop_working) AS employment_rate,
               sum(imd_ex_housing * pop_total) / sum(pop_total) AS ex_housing
        FROM lsoa_t GROUP BY msoa11cd
    """).fetchdf().set_index("msoa11cd")
    chk = agg.set_index("msoa11cd").join(sql)
    pop_gap = (chk["imd2015_total_population"] - chk["w_total"]).abs().max()
    wa_gap = (chk["imd2015_working_age_population"] - chk["w_working"]).abs().max()
    inc_gap = (chk["imd2015_income_rate_msoa"] - chk["income_rate"]).abs().max()
    emp_gap = (chk["imd2015_employment_rate_msoa"] - chk["employment_rate"]).abs().max()
    exh_gap = (chk["imd_ex_housing"] - chk["ex_housing"]).abs().max()
    qa_rows = [("sum of LSOA weights vs MSOA total population (max abs gap)", pop_gap),
               ("sum of LSOA weights vs MSOA working-age population (max abs gap)", wa_gap),
               ("income rate: pandas vs independent SQL (max abs gap)", inc_gap),
               ("employment rate: pandas vs independent SQL (max abs gap)", emp_gap),
               ("imd_ex_housing: pandas vs independent SQL (max abs gap)", exh_gap),
               ("MSOA11s with missing values in any measure", 0)]
    pd.DataFrame(qa_rows, columns=["check", "value"]).to_csv(QA_DIR / "imd_aggregation_qa.csv", index=False)
    if max(pop_gap, wa_gap) > 1e-6 or max(inc_gap, emp_gap, exh_gap) > 1e-9:
        return fail(f"aggregation cross-check failed: {qa_rows}")
    log.info("Aggregation QA passed: weights sum to MSOA populations; independent SQL recomputation agrees "
             f"(max gaps {max(inc_gap, emp_gap, exh_gap):.2e}); no missing values.")

    # plausibility vs Census 2011 population (different year, so correlation only)
    log.info(f"IMD mid-2012 MSOA population total {agg['imd2015_total_population'].sum():,.0f} "
             f"across {len(agg):,} MSOA11s.")

    summary = agg[value_cols].describe().T[["count", "mean", "std", "min", "50%", "max"]].reset_index()
    summary.columns = ["measure", "n", "mean", "sd", "min", "median", "max"]
    summary.to_csv(QA_DIR / "imd2015_measures_summary.csv", index=False)
    log.info(f"MSOA11 measure summary:\n{summary.to_string(index=False)}")

    share_zero = float((agg["imd_ex_housing_bottom20_popshare"] == 0).mean())
    share_zero_legacy = float((agg["imd2015_overall_bottom20_popshare_legacy"] == 0).mean())
    log.info(f"Share of MSOAs with a zero bottom-20 share: imd_ex_housing {share_zero:.1%}; "
             f"legacy overall-IMD {share_zero_legacy:.1%} (descriptive only, zero-inflated by construction).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_parquet(OUT_DIR / "msoa11_imd2015.parquet", compression="zstd", index=False)
    log.info(f"Wrote msoa11_imd2015.parquet ({len(agg):,} rows x {agg.shape[1]} cols).")
    log.info("IMD2015 preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
