"""Build the data files and figures the brief lists:
1. New housing completions by year
2. Distribution of new builds per 1,000 households
3. Median existing-home price/m2 by year
4-7. Construction intensity by income/deprivation/social-rent/density quartile
8. England map of cumulative construction intensity
9. England map of baseline deprivation
10. Scatter/binned plot of price growth against construction
11. Data for later marginal-effect plots (percentile grid of the primary
    treatment and each moderator z-score)
"""

from __future__ import annotations

import sys

import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from utils import PROJECT_ROOT, get_duckdb_connection, get_logger, load_config

log = get_logger("19_make_figures")

FINAL_PANEL = PROJECT_ROOT / "data" / "processed" / "final_msoa_year_dissertation_panel.parquet"
TRANS_TRIMMED = PROJECT_ROOT / "data" / "interim" / "geography_crosswalks" / "transactions_england_2012_2023_trimmed_geo.parquet"
BOUNDARIES_GEOJSON = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa21_boundaries" / "MSOA21_BSC_England.geojson"
FIG_DIR = PROJECT_ROOT / "outputs" / "figures"
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"


def savefig(fig, name: str) -> None:
    path = FIG_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info(f"Wrote {path}")


def main() -> int:
    if not FINAL_PANEL.exists():
        log.error(f"{FINAL_PANEL} not found - run src/14_merge_final_panel.py first.")
        return 1

    cfg = load_config()
    reg_start = cfg["study_period"]["regression_sample_start_year"]
    reg_end = cfg["study_period"]["regression_sample_end_year"]

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    con = get_duckdb_connection()
    con.execute(f"CREATE OR REPLACE VIEW panel AS SELECT * FROM read_parquet('{FINAL_PANEL.as_posix()}')")

    # --- 1. New housing completions by year (national) ---
    completions_by_year = con.sql(
        "SELECT year, sum(newbuild_total) AS newbuild_total FROM panel GROUP BY 1 ORDER BY 1"
    ).df()
    completions_by_year.to_csv(TABLES_DIR / "fig1_completions_by_year.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(completions_by_year["year"], completions_by_year["newbuild_total"])
    ax.set_title("New housing completions by year (England)")
    ax.set_xlabel("Year")
    ax.set_ylabel("New-build dwellings")
    savefig(fig, "fig1_completions_by_year.png")

    # --- 2. Distribution of new builds per 1,000 households ---
    intensity = con.sql("SELECT newbuilds_per_1000 FROM panel").df()
    intensity.to_csv(TABLES_DIR / "fig2_newbuild_intensity_raw.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(intensity["newbuilds_per_1000"].clip(upper=intensity["newbuilds_per_1000"].quantile(0.99)), bins=50)
    ax.set_title("Distribution of new-builds per 1,000 households (99th pct capped)")
    ax.set_xlabel("New-builds per 1,000 households")
    ax.set_ylabel("MSOA-years")
    savefig(fig, "fig2_newbuild_intensity_distribution.png")

    # --- 3. Median existing-home price/m2 by year (national) ---
    con.execute(f"CREATE OR REPLACE VIEW trans AS SELECT * FROM read_parquet('{TRANS_TRIMMED.as_posix()}')")
    price_by_year = con.sql(
        "SELECT transaction_year AS year, median(nominal_ppsqm) AS median_ppsqm FROM trans GROUP BY 1 ORDER BY 1"
    ).df()
    price_by_year.to_csv(TABLES_DIR / "fig3_median_ppsqm_by_year.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(price_by_year["year"], price_by_year["median_ppsqm"], marker="o")
    ax.set_title("Median existing-home price per m2 by year (England)")
    ax.set_xlabel("Year")
    ax.set_ylabel("Median £/m2")
    savefig(fig, "fig3_median_ppsqm_by_year.png")

    # --- 4-7. Construction intensity by quartile (income/deprivation/social-rent/density) ---
    quartile_specs = [
        ("income_quartile", "fig4_construction_by_income_quartile", "Baseline income quartile"),
        ("deprivation_quartile", "fig5_construction_by_deprivation_quartile", "Deprivation quartile"),
        ("social_rent_quartile", "fig6_construction_by_socialrent_quartile", "Social-rent-share quartile"),
        ("density_quartile", "fig7_construction_by_density_quartile", "Population-density quartile"),
    ]
    for col, fname, xlabel in quartile_specs:
        df = con.sql(
            f"""
            SELECT "{col}" AS quartile, avg(newbuilds_lag1_per_1000) AS mean_newbuilds_lag1_per_1000
            FROM panel WHERE "{col}" IS NOT NULL GROUP BY 1 ORDER BY 1
            """
        ).df()
        df.to_csv(TABLES_DIR / f"{fname}.csv", index=False)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(df["quartile"].astype(str), df["mean_newbuilds_lag1_per_1000"])
        ax.set_title(f"Construction intensity by {xlabel.lower()}")
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Mean newbuilds_lag1_per_1000")
        savefig(fig, f"{fname}.png")

    # --- 8 & 9: England maps ---
    if BOUNDARIES_GEOJSON.exists():
        gdf = gpd.read_file(BOUNDARIES_GEOJSON)

        cumulative = con.sql(
            "SELECT msoa21cd_reference, sum(newbuilds_per_1000) AS cumulative_newbuilds_per_1000 FROM panel "
            "WHERE msoa21cd_reference IS NOT NULL GROUP BY 1"
        ).df()
        cumulative.to_csv(TABLES_DIR / "fig8_cumulative_construction_intensity.csv", index=False)
        map_df = gdf.merge(cumulative, left_on="MSOA21CD", right_on="msoa21cd_reference", how="left")
        fig, ax = plt.subplots(figsize=(7, 8))
        map_df.plot(
            column="cumulative_newbuilds_per_1000", cmap="viridis", legend=True, ax=ax,
            missing_kwds={"color": "lightgrey"}, vmax=map_df["cumulative_newbuilds_per_1000"].quantile(0.98),
        )
        ax.set_title("Cumulative construction intensity, 2012-2023\n(new-builds per 1,000 households)")
        ax.set_axis_off()
        savefig(fig, "fig8_map_cumulative_construction_intensity.png")

        deprivation = con.sql(
            "SELECT DISTINCT msoa21cd_reference, deprivation_moderator_value, deprivation_moderator_used "
            "FROM panel WHERE msoa21cd_reference IS NOT NULL"
        ).df()
        dep_name = deprivation["deprivation_moderator_used"].iloc[0]
        deprivation.to_csv(TABLES_DIR / "fig9_baseline_deprivation.csv", index=False)
        map_df2 = gdf.merge(deprivation, left_on="MSOA21CD", right_on="msoa21cd_reference", how="left")
        fig, ax = plt.subplots(figsize=(7, 8))
        map_df2.plot(
            column="deprivation_moderator_value", cmap="magma_r", legend=True, ax=ax,
            missing_kwds={"color": "lightgrey"},
        )
        ax.set_title(f"Baseline deprivation, IMD 2015: {dep_name}\nhigher = more deprived; excludes the housing\naffordability domain", fontsize=10)
        ax.set_axis_off()
        savefig(fig, "fig9_map_baseline_deprivation.png")
    else:
        log.warning(f"{BOUNDARIES_GEOJSON} not found - skipping the two map figures.")

    # --- 10. Price growth vs construction (binned scatter) ---
    growth = con.sql(
        f"""
        WITH by_msoa_year AS (
            SELECT msoa11cd, year, log_median_ppsqm, newbuilds_per_1000 FROM panel
        ),
        start_end AS (
            SELECT
                msoa11cd,
                max(CASE WHEN year = {reg_start} THEN log_median_ppsqm END) AS log_ppsqm_start,
                max(CASE WHEN year = {reg_end} THEN log_median_ppsqm END) AS log_ppsqm_end,
                sum(newbuilds_per_1000) AS cumulative_newbuilds_per_1000
            FROM by_msoa_year
            WHERE year BETWEEN {reg_start} AND {reg_end}
            GROUP BY msoa11cd
        )
        SELECT msoa11cd, cumulative_newbuilds_per_1000,
               (log_ppsqm_end - log_ppsqm_start) AS log_price_growth
        FROM start_end
        WHERE log_ppsqm_start IS NOT NULL AND log_ppsqm_end IS NOT NULL
        """
    ).df()
    growth.to_csv(TABLES_DIR / "fig10_price_growth_vs_construction.csv", index=False)

    growth["construction_bin"] = pd.qcut(growth["cumulative_newbuilds_per_1000"], 20, duplicates="drop")
    binned = growth.groupby("construction_bin", observed=True).agg(
        mean_construction=("cumulative_newbuilds_per_1000", "mean"),
        mean_price_growth=("log_price_growth", "mean"),
    ).reset_index(drop=True)
    binned.to_csv(TABLES_DIR / "fig10_binned_price_growth_vs_construction.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(binned["mean_construction"], binned["mean_price_growth"], s=30)
    ax.set_title(f"Price growth ({reg_start}-{reg_end}) vs cumulative construction (binned)")
    ax.set_xlabel("Mean cumulative new-builds per 1,000 households")
    ax.set_ylabel(f"Mean log price/m2 growth, {reg_start}-{reg_end}")
    savefig(fig, "fig10_price_growth_vs_construction.png")

    # --- 11. Data for later marginal-effect plots ---
    treatment_pctiles = con.sql(
        """
        SELECT quantile_cont(newbuilds_lag1_per_1000, 0.10) AS p10,
               quantile_cont(newbuilds_lag1_per_1000, 0.25) AS p25,
               quantile_cont(newbuilds_lag1_per_1000, 0.50) AS p50,
               quantile_cont(newbuilds_lag1_per_1000, 0.75) AS p75,
               quantile_cont(newbuilds_lag1_per_1000, 0.90) AS p90
        FROM panel
        """
    ).df()
    moderator_grid = pd.DataFrame({"zscore_value": [-2, -1, 0, 1, 2]})
    treatment_pctiles.to_csv(TABLES_DIR / "fig11_marginal_effects_treatment_percentiles.csv", index=False)
    moderator_grid.to_csv(TABLES_DIR / "fig11_marginal_effects_moderator_grid.csv", index=False)
    log.info(
        "Wrote marginal-effects grid data: treatment percentiles "
        f"{treatment_pctiles.iloc[0].to_dict()} x moderator z-scores {moderator_grid['zscore_value'].tolist()}"
    )

    log.info("Figures complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
