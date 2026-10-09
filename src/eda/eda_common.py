"""Shared helpers for the exploratory analysis (src/eda) and the baseline
specification lab (src/models). Nothing here writes to data/processed or touches raw
files. Derived variables (leads, spatial lags, interactions) are built in memory at
analysis time and are NOT stored in the permanent panel.
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*numexpr.*")

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import PROJECT_ROOT, get_logger, load_config  # noqa: E402,F401

PANEL = PROJECT_ROOT / "data" / "processed" / "final_msoa_year_dissertation_panel.parquet"
BASELINE = PROJECT_ROOT / "data" / "processed" / "msoa_baseline_characteristics.parquet"
BOUNDARIES = PROJECT_ROOT / "data" / "raw" / "geography" / "msoa11_boundaries" / "MSOA11_BGC_England.geojson"
EDA_DIR = PROJECT_ROOT / "outputs" / "eda"
MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim" / "eda"

# Variable groups used throughout. The panel stores the treatment as new-build
# completions per 1,000 baseline (2011) households; x is lagged one year.
OUTCOME = "log_median_ppsqm"
TREATMENT = "newbuilds_lag1_per_1000"
MODERATORS = {  # name -> raw baseline column (z-scored columns are built in memory)
    "income": "log_baseline_income",
    "deprivation": "deprivation_moderator_value",
    "social_rent": "social_rent_share_2011",
    "density": "log_population_density_2011",
    "degree": "degree_share_2011",
    "unemployment": "unemployment_rate_2011",
}
PRIMARY_MODERATORS = ["income", "deprivation", "social_rent", "density"]
OPTIONAL_MODERATORS = ["degree", "unemployment"]
Z_COL = {k: f"{k}_z_eda" for k in MODERATORS}


def out_dir(root: Path, *parts: str) -> Path:
    p = root.joinpath(*parts)
    p.mkdir(parents=True, exist_ok=True)
    return p


def savefig(fig, path: Path, dpi: int = 130) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def load_panel(add_derived: bool = True) -> pd.DataFrame:
    """Panel sorted by msoa11cd, year; contiguous years so groupby shifts are valid."""
    df = pd.read_parquet(PANEL).sort_values(["msoa11cd", "year"]).reset_index(drop=True)
    if not add_derived:
        return df
    g = df.groupby("msoa11cd", sort=False)
    # leads/lags of the same-year construction rate, computed in memory only
    for k in (1, 2, 3):
        df[f"nb_lead{k}"] = g["newbuilds_per_1000"].shift(-k)
    df["nb_lag1"] = g["newbuilds_per_1000"].shift(1)
    df["d_log_price"] = g[OUTCOME].diff()                       # price growth t-1 -> t
    df["d_log_price_lag1"] = g["d_log_price"].shift(1)
    df["log_sale_count"] = np.log(df["sale_count"])
    # England-only z-scores of each baseline moderator (MSOA-level, unweighted)
    base = df.drop_duplicates("msoa11cd")
    for k, col in MODERATORS.items():
        df[Z_COL[k]] = (df[col] - base[col].mean()) / base[col].std(ddof=0)
    df["lad_year"] = df["lad23cd_analysis"] + "_" + df["year"].astype(str)
    df["region_year"] = df["region_code"] + "_" + df["year"].astype(str)
    return df


def load_baseline() -> pd.DataFrame:
    return pd.read_parquet(BASELINE)


def msoa_gdf() -> gpd.GeoDataFrame:
    return gpd.read_file(BOUNDARIES).rename(columns={"MSOA11CD": "msoa11cd", "MSOA11NM": "msoa11nm"})


def queen_weights(ids_order: list[str] | None = None):
    """Queen contiguity on MSOA11 boundaries. Islands are attached to their nearest
    neighbour (k=1 on centroids) so every area has at least one neighbour; the number
    attached is returned for the log. Returned weights are binary (untransformed)."""
    import libpysal
    from libpysal.weights import KNN, Queen
    from libpysal.weights.util import attach_islands

    gdf = msoa_gdf().sort_values("msoa11cd").reset_index(drop=True)
    w = Queen.from_dataframe(gdf, ids=gdf["msoa11cd"].tolist(), use_index=False, silence_warnings=True)
    islands = list(w.islands)
    if islands:
        proj = gdf.to_crs(27700)
        wk = KNN.from_dataframe(proj, k=1, ids=gdf["msoa11cd"].tolist())
        w = attach_islands(w, wk)
    return w, islands


def queen_edge_table() -> pd.DataFrame:
    """Directed queen-contiguity edges (focal, neighbour) with edge_type 'queen' for genuine shared-boundary contiguity and
    'island_fallback' for the nearest-neighbour connections that queen_weights() adds for island MSOAs. Uses exactly the same
    construction as queen_weights(), so the union of both types equals its neighbour sets."""
    from libpysal.weights import Queen

    gdf = msoa_gdf().sort_values("msoa11cd").reset_index(drop=True)
    w0 = Queen.from_dataframe(gdf, ids=gdf["msoa11cd"].tolist(), use_index=False, silence_warnings=True)
    genuine = {(a, b) for a, nb in w0.neighbors.items() for b in nb}
    w, _ = queen_weights()
    allpairs = {(a, b) for a, nb in w.neighbors.items() for b in nb}
    if not genuine <= allpairs:
        raise RuntimeError("queen_weights() dropped genuine queen-contiguity edges.")
    rows = [(a, b, "queen" if (a, b) in genuine else "island_fallback") for a, b in sorted(allpairs)]
    return pd.DataFrame(rows, columns=["focal_msoa11cd", "neighbour_msoa11cd", "edge_type"])


def style() -> None:
    plt.rcParams.update({"figure.dpi": 100, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "font.size": 9})
