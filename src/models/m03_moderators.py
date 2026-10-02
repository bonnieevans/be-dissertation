"""M03 - adding the socioeconomic moderators.

Specification: log price = b0*x + sum_k b_k*(x * z_k) + fixed effects, where x is the lagged new-build rate and
z_k are England-wide z-scores of the baseline moderators. Moderator main effects are absorbed by the MSOA
fixed effects. Interactions are built in memory here and are not stored in any dataset.

The order and the combinations are set by theory and by the structure found in the EDA (correlations, PCA, VIF),
NOT by the significance of the interaction terms, to limit specification searching. Every specification run is
written to the registry and the interaction p-values are also reported with a Holm adjustment within each
family of tests.

Two ways of handling moderator-specific time shocks are compared:
  (A) interactions only (moderator main effects absorbed by MSOA FE; time effects common to all MSOAs, or common
      within a LAD-year in M6);
  (B) interactions plus moderator x year controls (each moderator allowed its own year-specific effect on price).
Both FE structures are run: M3 (MSOA + year) and M6 (MSOA + LAD x year).
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_common import *  # noqa: F401,F403
import model_common as MC
from scipy import stats

log = get_logger("m03_moderators")
style()
T = out_dir(MODEL_DIR, "tables")
F = out_dir(MODEL_DIR, "figures")
Y, X = OUTCOME, TREATMENT
FE = {"M3": "| msoa11cd + year", "M6": "| msoa11cd + lad_year"}
PCA_SCORES = MC.EC.EDA_DIR / "tables" / "pca" / "04_scores_four_primary_moderators.csv"


def holm(p: np.ndarray) -> np.ndarray:
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


def within(d: pd.DataFrame, cols: list[str], fe: str) -> pd.DataFrame:
    """Within transformation (exact for balanced panels: MSOA, then year or LAD x year means)."""
    V = d[cols]
    V = V - V.groupby(d.msoa11cd.values).transform("mean")
    key = d.year.values if fe == "M3" else d.lad_year.values
    return V - V.groupby(key).transform("mean")


def vif_table(d: pd.DataFrame, regs: list[str], fe: str) -> pd.DataFrame:
    W = within(d, regs, fe)
    R = np.linalg.inv(np.corrcoef(W.values.T)) if len(regs) > 1 else np.array([[1.0]])
    return pd.DataFrame({"regressor": regs, "VIF": np.diag(R)})


def marginal_effects(f, mod_terms: dict[str, str], at=(-1, 0, 1)) -> pd.DataFrame:
    """Effect of x on log price at z = -1, 0, +1 SD of one moderator (others at their mean), delta-method SE."""
    names = list(f._coefnames)
    b, V = np.asarray(f._beta_hat), np.asarray(f._vcov)
    rows = []
    for m, term in mod_terms.items():
        for a in at:
            g = np.zeros(len(names))
            g[names.index(X)] = 1
            g[names.index(term)] = a
            est, se = float(g @ b), float(np.sqrt(g @ V @ g))
            rows.append({"moderator": m, "z": a, "effect": est, "se": se, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se})
    return pd.DataFrame(rows)


def main() -> int:
    df = load_panel()
    d = estimation_sample(df)
    pcs = pd.read_csv(PCA_SCORES)
    for c in ("PC1", "PC2"):
        pcs[c] = (pcs[c] - pcs[c].mean()) / pcs[c].std(ddof=0)
    d = d.merge(pcs.rename(columns={"PC1": "pc1_z_eda", "PC2": "pc2_z_eda"})[["msoa11cd", "pc1_z_eda", "pc2_z_eda"]], on="msoa11cd", how="left")
    zcol = {**Z_COL, "pc1": "pc1_z_eda", "pc2": "pc2_z_eda"}
    for m, c in zcol.items():
        d[f"x_{m}"] = d[X] * d[c]
    x_sd = d[X].std()

    # Specification sets (pre-specified; see module docstring)
    P4 = PRIMARY_MODERATORS
    sets: dict[str, list[str]] = {}
    for k in (1, 2, 3, 4):
        for c in itertools.combinations(P4, k):
            sets["+".join(c)] = list(c)
    for o in OPTIONAL_MODERATORS:
        sets[o] = [o]
        sets["+".join(P4) + "+" + o] = P4 + [o]
    sets["all_six"] = P4 + OPTIONAL_MODERATORS
    sets["pc1"] = ["pc1"]
    sets["pc1+pc2"] = ["pc1", "pc2"]
    pd.DataFrame({"set": list(sets), "moderators": ["+".join(v) for v in sets.values()]}).to_csv(T / "m03_specification_sets.csv", index=False)

    # ---- 1. Run every set under both FE structures, with and without moderator x year controls ---------------------------
    # WHAT: for each set of moderators, estimate the model with the interactions and record the coefficient on x, each
    #       interaction, the joint Wald test of all interactions (LAD-clustered), and the within-R2.
    # LOOK FOR: the stability of the coefficient on x and of each interaction as other moderators are added; the joint
    #           test against the individual t-tests; how results differ between (A) and (B) and between M3 and M6; the
    #           Holm-adjusted p-values (adjusted for the number of interaction tests in the same family).
    # UNITS:    the coefficient on x is the effect of one additional new-build per 1,000 households when every moderator
    #           z-score is zero (the England-wide average MSOA), in log points (x 100 ~ %). The coefficient on x*z_k is how
    #           much that effect changes (in log points per unit of x) for an MSOA one England-wide standard deviation
    #           higher on moderator k, holding the other included moderators fixed; direction of z_k: income higher = richer,
    #           deprivation higher = more deprived, social_rent and density higher = larger, degree higher = more graduates,
    #           unemployment higher = more unemployed.
    rows, joint, marg = [], [], []
    fits = {}
    for fe_id, fe in FE.items():
        for ctl in (False, True):
            for sname, mods in sets.items():
                rhs = " + ".join([X] + [f"x_{m}" for m in mods])
                if ctl:
                    rhs += " + " + " + ".join(f"i(year, {zcol[m]}, ref=2016)" for m in mods)
                sid = f"{fe_id}_{'zyear' if ctl else 'plain'}_{sname}"
                f = fit(f"{Y} ~ {rhs} {fe}", d, sid, note=f"moderators: {'+'.join(mods)}; moderator x year controls: {ctl}")
                fits[sid] = f
                terms = [X] + [f"x_{m}" for m in mods]
                t = coef_table(f, sid, sname, terms, x_sd).assign(fe=fe_id, year_controls=ctl, set=sname, n_moderators=len(mods))
                rows.append(t)
                wj = wald(f, [f"x_{m}" for m in mods])
                joint.append({"spec_id": sid, "fe": fe_id, "year_controls": ctl, "set": sname, "n_moderators": len(mods), **wj,
                              "r2_within": f._r2_within, "n_obs": f._N})
                if len(mods) == 1 or sname in ("income+deprivation+social_rent+density", "all_six"):
                    mt = marginal_effects(f, {m: f"x_{m}" for m in mods}).assign(spec_id=sid, fe=fe_id, year_controls=ctl, set=sname)
                    marg.append(mt)
    res = pd.concat(rows, ignore_index=True)
    # Holm adjustment within family = (fe, year_controls, number of moderators in the set), interaction terms only
    res["p_holm_within_family"] = np.nan
    inter = res.term != X
    for _, idx in res[inter].groupby(["fe", "year_controls", "n_moderators"]).groups.items():
        res.loc[idx, "p_holm_within_family"] = holm(res.loc[idx, "p"].values)
    res["term_is_interaction"] = inter
    res["interaction_change_in_effect_per_1sd_moderator_pp"] = np.where(inter, 100 * res["coef"], np.nan)
    res.to_csv(T / "m03_moderator_models_coefficients.csv", index=False)
    pd.DataFrame(joint).to_csv(T / "m03_moderator_models_joint_tests.csv", index=False)
    pd.concat(marg).to_csv(T / "m03_marginal_effects.csv", index=False)
    log.info(f"{len(fits)} moderator specifications estimated.")

    # ---- 2. One interaction at a time vs all together ------------------------------------------------------------------
    # WHAT: side-by-side comparison of each moderator's interaction coefficient when it is the only interaction in
    #       the model and when all four primary (and all six) moderators are included.
    # LOOK FOR: changes in sign, size or standard error from the single to the joint model; a coefficient that is
    #           clear alone but not jointly is a sign that it was picking up a correlated moderator.
    comp = []
    for fe_id in FE:
        for ctl in (False, True):
            tag = "zyear" if ctl else "plain"
            for m in P4 + OPTIONAL_MODERATORS:
                row = {"fe": fe_id, "year_controls": ctl, "moderator": m}
                for lab, sname in (("alone", m), ("with_four_primary", "+".join(P4)), ("with_all_six", "all_six")):
                    r = res[(res.spec_id == f"{fe_id}_{tag}_{sname}") & (res.term == f"x_{m}")]
                    if len(r):
                        row[f"coef_{lab}"], row[f"se_{lab}"], row[f"p_{lab}"] = r.coef.iloc[0], r.se.iloc[0], r.p.iloc[0]
                comp.append(row)
    pd.DataFrame(comp).to_csv(T / "m03_alone_vs_joint.csv", index=False)

    # ---- 3. Collinearity of the interaction terms ----------------------------------------------------------------------------
    # WHAT: VIFs and correlations of the regressors (x and the x*z terms) after the within transformation implied by each FE
    #       structure, for the four-moderator, six-moderator and PCA sets; and the correlation matrix of the
    #       interaction terms.
    # LOOK FOR: VIFs above about 5-10 and pairwise correlations of the within-transformed interactions near 0.8 or above;
    #           the change when the FE structure moves from M3 to M6; whether the PCA components reduce the VIFs.
    vrows, crows = [], []
    for fe_id in FE:
        for sname in ("+".join(P4), "all_six", "pc1+pc2", "income+deprivation"):
            mods = sets[sname]
            regs = [X] + [f"x_{m}" for m in mods]
            v = vif_table(d, regs, fe_id).assign(fe=fe_id, set=sname)
            vrows.append(v)
            C = within(d, regs, fe_id).corr()
            crows.append(C.assign(fe=fe_id, set=sname).reset_index().rename(columns={"index": "regressor"}))
    pd.concat(vrows).to_csv(T / "m03_interaction_vif.csv", index=False)
    pd.concat(crows).to_csv(T / "m03_interaction_correlations.csv", index=False)
    log.info("VIFs:\n" + pd.concat(vrows).round(2).to_string())

    # ---- 4. Marginal-effect figure ---------------------------------------------------------------------------------------------------
    # WHAT: the effect of x on log price at z = -1, 0, +1 SD of each moderator when it enters alone, for M3 and M6.
    # LOOK FOR: the slope of the line (the interaction) against the width of the confidence intervals.
    mg = pd.concat(marg)
    fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharey="row")
    for i, fe_id in enumerate(FE):
        for j, m in enumerate(P4):
            ax = axes[i, j]
            g = mg[(mg.fe == fe_id) & (~mg.year_controls) & (mg.set == m) & (mg.moderator == m)]
            ax.errorbar(g.z, g.effect, yerr=1.96 * g.se, fmt="o-", capsize=3)
            ax.axhline(0, color="k", lw=.6)
            ax.set_title(f"{fe_id}: {m}")
            ax.set_xlabel("moderator z-score")
    axes[0, 0].set_ylabel("effect of x on log price")
    axes[1, 0].set_ylabel("effect of x on log price")
    savefig(fig, F / "m03_marginal_effects_single_moderators.png")
    MC.save_registry("m03")
    log.info("M03 complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
