# IMD 2015 revision: task specification (as given 2026-10-02)

TASK: Revise IMD 2015 processing (Section E "Deprivation" of the data processing log).
Do not change anything outside the IMD steps and their downstream uses listed below.

CONTEXT
- The overall IMD includes the Barriers to Housing and Services domain, which contains a
  housing affordability indicator (house prices relative to incomes). The outcome is house
  prices, so the overall IMD must not be the primary moderator.
- Averaging LSOA ranks is not valid. Remove all rank-averaging.
- LSOA11 nests exactly in MSOA11. Keep using the existing LSOA11→MSOA11 lookup with no
  crosswalk.

1. REMOVE
   - Drop imd2015_rank_percentile_pw from all outputs, the variable dictionary, the Excel
     workbook, methodology notes and tests.
   - Stop basing deprivation_z, deprivation_quartile and high_deprivation_q4 on
     imd2015_bottom20_popshare.

2. LOAD DOMAIN DATA (IMD 2015 File 7, already downloaded)
   - Print the header first and map the real column names in config.yaml. Do not guess them.
   - Keep these LSOA-level SCORES: overall IMD; Income; Employment; Education, Skills and
     Training; Health Deprivation and Disability; Crime; Barriers to Housing and Services;
     Living Environment.
   - Keep these denominators: total population (mid-2012) and working-age population
     (the denominator used for the Employment domain).
   - Assert that there are 32,844 England LSOA11s, that each maps to exactly one MSOA11, and
     that the 6,791 MSOA11s are fully covered.

3. LSOA-LEVEL RECOMBINED INDICES (before aggregating)
   - Use the method from the IMD 2015 Technical Report (Section 3). Check the exact formula
     against the report before implementing.
     a) Rank each domain score across all 32,844 LSOAs and scale to R in (0,1], with 1 =
        most deprived.
     b) Apply the exponential transform X = -23 * ln(1 - R * (1 - exp(-100/23))).
     c) Combine using the published weights: Income .225, Employment .225, Education .135,
        Health .135, Crime .093, Barriers .093, Living Env .093. When domains are excluded,
        rescale the remaining weights to sum to 1.
   - VALIDATION (hard stop): recombining all 7 domains must reproduce the published IMD
     score. Require rank correlation > 0.999 and report the maximum absolute difference.
     Add this as a unit test.
   - Build three variants:
       imd_ex_housing        = 6 domains (drop Barriers)               <- proposed primary
       imd_ex_housing_living = 5 domains (drop Barriers and Living Env) <- sensitivity
       imd_ex_income_housing = 5 domains (drop Income and Barriers)     <- use if SAIE is
                                                                           kept as the
                                                                           income moderator
   - Put weights and domain lists in config.yaml.

4. AGGREGATE TO MSOA11 (population-weighted means of LSOA values, never ranks or deciles)
   - Income domain rate: weight by total population, which gives the exact MSOA rate.
     Name it imd2015_income_rate_msoa. Higher = more deprived.
   - Employment domain rate: weight by working-age population.
   - All other domain scores, the three recombined indices and the overall IMD score: weight
     by total population.
   - Rename the existing overall measure to imd2015_overall_score_pw and label it
     "robustness only, contains housing affordability".
   - QA: the sum of LSOA weights within each MSOA equals the MSOA population. No missing
     values.

5. BOTTOM-20 SHARE
   - Recompute the share using national deciles of imd_ex_housing at LSOA level
     (imd_ex_housing_bottom20_popshare). Keep it as a secondary descriptive variable only,
     because about half of MSOAs have zero by construction.
   - Move the old overall-IMD bottom-20 share to a legacy column, flagged as not for
     estimation.

6. PRIMARY MODERATOR AND INCOME CHOICE (config switches)
   - deprivation_moderator: one of {imd_ex_housing, imd_ex_housing_living,
     imd_ex_income_housing, imd_income_rate}. Default: imd_ex_housing.
   - income_moderator: one of {saie, imd_income_rate}. Default: saie.
   - Rebuild deprivation_z (England-only, unweighted), deprivation_quartile and
     high_deprivation_q4 from the chosen deprivation moderator.
   - Warn and stop if the same underlying measure is selected for both switches.
   - Note the sign convention: SAIE higher = richer; IMD measures higher = more deprived.

7. DIAGNOSTICS
   - Pearson and Spearman correlations between: log SAIE income, imd2015_income_rate_msoa,
     imd_ex_housing, imd_ex_income_housing, imd2015_overall_score_pw, and the Barriers
     domain score. Also a scatter of SAIE against the income rate.
   - Rerun the z-score correlation matrix and VIFs for the joint model under (a) SAIE +
     imd_ex_housing and (b) SAIE + imd_ex_income_housing.
   - Save the outputs to outputs/qa/imd_revision/.

8. DOWNSTREAM UPDATES
   - Regenerate final_msoa_year_dissertation_panel and msoa_baseline_characteristics
     (expect the column count to change; log it).
   - Regenerate Tables 1, 3 and 4 (construction by deprivation quartile), the baseline
     deprivation map, the Excel workbook (variable dictionary, methodology notes), and
     Sections E and H of the processing log.
   - Existing hard stops must still pass: baseline values constant within MSOA across years,
     no duplicate keys, England only.

9. REPORT BACK
   - The reproduction test result.
   - The correlation table.
   - The VIFs.
   - A before/after comparison of quartile membership: the share of MSOAs that change
     deprivation quartile when moving from overall IMD to imd_ex_housing.

## Outcome notes (added after implementation)
- Item 3a: the published domain ranks are used by default instead of re-ranking the rounded
  published scores (flagged deviation, switchable via `imd2015.rank_source`; evidence in
  `outputs/qa/imd_revision/IMD_REVISION_REPORT.md`).
- Item 5: "about half" is 61.7% of MSOAs with a zero share in this build.
- Results: `outputs/qa/imd_revision/IMD_REVISION_REPORT.md`, workbook sheet `imd_revision`,
  `DATA_PROCESSING_LOG.md` section I.
