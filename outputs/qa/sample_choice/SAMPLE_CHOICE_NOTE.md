# Sample window: new-build coverage by year

Files in this folder: `newbuild_coverage_by_year_vs_official.csv` (panel and EPC counts vs MHCLG Live Table 120, calendar-year approximation
= 0.25 x FY ending that year + 0.75 x FY starting that year), `epc_newdwelling_uprn_loss_by_year.csv` (missing-UPRN share by lodgement year),
`baseline_coefficient_by_sample_window.csv` (M3 and M6 baseline coefficient by window), `official_new_build_completions_by_fiscal_year.csv`.

- Panel / official ratio: 2012 1.02, 2013 0.94, 2014 0.94, 2015 1.12, 2016 1.09, 2017 1.06, 2018 1.07, 2019 1.04, 2020 0.92, 2021 0.96, 2022 0.88, 2023 0.38.
- Missing-UPRN share of new-dwelling EPCs: about 2% to 2020, 4.3% 2021, 11.2% 2022, 51.5% 2023.
- Treatment is the previous year's construction, so the outcome years 2013-2023 use construction in 2012-2022; the 2023 count itself enters only same-year and lead specifications.
- Caveat: the official series is financial-year and net of a different definition (note 2 in the table: may not match quarterly housebuilding); the ratio is a coverage indicator, not an exact match. Panel/official above 1 in 2015-2016 is not explained.
