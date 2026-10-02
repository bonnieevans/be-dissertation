# Instrumenting new-build construction at MSOA scale: feasibility note

Status: working note. The literature list below was written from memory because web search was unavailable when this was
prepared; **check every citation and what each paper actually does before relying on it.** The mechanical findings
come from `src/models/m07_iv_feasibility.py` (outputs/models/tables/m07_*.csv).

## What the data say mechanically (M07)
Share of each synthetic variable's variance left after the fixed effects (balanced panel, main sample):

| Variable type | After MSOA + year FE (M3) | After MSOA + LAD x year FE (M6) |
|---|---|---|
| LAD-level, constant over time | 0 | 0 |
| LAD-level, varying by year | 0.854 | 0 (R2 on the LAD x year effects = 1) |
| LAD-level constant x national year shock | 0.843 | 0 (R2 = 1) |
| MSOA-level constant (stand-in: baseline density) x national year shock | 0.857 | 0.412 |
| MSOA-level, varying by year | 0.871 | 0.832 |
| The treatment itself (lag-1 new-builds per 1,000) | 0.468 | 0.445 |

Reading: with LAD x year effects, anything defined at LAD or LAD-year level is absorbed; the same holds for a LAD-year
funding indicator. A variable must vary across MSOAs within a LAD in a given year to be used in M6. Constant MSOA
characteristics are absorbed by MSOA effects unless they are interacted with something that varies over time.

## Reading list (verify)
- Hilber & Vermeulen (2016), *Economic Journal*, on the effect of planning restrictiveness on house prices in England
  (local-authority panel; restrictiveness measured from planning decisions, instrumented).
- Saiz (2010), *Quarterly Journal of Economics*, on geographic determinants of housing supply (topographical constraints
  as a supply-elasticity measure; US metropolitan areas).
- Goldsmith-Pinkham, Sorkin & Swift (2020), *American Economic Review*, on shift-share (Bartik) instruments, and
  Borusyak, Hull & Jaravel (2022), *Review of Economic Studies*, on the same: the identification conditions for
  instruments built as a fixed local characteristic times a common shock.
- Stock & Yogo (2005) and Andrews, Stock & Sun (2019) on weak instruments.
- Cameron & Miller (2015), *Journal of Human Resources*, on cluster-robust inference (clustering with IV).

## Questions to settle when reading
1. At what geography is the planning-restrictiveness measure defined (local authority), and how much does it vary over time
   within an authority? If it is LAD-level it is absorbed in M6 (table above).
2. Is there any MSOA-level source of variation in supply that is plausibly unrelated to local price shocks (for example
   fixed physical or land-use constraints interacted with a national construction cycle)? That structure is the
   fourth row of the table; the exclusion restriction and the shift-share conditions in the references are what has to be argued.
3. Does the first stage have enough strength once the fixed effects are removed (the treatment keeps 45-47% of its
   variance after M3 / M6, so there is variation to explain)?
4. Where this belongs in the dissertation (limitations, future research, or an extension) is your decision.
