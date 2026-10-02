# Why MSOA11? Draft justification (for you to edit)

Status: draft prepared from the data in this project. The arguments in the last section are yours to make.

## Facts from this project
- **Unit:** 6,791 England MSOA11s; median 3,160 households (5th-95th percentile 2,326-4,500) and median population
  7,635 (5,660-10,682) at the 2011 Census. 296 LADs, 22.9 MSOAs per LAD on average (median 18, range 1-132).
- **Transactions:** 8.14 million sales in the price sample, a mean of 99.9 sales per MSOA-year. Only 0.1% of
  MSOA-years have fewer than 10 sales and 2.3% fewer than 30.
- **Data availability by geography:**

| Moderator | Native geography | Available below MSOA? |
|---|---|---|
| Baseline income (ONS small-area income estimates, FYE2012) | MSOA11 | No |
| Deprivation (IMD 2015) | LSOA11 | Yes (LSOA) |
| Social-rented share, density, degree share, unemployment (Census 2011) | OA / LSOA / MSOA | Yes |
| New-build counts (EPC + UPRN) | Address | Yes |
| Prices (Land Registry PPD, linked) | Address | Yes |

- **Smaller units:** there are 32,844 LSOA11s (4.8 per MSOA). The same sales would give a mean of about 20.7
  sales per LSOA-year, so LSOA medians would be noisier; and income is not available at LSOA level.
- **Larger units:** LADs (296) pool about 23 MSOAs each and would remove most of the within-LAD variation that M6
  (MSOA + LAD x year effects) uses.
- **Direct evidence in the data:** spatial autocorrelation of price levels is very high at MSOA level (Moran's I about
  0.9; outputs/eda/tables/spatial/06_morans_I.csv), and residual correlation among neighbouring MSOAs after year
  effects is large (outputs/models/tables/m01_residual_cross_sectional_dependence.csv), which bears on whether an MSOA
  is smaller than the area that shares a housing market.

## Points the justification needs to address (your argument)
1. **Why MSOA is large enough:** adequate sales per cell, a stable denominator (baseline households), and the
   availability of all six moderators at one geography, including income.
2. **Why it may be too large for local effects:** new development may affect prices within a few hundred metres, and an
   MSOA mixes areas near to and far from a development; the estimate then averages over those.
3. **Why it may be too small for housing-market effects:** effects that operate across a travel-to-work or
   housing-market area spill into neighbouring MSOAs (see M05 for the neighbour-variable check).
4. **Alternatives:** LSOA (possible for every moderator except income, thinner sales per cell); LAD or housing-market
   area (more aggregated; fewer than 300 units); address-level transaction models (the parquet file
   `transactions_regression_ready` carries the MSOA-level moderators); a distance-band approach would need new-build
   coordinates (UPRN coordinates), which are not in this build.
5. **Modifiable areal unit problem:** results can change with the boundaries chosen; the LSOA robustness run is the
   direct way to show how much.
