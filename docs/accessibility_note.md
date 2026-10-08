# Accessibility variables: definitions and limits (draft)

- **Source and year.** DfT *Journey Time Statistics: access to services 2014* (LSOA11). Modelled (theoretical) travel times, not
  observed journeys; AM peak on a Tuesday in October; public transport timetables 7-10am. Town centres are a fixed 2004 list.
- **Why 2014.** The earliest year with downloadable LSOA tables that I could confirm. 2015 and 2016 reports exist but I did not find
  their data tables; the DfT notes warn that employment destinations are not strictly comparable between 2015 and 2016.
- **Eight services x two modes.** Employment centres of 500-4,999 jobs, primary and secondary schools, further education, GPs,
  hospitals, food stores, town centres; public transport/walk and car. Cycle times are in the DfT tables but not extracted.
- **Composite.** Equal-weighted mean of the eight service times (DfT's key-services average, which DfT publishes only at larger
  geographies); a working-age-weighted LSOA-level version correlates 0.9997 (public transport) / 0.9998 (car) with it.
- **Distance.** Straight-line, from the ONS population-weighted centroid to the nearest 2004 town-centre polygon.
- **Limits to keep in mind.** Higher = less accessible. Times capped at 120. Strong overlap with population density (r about -0.8).
  The Isles of Scilly is an extreme outlier. 1,232 town-centre polygons were found against DfT's stated 1,211 (not reconciled).
  The measures are fixed at baseline, so MSOA fixed effects absorb them; only interactions with the treatment are identified.
