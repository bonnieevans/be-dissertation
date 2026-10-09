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

## Method paragraph (draft for the dissertation)

Accessibility is measured with the Department for Transport's Journey Time Statistics for 2014 (the first year of DfT's current method). These are modelled, not observed, journey times: DfT's routing software calculates the quickest journey from the population-weighted centroid of each 2011 Census Output Area to the nearest destination of eight types (employment centres of 500-4,999 jobs, primary and secondary schools, further education, GPs, hospitals, food stores and town centres), by public transport/walk and by car. Public transport journeys use Traveline, rail (ATOC) and national coach timetables for the morning peak (7-10am) of a Tuesday in the second week of October 2014; car journeys use average link speeds from Trafficmaster satnav data for the preceding twelve months to August 2014 on the Ordnance Survey Integrated Transport Network. Employment destinations use jobs by LSOA from the Business Register and Employment Survey for the year before the calculation year; the town-centre list is fixed at 2004. Times are capped at 120 minutes. LSOA values are aggregated to MSOA11 as means weighted by each service's user population, and the key-services composite is the equal-weighted mean of the eight service times. Because the timetables, road speeds and destination data are all specific to 2014, the measure describes accessibility around 2014; it is treated as a fixed baseline characteristic.

## Can the series be extended to 2012-2023? (as found, 2026-10-09)
- Modelling removes dependence on observed travel behaviour, but not on the year: the inputs (timetables, traffic speeds, job locations, school and GP locations) are all dated.
- Comparable annual tables exist for 2014 onward only (2014 and 2017 downloadable and checked; 2019 is the latest I found). DfT's earlier accessibility statistics (2011-2013) use a different method and DfT states they are not comparable.
- A time-varying version would no longer be a pre-treatment baseline, because new housing can change bus services, jobs and destinations; the interactions would then be identified from changes in a variable the treatment may affect.
- What a later year can do is a persistence check: do MSOAs keep their rank between 2014 and 2017 or 2019?
