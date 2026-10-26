# Orbital inputs

These files are immutable inputs used by the paper workflow:

- `Active_2025-04-15.tle`: active-satellite catalogue downloaded from
  CelesTrak on 2025-04-15 for the blind comparison in Chapter 3.
- `STARLINK-32838_2025-04-10_2025-04-20.tle`: pre-event TLE history used for
  the differential-Doppler comparison.
- `STARLINK-32838_SpaceX-E_SupGP.csv`: historical SpaceX-E SupGP records used
  to validate the recovered state vector and reconstruct the event geometry.
- `STARLINK-32838_full_history.tle`: historical TLE records used to generate
  the target's long-term altitude history.
- `STARLINK_Group11-6_full_history.tle`: historical TLE records used to
  compare STARLINK-32838 with the other members of its launch group.

The notebooks write all derived tables and figures to `results/`; none of
those generated products is required as an input here.
