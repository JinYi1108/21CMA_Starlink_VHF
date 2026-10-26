# Space-Track inputs

The analysis-ready historical GP/TLE tables used for the paper are included in
this directory. Consequently, reproducing the orbital-phase analysis does not
require a Space-Track account or a new download.

`all_137p05_emitters/` contains the date-centred records used for the coverage
audit and local trend analysis. `full_137p05_emitters/` contains the
deduplicated long-term histories used for the final orbital-state
classification and four-panel lifecycle figures. Raw authenticated JSON
responses and a redundant combined full-history table are intentionally not
included; neither is read by the final notebook.

The download scripts are retained only for users who intentionally want to
refresh or extend the data. After generating the emission-segment and
satellite-date tables in
`5_flux_density_&_EDA2_orbit/5_flux_density_&_EDA2_orbit.ipynb`, set
`SPACETRACK_USER` and `SPACETRACK_PASSWORD` and run:

```bash
python '5_flux_density_&_EDA2_orbit/download_spacetrack_all_137p05_emitters.py'
python '5_flux_density_&_EDA2_orbit/download_spacetrack_full_history_137p05_emitters.py' \
  --end-date 2026-10-18
```

The fixed end date reproduces the historical coverage used for the paper;
omit that option only when intentionally extending the analysis. The scripts
write refreshed records under this directory.

Space-Track data are subject to the provider's terms of use.
