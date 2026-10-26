# Space-Track inputs

The Chapter 5 orbital-state re-analysis downloads historical GP records from
Space-Track. Do not commit credentials or downloaded responses. First run
`5_flux_density/5_flux_density.ipynb` through Sect. 10.2; this generates the
emission-segment and satellite-date input tables. Then set
`SPACETRACK_USER` and `SPACETRACK_PASSWORD` in the shell and run, from the
repository root:

```bash
python 5_flux_density/download_spacetrack_all_137p05_emitters.py
python 5_flux_density/download_spacetrack_full_history_137p05_emitters.py \
  --end-date 2026-10-18
```

The fixed end date reproduces the historical coverage used for the paper;
omit that option only when intentionally extending the analysis. The scripts
write the authenticated downloads under this directory. Resume the notebook
at Sect. 10.3 to audit the files and generate the orbital-state analysis. The
downloaded responses are Git-ignored.

Space-Track data are subject to the provider's terms of use.
