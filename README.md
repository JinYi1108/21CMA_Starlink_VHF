# 21CMA Starlink VHF analysis

This repository contains the processing notebooks and supporting scripts for
the analysis of two Starlink VHF TT&C packets detected with the 21 CentiMeter
Array (21CMA) at 137.055 MHz on 2025-04-15 (UTC). The repository is organised by the
corresponding paper sections; generated products are written to `results/`
and are not used as hidden inputs to later stages.

## Reproduction scope

The public **Zenodo** record contains the two 8.192 GB raw-voltage files that hold
the detected packets. With those files, this repository reproduces the event
demodulation, dynamic spectra, packet interpretation, satellite
identification, orbital-context analysis, and flux-density estimate.

The paper's blind search covered five observing windows totalling about
2,500 s and requires approximately 2.4 TB of raw voltages. Those complete
observations are not part of the Zenodo record.

## 1. Installation

```bash
conda env create -f environment.yml
conda activate 21cma-starlink-vhf
python -m ipykernel install --user --name 21cma-starlink-vhf
```

The environment installs the tagged `LFdemod` Starlink-VHF decoder and its
`lora-phy==0.3.0` backend. The optional OSKAR station-beam calculation also
requires an OSKAR 2.13 executable available as `oskar_sim_beam_pattern` (or
through the `OSKAR_BIN` environment variable).

## 2. Download the two raw files

Download both files from <https://doi.org/10.5281/zenodo.23309576> into
`data/raw/`:

```text
data/raw/20250415-2200-0.dat
data/raw/20250415-2200-12.dat
```

The filenames, sizes, MD5 checksums, sample format, and original acquisition
UTC are recorded in `data/observation/event_files.csv`. Run this optional
integrity check from the repository root:

```bash
python -c "from project_config import verify_raw_data; verify_raw_data(calculate_md5=True)"
```

## 3. Run order

Run notebooks from the repository root in this order:

1. `2_signal_analysis/2.2_demodulation.ipynb` — decode both Zenodo files with
   LFdemod and write `results/2_demodulation/`.
2. `2_signal_analysis/2.1_waterfall.ipynb` — construct the multi-scale dynamic
   spectra and derive the E08 station centre.
3. `3_satellite_identification/3_satellite_identification.ipynb` — parse the
   payload state vector, perform the active-catalogue blind comparison, and
   test the differential Doppler shift and SupGP match.
4. `4_orbital_context/4_orbital_context.ipynb` — evaluate cadence-predicted
   visibility and the orbital evolution of STARLINK-32838.
5. `5_flux_density_&_EDA2_orbit/prepare_oskar_and_flux_inputs.ipynb` — remeasure the two
   events, reconstruct the satellite directions, and prepare the station-beam
   inputs. This step reproduces the event-power and ideal-array-factor tables
   without requiring the large EDA2 catalogue or the OSKAR executable.
6. `5_flux_density_&_EDA2_orbit/5_flux_density_&_EDA2_orbit.ipynb` — run the complete flux-density and EDA2-orbit workflow,
   including a fresh execution of the preparation notebook, the OSKAR runner,
   the EDA2 comparison, and the orbital-state re-analysis.

The full-survey search notebook is intentionally separate because it requires
the non-public 2.4 TB observation set.

## 4. Included third-party inputs

The EDA2 public catalogue is included as a losslessly compressed CSV, and the
analysis-ready historical Space-Track GP/TLE tables used in the paper are also
included. Their provenance and checks are documented in
`data/external/eda2/README.md` and `data/external/spacetrack/README.md`.
No Space-Track login or EDA2 catalogue download is required to reproduce the
paper analysis.

The most important paper figures, decoded packets, and summary tables are
collected in `paper_results/` for direct inspection. These are convenience
copies; the notebooks continue to write freshly generated products to
`results/`.

## Directory map

```text
2_signal_analysis/            event decoding, spectra, optional full search
3_satellite_identification/   payload interpretation and satellite matching
4_orbital_context/            visibility and orbital-evolution analysis
5_flux_density_&_EDA2_orbit/  power, beam, EDA2 flux-density comparison and orbit-phase analysis
data/                         immutable primary inputs and download locations
paper_results/                curated paper figures and principal results
results/                      generated products
```
