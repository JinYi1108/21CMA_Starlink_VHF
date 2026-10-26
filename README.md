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
5. `5_flux_density/prepare_oskar_and_flux_inputs.ipynb` — remeasure the two
   events, reconstruct the satellite directions, and prepare the station-beam
   inputs. This step reproduces the event-power and ideal-array-factor tables
   without requiring the large EDA2 catalogue or the OSKAR executable.
6. `5_flux_density/5_flux_density.ipynb` — run the complete Chapter 5 workflow,
   including a fresh execution of the preparation notebook, the OSKAR runner,
   the EDA2 comparison, and the orbital-state re-analysis. The Space-Track
   portion has one documented download checkpoint after Sect. 10.2; follow
   the external-input instructions below before resuming at Sect. 10.3.

The full-survey search notebook is intentionally separate because it requires
the non-public 2.4 TB observation set.

## 4. External Chapter 5 inputs

The EDA2 catalogue and Space-Track GP histories are not committed. Follow the
instructions in `data/external/eda2/README.md` and
`data/external/spacetrack/README.md`. The notebooks generate all subsequent
tables and figures from these inputs.

## Directory map

```text
2_signal_analysis/            event decoding, spectra, optional full search
3_satellite_identification/   payload interpretation and satellite matching
4_orbital_context/            visibility and orbital-evolution analysis
5_flux_density/               power, beam, EDA2 flux density comparsion and orbit phase analysis
data/                         immutable primary inputs and download locations
results/                      generated products
```

