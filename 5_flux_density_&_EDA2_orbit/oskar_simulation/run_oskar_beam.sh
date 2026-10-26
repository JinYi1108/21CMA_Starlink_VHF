#!/usr/bin/env bash
# Run the complete CPU-only E08 OSKAR validation at 137.055 MHz.
#
# Optional environment variable:
#   IMAGE_SIZE=2048  Increase the full-horizon FITS image resolution.
#                    The default is 1024; exact event gains do not depend on it.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-python}"
oskar_bin="${OSKAR_BIN:-oskar_sim_beam_pattern}"
image_size="${IMAGE_SIZE:-1024}"
log_dir="${script_dir}/results/logs"
mkdir -p "${log_dir}"

if ! command -v "${oskar_bin}" >/dev/null 2>&1 && [[ ! -x "${oskar_bin}" ]]; then
    echo "OSKAR executable not found: ${oskar_bin}. Set OSKAR_BIN if needed." >&2
    exit 1
fi

echo "[1/4] Preparing the E08 telescope model and settings (image ${image_size} x ${image_size})"
"${python_bin}" "${script_dir}/prepare_oskar_run.py" --image-size "${image_size}"

echo "[2/4] Running the full-horizon OSKAR beam simulation"
(cd "${log_dir}" && "${oskar_bin}" "${script_dir}/config/e08_horizon_beam_137055.ini") \
    2>&1 | tee "${log_dir}/horizon_beam.log"

echo "[3/4] Evaluating OSKAR power at every sample of the four event tracks"
(cd "${log_dir}" && "${oskar_bin}" "${script_dir}/config/e08_four_event_samples_137055.ini") \
    2>&1 | tee "${log_dir}/event_samples.log"

echo "[4/4] Creating figures, CSV tables, and the result manifest"
"${python_bin}" "${script_dir}/postprocess_oskar_results.py" \
    2>&1 | tee "${log_dir}/postprocess.log"

echo
echo "Finished. Results are under: ${script_dir}/results"
