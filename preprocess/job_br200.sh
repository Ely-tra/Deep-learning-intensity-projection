F TC Preprocessing Job Script
#
# This script runs step1.py to crop raw WRF outputs around TC tracks,
# then runs step2.py to convert the cropped NetCDF files into NumPy arrays.
# ==============================================================================

#SBATCH -N 1
#SBATCH -t 24:00:00
#SBATCH -J WRF-TC-Prep
#SBATCH -p cpu
#SBATCH -A r00043
#SBATCH --mem=64G

module load python/3.10.10
cd /N/slate/kmluong/PROJECT2/
set -x

# ------------------------------------------------------------------------------
# File/Directory for input data and outputs
# ------------------------------------------------------------------------------
track_file='/N/project/hurricane-deep-learning/data/cmip6/baseline_track.txt'    # TC track ASCII
raw_data_dir='/N/project/hurricane-deep-learning/data/cmip6/baseline/'         # Raw WRF outputs
level1_dir='/N/slate/kmluong/PROJECT2/level_1_data'                            # Cropped .nc output
level2_dir='/N/slate/kmluong/PROJECT2/level_2_data'                            # Final .npy output

# ------------------------------------------------------------------------------
# Parameters
# ------------------------------------------------------------------------------
imsize_x=100           # Crop width (grid points)
imsize_y=100           # Crop height (grid points)
frames=5               # Number of consecutive frames for step2.py
var_levels=(           # Variables/levels for step2.py
  U10m V10m SST LANDMASK
  U28 V28 U05 V05
  T23 QVAPOR10 PHB10
)
prefix="wrf_tropical_cyclone_track_${frames}_dataset"

# ------------------------------------------------------------------------------
# Step toggles: set to 1 to run, 0 to skip
# ------------------------------------------------------------------------------
run_step1=1
run_step2=1

# ------------------------------------------------------------------------------
# 1) Crop raw WRF outputs around TC tracks (step1.py)
# ------------------------------------------------------------------------------
if [ "$run_step1" -eq 1 ]; then
    python step1.py \
        --track_file "$track_file" \
        --data_dir   "$raw_data_dir" \
        --workdir    "$level1_dir" \
        --imsize_x   $imsize_x \
        --imsize_y   $imsize_y
fi

# ------------------------------------------------------------------------------
# 2) Process cropped .nc files into NumPy arrays (step2.py)
# ------------------------------------------------------------------------------
if [ "$run_step2" -eq 1 ]; then
    python step2.py \
        --indir       "$level1_dir" \
        --outdir      "$level2_dir" \
        --frames      $frames \
        --var_levels  "${var_levels[@]}" \
        --prefix      "$prefix"
fi

echo "All requested steps completed."

