#!/bin/bash
#SBATCH --job-name=test
#SBATCH --time=00:30:00
#SBATCH --mem=180GB
#SBATCH -p serc
#SBATCH --gpus 1
#SBATCH -o ./../sbatch_outputs/out_jcp_test_grain.out
#SBATCH -e ./../sbatch_outputs/err_jcp_test_grain.err

export PYTHONUNBUFFERED=1

# ml py-scipy/1.6.3_py39
# ml py-pytorch/1.11.0_py39
ml py-h5py/3.7.0_py39
ml py-numpy/1.24.2_py39
# ml py-scipy/1.12.0_py312
# ml py-pytorch/2.2.1_py312
# ml py-h5py/3.10.0_py312
# ml py-numpy/1.26.3_py312
start_time=$(date +%s)
date +"Job started at: %Y-%m-%d %H:%M:%S"

python3 test_grain_kde.py

end_time=$(date +%s)
date +"Job ended at: %Y-%m-%d %H:%M:%S"

duration=$((end_time - start_time))
hours=$((duration / 3600))
minutes=$(( (duration % 3600) / 60 ))
seconds=$((duration % 60))

echo "Job took $hours hours, $minutes minutes, and $seconds seconds to complete."
