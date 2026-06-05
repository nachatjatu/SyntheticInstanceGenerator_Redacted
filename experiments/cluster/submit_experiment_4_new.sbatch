#!/bin/bash
#SBATCH --job-name=exp4new
#SBATCH --array=0-195%20
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --output=logs/slurm/exp_4/out/%x_%A_%a.out
#SBATCH --error=logs/slurm/exp_4/err/%x_%A_%a.err

# Load required modules
ml gurobi/11.0.2
ml python/3.12

# cd to repo root (wherever sbatch was submitted from)
cd $SLURM_SUBMIT_DIR

source .venv/bin/activate

# Keep all native libraries within the CPUs requested from SLURM
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Optional but useful for your own code
export GUROBI_THREADS=$SLURM_CPUS_PER_TASK

# Run experiment for this array index
python3 experiments/experiment_4_new.py $SLURM_ARRAY_TASK_ID
