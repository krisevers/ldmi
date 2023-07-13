#!/bin/bash -l
#SBATCH --job-name="laminarDMF"
#SBATCH --account=ich020m
#SBATCH --time=01:30:00
#SBATCH --nodes=144
#SBATCH --ntasks-per-core=1
#SBATCH --ntasks-per-node=36
#SBATCH --cpus-per-task=1
#SBATCH --partition=normal
#SBATCH --constraint=mc
#SBATCH --hint=nomultithread

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

module load singularity

export TF_CPP_MIN_LOG_LEVEL="3"
export DISPLAY=:0

srun --mpi=pmi2 singularity run --nv --no-home --bind /apps:/apps,results:/data ldmi_explore_latest.sif python3 -u /explore.py -p /data -m DCM NVC LBR -n 144000

