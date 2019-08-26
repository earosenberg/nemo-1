#!/bin/sh
#SBATCH --nodes=14
#SBATCH --ntasks-per-node=20
#SBATCH --mem=64000
#SBATCH --time=23:59:00

source ~/.bashrc
time mpiexec nemo MFMF_S16_auto.yml -M
