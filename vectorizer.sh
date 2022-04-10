#!/bin/bash
#PBS -N d_eng
#PBS -e d_eng.err
#PBS -o d_eng.out
#PBS -l walltime=8:00:00

#source activate your_env_name
#conda activate your_env_name
#source "/home/projects/ku_00039/people/zelili/programs/miniconda2/bin"
source activate phyluce-1.7.1
cd /home/projects/ku_10024/people/zelili/berter/
python vectorizer.py
