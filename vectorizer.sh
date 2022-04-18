#!/bin/bash
#PBS -N vectorizer
#PBS -e vectorizer.err
#PBS -o vectorizer.out
#PBS -l nodes=1:ppn=10
#PBS -l mem=50gb
#PBS -l walltime=96:00:00

#source activate your_env_name
#conda activate your_env_name
#source "/home/projects/ku_00039/people/zelili/programs/miniconda2/bin"
source activate phyluce-1.7.1
cd /home/projects/ku_10024/people/zelili/berter/
python vectorizer.py output_uniq_rmempty.tsv

