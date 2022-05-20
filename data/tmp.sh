#!/bin/bash
#PBS -N veccos_ef
#PBS -e logs/veccos_ef.err
#PBS -o logs/veccos_ef.out
#PBS -l nodes=1:ppn=4
#PBS -l mem=32gb
#PBS -l walltime=48:00:00

source activate phyluce-1.7.1
cd /home/projects/ku_10024/people/zelili/berter/data
python distance.py abstracts_ef

