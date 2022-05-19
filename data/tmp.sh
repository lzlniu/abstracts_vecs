#!/bin/bash
#PBS -N vec_eaaj
#PBS -e vec_eaaj.err
#PBS -o vec_eaaj.out
#PBS -l nodes=1:ppn=8
#PBS -l mem=80gb
#PBS -l walltime=48:00:00

source activate phyluce-1.7.1
cd /home/projects/ku_10024/people/zelili/berter/data
python vectorizer.py /home/projects/ku_10024/people/zelili/berter/data/abstracts_eaaj.tsv

