#!/bin/bash
#PBS -N random_plot
#PBS -e random_plot.err
#PBS -o random_plot.out
#PBS -l nodes=1:ppn=20:fatnode,mem=192gb
#PBS -l walltime=96:00:00

source activate phyluce-1.7.1
work='/home/projects/ku_10024/people/zelili/berter/data'
cd $work
#ls *.vecs > $work/vecs_filelist.txt
python $work/plot.py $work

