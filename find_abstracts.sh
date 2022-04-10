#!/bin/bash
#PBS -N find_abstracts
#PBS -e find_abstracts.err
#PBS -o find_abstracts.out
#PBS -l walltime=24:00:00

cd /home/projects/ku_10024/people/zelili/berter
make
./find_abstracts uniprot_pmid_uniq.txt uniprot_abstracts.tsv $(ls /home/projects/ku_10024/data/databases/pubmed/pubmed*.tsv.gz)

