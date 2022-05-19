#!/bin/bash
for i in $(ls /home/projects/ku_10024/people/zelili/berter/data/abstracts_*_dists.tsv); do j=`echo $i | sed -e 's/.tsv_dists.tsv/.dists/g'`; mv $i $j; done
for i in $(ls /home/projects/ku_10024/people/zelili/berter/data/abstracts_*_vecs.tsv); do j=`echo $i | sed -e 's/.tsv_vecs.tsv/.vecs/g'`; mv $i $j; done

