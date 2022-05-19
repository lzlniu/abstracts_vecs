#!/bin/bash
cd /home/projects/ku_10024/people/zelili/berter/data
#awk 'length($2)' all_2021_abstracts.tsv > all_noempty_2021_abstracts.tsv
split --additional-suffix=.tsv -l 300000 all_noempty_2021_abstracts.tsv abstracts_

