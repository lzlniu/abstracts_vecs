#!/bin/bash

for i in $(ls abstracts_*.tsv | sed 's/.tsv//g'); do
	if [ ! -f ${i}.vecs ]; then
		echo $i.tsv
		#split --additional-suffix=.tsv -l 30000 ${i}.tsv abstracts_${i}
	fi
done

