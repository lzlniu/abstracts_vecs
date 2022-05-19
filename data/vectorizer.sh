#!/bin/bash

#source activate your_env_name
#conda activate your_env_name
#source "/home/projects/ku_00039/people/zelili/programs/miniconda2/bin"

for i in $(cat process_list.txt); do
j=`echo $i | awk -F '_' '{print $NF}' | sed -e 's/.tsv//g'`
echo "#!/bin/bash
#PBS -N vec_${j}
#PBS -e vec_${j}.err
#PBS -o vec_${j}.out
#PBS -l nodes=1:ppn=8
#PBS -l mem=80gb
#PBS -l walltime=48:00:00

source activate phyluce-1.7.1
cd /home/projects/ku_10024/people/zelili/berter/data
python vectorizer.py /home/projects/ku_10024/people/zelili/berter/data/${i}
" > /home/projects/ku_10024/people/zelili/berter/data/tmp.sh
qsub /home/projects/ku_10024/people/zelili/berter/data/tmp.sh
done

