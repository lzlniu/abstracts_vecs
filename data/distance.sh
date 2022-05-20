#!/bin/bash

#source activate your_env_name
#conda activate your_env_name
#source "/home/projects/ku_00039/people/zelili/programs/miniconda2/bin"

for i in $(cat vecs_filelist.txt); do
j=`echo $i | awk -F '_' '{print $NF}' | sed -e 's/.vecs//g'`
echo "#!/bin/bash
#PBS -N veccos_${j}
#PBS -e logs/veccos_${j}.err
#PBS -o logs/veccos_${j}.out
#PBS -l nodes=1:ppn=4
#PBS -l mem=32gb
#PBS -l walltime=48:00:00

source activate phyluce-1.7.1
cd /home/projects/ku_10024/people/zelili/berter/data
python distance.py abstracts_${j}
" > /home/projects/ku_10024/people/zelili/berter/data/tmp.sh
qsub /home/projects/ku_10024/people/zelili/berter/data/tmp.sh
done

