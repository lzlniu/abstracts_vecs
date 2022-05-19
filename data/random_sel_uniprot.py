#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
#PBS -N random_uniprot
#PBS -e random_uniprot.err
#PBS -o random_uniprot.out
#PBS -l nodes=1:ppn=2,mem=16gb
#PBS -l walltime=12:00:00
import os
import numpy as np
import pandas as pd

#sel_pmids = set(line.strip() for line in open('/home/projects/ku_10024/people/zelili/berter/uniprot_pmid_uniq_have_abstract.txt'))

#folderpath = r"/home/projects/ku_10024/people/zelili/berter/data/vecs"
#filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

#tmpdf = pd.DataFrame(columns=['pmid', 'fasttext', 'biowordvec', 'bert', 'biobert'])
#for path in filepaths:
#  vecs = pd.read_csv(path, index_col=0, sep='\t')
#  if(len(path)>len('/home/projects/ku_10024/people/zelili/berter/data/vecs/abstracts_aa.vecs')):
#    vecs = vecs.sample(n = 25)
#  else:
#    vecs = vecs.sample(n = 250)
#  tmpdf = pd.concat([tmpdf, vecs])

tmpdf = pd.read_csv('/home/projects/ku_10024/people/zelili/berter/data/uniprot.vecs', index_col=0, sep='\t')
tmpdf = tmpdf.sample(n = 2500)
print('row count:', len(tmpdf.index))
tmpdf.to_csv('/home/projects/ku_10024/people/zelili/berter/data/random_uniprot.vecs', sep="\t")
