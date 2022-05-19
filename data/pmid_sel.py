#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
#PBS -N pmid_sel_prots_least_one
#PBS -e pmid_sel_prots_least_one.err
#PBS -o pmid_sel_prots_least_one.out
#PBS -l nodes=1:ppn=32:fatnode,mem=1500gb
#PBS -l walltime=24:00:00
import os
import numpy as np
import pandas as pd

#sel_pmids = set(line.strip() for line in open('/home/projects/ku_10024/people/zelili/berter/uniprot_pmid_uniq_have_abstract.txt'))
sel_pmids = set(line.strip() for line in open('/home/projects/ku_10024/people/zelili/berter/prots_least_one_matches_pmid.txt'))

folderpath = r"/home/projects/ku_10024/people/zelili/berter/data/vecs"
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

tmpdf = pd.DataFrame(columns=['pmid', 'fasttext', 'biowordvec', 'bert', 'biobert'])
for path in filepaths:
  vecs = pd.read_csv(path, index_col=0, sep='\t')
  tmpdf = pd.concat([tmpdf, vecs[vecs['pmid'].isin(sel_pmids)]])

print('total row count:', len(tmpdf.index))
tmpdf.to_csv('/home/projects/ku_10024/people/zelili/berter/data/prots_least_one_matches.vecs', sep="\t")

tmpdf_random = tmpdf.sample(n = 27500)
print('random selection row count:', len(tmpdf_random.index))
tmpdf_random.to_csv('/home/projects/ku_10024/people/zelili/berter/data/random_prots_least_one_matches.vecs', sep="\t")
