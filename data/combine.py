#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
import os
import sys
import csv
import numpy as np
import pandas as pd

folderpath = r"/home/projects/ku_10024/people/zelili/berter/data/vecs"
filepaths  = [os.path.join(folderpath, name) for name in os.listdir(folderpath)]

i = 0
for path in filepaths:
  if(len(path)>len('/home/projects/ku_10024/people/zelili/berter/data/vecs/abstracts_aa.vecs')):
    if(i%10==0 or i == 159):
      if(i != 0):
        print('save abstracts_'+str(i)+'.vecs')
        tmpdf.to_csv('/home/projects/ku_10024/people/zelili/berter/data/abstracts_'+str(i)+'.vecs', sep="\t")
      print('create new df from', i)
      tmpdf = pd.DataFrame(columns=['pmid', 'fasttext', 'biowordvec', 'bert', 'biobert'])
    vecs = pd.read_csv(path, index_col=0, sep='\t')
    tmpdf = pd.concat([tmpdf, vecs])
    i = i + 1

