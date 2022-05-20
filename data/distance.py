#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
import numpy as np
import pandas as pd
import sys
import csv

inputf = sys.argv[1]

str2np = lambda str_vec: np.fromstring(str_vec.strip('][').replace('\n',''), dtype=float, sep=' ')

def read_vecs(infile):
  vecs = pd.read_csv(infile, index_col=0, sep='\t')
  vecs['fasttext'] = vecs['fasttext'].apply(str2np)
  vecs['biowordvec'] = vecs['biowordvec'].apply(str2np)
  vecs['bert'] = vecs['bert'].apply(str2np)
  vecs['biobert'] = vecs['biobert'].apply(str2np)
  return vecs

if __name__ == '__main__':
  with open("/home/projects/ku_10024/people/zelili/berter/vecs_out.csv", newline="") as f:
    spamreader = csv.reader(f)
    mean_vecs_list = []
    for row in spamreader:
      mean_vecs_list.append(np.array(row).astype('float32'))
  
  #with open('/home/projects/ku_10024/people/zelili/berter/data/vecs_filelist.txt', 'r') as fl:
  #  fl_list = fl.read().split("\n")
  #fl_list.pop()
  
  tmpdf = pd.DataFrame(columns=['pmid', 'fasttext', 'biowordvec', 'bert', 'biobert'])
  #for filename in fl_list:
  #  tmpdf = pd.concat([tmpdf, read_vecs('/home/projects/ku_10024/people/zelili/berter/data/vecs/'+filename)])
  tmpdf = pd.concat([tmpdf, read_vecs('/home/projects/ku_10024/people/zelili/berter/data/vecs/'+inputf+'.vecs')])
  
  #device = "cuda:0" if torch.cuda.is_available() else "cpu"
  #print(f"Using device: {device}")
  model_names = ['fasttext', 'biowordvec', 'bert', 'biobert']
  for model_name in model_names:
    tmpdf['cossim_'+model_name] = tmpdf.apply(lambda x: np.dot(x[model_name], mean_vecs_list[model_names.index(model_name)])/(np.linalg.norm(x[model_name]) * np.linalg.norm(mean_vecs_list[model_names.index(model_name)])), axis=1)
  tmpdf[['pmid', 'cossim_fasttext', 'cossim_biowordvec', 'cossim_bert', 'cossim_biobert']].to_csv('/home/projects/ku_10024/people/zelili/berter/data/vecs/'+inputf+'.cossim', sep="\t", header=False, index=False)

