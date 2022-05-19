#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
import sys
import csv
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
#%matplotlib inline

uniprot_pmids = set(line.strip() for line in open('/home/projects/ku_10024/people/zelili/berter/uniprot_pmid_uniq_have_abstract.txt'))
workdir = sys.argv[1]
str2np = lambda str_vec: np.fromstring(str_vec.strip('][').replace('\n',''), dtype=float, sep=' ')

def read_vecs(infile):
    vecs = pd.read_csv(infile, index_col=0, sep='\t')
    vecs['fasttext'] = vecs['fasttext'].apply(str2np)
    vecs['biowordvec'] = vecs['biowordvec'].apply(str2np)
    vecs['bert'] = vecs['bert'].apply(str2np)
    vecs['biobert'] = vecs['biobert'].apply(str2np)
    return vecs

def pca_plot(vecs, outfile, labels=None):
    vecs_embedded = PCA(n_components=2).fit_transform(vecs)
    plt.figure(figsize=(24, 16))
    for i in range(len(vecs_embedded)):
        if(labels is not None):
          if(str(labels[i]) in uniprot_pmids): dotcolor='red'
          else: dotcolor='black'
        plt.scatter(vecs_embedded[i][0], vecs_embedded[i][1], s=1, color=dotcolor)
        #if(labels is not None):
        #  plt.annotate(labels[i], xy=(vecs_embedded[i][0], vecs_embedded[i][1]), ha='center', va='center', fontsize=1/np.log(1+len(str(labels[i])))) # xytext=(5, 2), textcoords='offset points'
    plt.xlabel('1st component')
    plt.ylabel('2nd component')
    plt.tight_layout()
    plt.savefig(outfile)


#with open("/home/projects/ku_10024/people/zelili/berter/vecs_out.csv", newline="") as f:
#    spamreader = csv.reader(f)
#    mean_vecs_list = []
#    for row in spamreader:
#        mean_vecs_list.append(np.array(row).astype('float32'))

with open(workdir+'/vecs_filelist.txt', 'r') as fl:
    fl_list = fl.read().split("\n")
fl_list.pop()

tmpdf = pd.DataFrame(columns=['pmid', 'fasttext', 'biowordvec', 'bert', 'biobert'])
for filename in fl_list:
    tmpdf = pd.concat([tmpdf, read_vecs(workdir+'/'+filename)])

pca_plot(tmpdf['fasttext'].values.tolist(), workdir+'/random_plot_fasttext.pdf', tmpdf['pmid'].values.tolist())
pca_plot(tmpdf['biowordvec'].values.tolist(), workdir+'/random_plot_biowordvec.pdf', tmpdf['pmid'].values.tolist())
pca_plot(tmpdf['bert'].values.tolist(), workdir+'/random_plot_bert.pdf', tmpdf['pmid'].values.tolist())
pca_plot(tmpdf['biobert'].values.tolist(), workdir+'/random_plot_biobert.pdf', tmpdf['pmid'].values.tolist())

