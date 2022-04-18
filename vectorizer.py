#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
#PBS -N vectorizer
#PBS -e vectorizer.err
#PBS -o vectorizer.out
#PBS -l nodes=1:ppn=10
#PBS -l mem=60gb
#PBS -l walltime=96:00:00

import nltk
import fasttext
import fasttext.util
import numpy as np
import pandas as pd
import sys
import csv
import torch
#from multiprocessing import Pool
#from functools import partial
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder

#fasttext.util.download_model('en', if_exists='ignore') # English
#fasttext.util.reduce_model(ft, 100)
#nltk.download('punkt')

ft = fasttext.load_model('/home/projects/ku_10024/people/zelili/berter/cc.en.300.bin')
ftbio = fasttext.load_model('/home/projects/ku_10024/people/zelili/berter/BioWordVec_PubMed_MIMICIII_d200.bin')

# Load pre-trained model (weights)
bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True) #.to(device)
biobert_model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2', output_hidden_states = True)
# Put the model in "evaluation" mode, meaning feed-forward operation
bert_model.eval()
biobert_model.eval()
# load pre-trained tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased') #add_special_tokens = True, return_special_tokens_mask = True
biobert_tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2') #add_special_tokens = True, return_special_tokens_mask = True
#marked_text = '[CLS] ' + orig_text + ' [SEP]'

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

#inputf = sys.argv[1]
inputf = '/home/projects/ku_10024/people/zelili/berter/output_uniq_rmempty.tsv'
print('Input file:', inputf)
uniprot_tsvs = pd.read_csv(inputf, sep='\t', header=None)
uniprot_tsvs.columns = ["PubMed ID", "Abstract"]

def split_abstract(abstract):
  sentenses = []
  for sentense in abstract.split(". "):
    sentenses.append(sentense+".")
  return sentenses

@torch.no_grad()
def get_embeddings(text, model_name):
  if model_name=='fasttext':
    toks = nltk.word_tokenize(text)
    vecs = [ft.get_word_vector(j) for j in toks]
    if(len(vecs)>=512): vecs = vecs[:512]
    return np.add.reduce(vecs)/len(vecs)
  elif model_name=='biowordvec':
    toks = nltk.word_tokenize(text)
    vecs = [ftbio.get_word_vector(j) for j in toks]
    if(len(vecs)>=512): vecs = vecs[:512]
    return np.add.reduce(vecs)/len(vecs)
  elif model_name=='bert':
    # Tokenize with BERT tokenizer
    #toks = bert_tokenizer.tokenize(text)
    tok_code = bert_tokenizer.encode(text)
    
    # Indexing tokens
    #indexed_tokens = bert_tokenizer.convert_tokens_to_ids(toks)
    
    if(len(tok_code)>=512): tok_code = tok_code[:512]
    # Mark each of the tokens as belonging to sentence "1".
    #segments_ids = [1] * len(toks) # same as tok_code[1:-1]
    segments_ids = [1] * len(tok_code)
    
    # Convert inputs to PyTorch tensors and output embedding as last hidden state
    #return model(torch.tensor([indexed_tokens]), torch.tensor([segments_ids])).last_hidden_state[0]
    vecs = bert_model(torch.tensor([tok_code]), torch.tensor([segments_ids])).last_hidden_state[0]
    return torch.mean(vecs, 0).detach().numpy()
  elif model_name=='biobert':
    # Tokenize with BioBERT tokenizer
    #toks = biobert_tokenizer.tokenize(text)
    tok_code = biobert_tokenizer.encode(text)
    
    # Indexing tokens
    #indexed_tokens = biobert_tokenizer.convert_tokens_to_ids(toks)
    
    if(len(tok_code)>=512): tok_code = tok_code[:512]
    # Mark each of the tokens as belonging to sentence "1".
    #segments_ids = [1] * len(toks) # same as tok_code[1:-1]
    segments_ids = [1] * len(tok_code)
    
    # Convert inputs to PyTorch tensors and output embedding as last hidden state
    #return model(torch.tensor([indexed_tokens]), torch.tensor([segments_ids])).last_hidden_state[0]
    vecs = biobert_model(torch.tensor([tok_code]), torch.tensor([segments_ids])).last_hidden_state[0]
    return torch.mean(vecs, 0).detach().numpy()
  else:
    print("ERROR: model not identical")

def get_abstract_mean_vec(text, model_name):
  sent_tensors = []
  for sent in sent_tokenizer.tokenize(text):
    sent_tensors.append(get_embeddings(sent, model_name))
  return np.mean(sent_tensors, 0)

def get_abstracts_mean_vec(model_name):
  tensors = []
  for text in uniprot_tsvs["Abstract"]:
    tensors.append(get_abstract_mean_vec(text, model_name))
  #all_tensors.append(np.vstack(tensors))
  return np.mean(tensors, 0)

#if __name__ == '__main__':
#device = "cuda:0" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")
model_names = ['fasttext', 'biowordvec', 'bert', 'biobert']

all_tensors = []
#with Pool(len(model_names)) as p:
#  all_tensors = p.map(get_abstracts_mean_vec, model_names)
for model_name in model_names:
  all_tensors.append(get_abstracts_mean_vec(model_name))
print('Output vecs:\n', all_tensors)

with open("/home/projects/ku_10024/people/zelili/berter/vecs_out.csv", "w", newline="") as f:
  writer = csv.writer(f)
  writer.writerows(all_tensors)
