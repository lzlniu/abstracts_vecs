#!/home/projects/ku_00039/people/zelili/programs/miniconda2/envs/phyluce-1.7.1/bin/python
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
#from pandarallel import pandarallel

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

inputf = sys.argv[1]
#inputf = '/home/projects/ku_10024/people/zelili/berter/all_2021_abstracts.tsv'
#inputf = '/home/projects/ku_10024/people/zelili/berter/testinput.tsv'
print('Input file:', inputf)
uniprot_tsvs = pd.read_csv(inputf, sep='\t', header=None)
uniprot_tsvs.columns = ["pmid", "abstract"]

#pandarallel.initialize()

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
    #if(len(toks)>=512): toks = toks[:512]
    
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
  for text in uniprot_tsvs["abstract"]:
    tensors.append(get_abstract_mean_vec(text, model_name))
  #all_tensors.append(np.vstack(tensors))
  return np.mean(tensors, 0)

def get_abstract_sum_vec(li, model_name):
  if not li: return 0
  return get_embeddings(li[0], model_name) + get_abstract_sum_vec(li[1:], model_name)

def get_abstracts_sum_vec(li, model_name):
  if not li: return 0
  return get_abstract_sum_vec(sent_tokenizer.tokenize(li[0]), model_name)/len(sent_tokenizer.tokenize(li[0])) + get_abstracts_sum_vec(li[1:], model_name)

def parallel_tsvs(pddf, model_name):
  pddf[model_name] = pddf.apply(lambda x: get_abstract_mean_vec(x['abstract'], model_name), axis=1)

if __name__ == '__main__':
  with open("/home/projects/ku_10024/people/zelili/berter/vecs_out.csv", newline="") as f:
    spamreader = csv.reader(f)
    mean_vecs_list = []
    for row in spamreader:
      mean_vecs_list.append(np.array(row).astype('float32'))
  
  #device = "cuda:0" if torch.cuda.is_available() else "cpu"
  #print(f"Using device: {device}")
  model_names = ['fasttext', 'biowordvec', 'bert', 'biobert']
  
  #all_tensors = []
  #with Pool(processes=40) as pool:
  #  pool.starmap(parallel_tsvs, [(uniprot_tsvs, 'fasttext'), (uniprot_tsvs, 'biowordvec'), (uniprot_tsvs, 'bert'), (uniprot_tsvs, 'biobert')])
  for model_name in model_names:
    uniprot_tsvs[model_name] = uniprot_tsvs.apply(lambda x: get_abstract_mean_vec(x['abstract'], model_name), axis=1)
    uniprot_tsvs['dist'+model_name] = uniprot_tsvs.apply(lambda x: np.linalg.norm(x[model_name] - mean_vecs_list[model_names.index(model_name)]), axis=1)
    #all_tensors.append(get_abstracts_mean_vec(model_name))
  #for model_name in model_names:
  #  all_tensors.append(get_abstracts_sum_vec(abstracts, model_name)/len(abstracts))
  #print('Output vecs:\n', all_tensors)
  #print(uniprot_tsvs)
  uniprot_tsvs[['pmid', 'fasttext', 'biowordvec', 'bert', 'biobert']].to_csv(sys.argv[1]+'.vecs', sep="\t")
  uniprot_tsvs[['pmid', 'distfasttext', 'distbiowordvec', 'distbert', 'distbiobert']].to_csv(sys.argv[1]+'.dists', sep="\t", header=False, index=False)
  #with open("/home/projects/ku_10024/people/zelili/berter/test_vecs_out.csv", "w", newline="") as f:
  #  writer = csv.writer(f)
  #  writer.writerows(all_tensors)
  
  #pd.read_csv('.tsv', sep='\t', index_col=0)

