import nltk
import fasttext
import fasttext.util
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.preprocessing import LabelEncoder

#fasttext.util.download_model('en', if_exists='ignore') # English
#fasttext.util.reduce_model(ft, 100)
#nltk.download('punkt')

orig_text = "This is a sentense, I doubt if this gene is correct. A certain gene can be the best!"

@torch.no_grad()
def get_embeddings(text, model_name):
  if model_name=='fasttext':
    ft = fasttext.load_model('cc.en.300.bin')
    toks = nltk.word_tokenize(orig_text)
    vecs = [ft.get_word_vector(j) for j in toks]
    return toks, vecs, np.add.reduce(vecs)/len(vecs)
  else:
    # Load pre-trained model (weights)
    model = BertModel.from_pretrained(model_name, output_hidden_states = True) #.to(device)
    # Put the model in "evaluation" mode, meaning feed-forward operation
    model.eval()
    # load pre-trained tokenizer
    tokenizer = BertTokenizer.from_pretrained(model_name) #add_special_tokens = True, return_special_tokens_mask = True
    #marked_text = '[CLS] ' + orig_text + ' [SEP]'
    
    # Tokenize with BERT tokenizer
    toks = tokenizer.tokenize(text)
    tok_code = tokenizer.encode(text)
    
    # Indexing tokens
    #indexed_tokens = tokenizer.convert_tokens_to_ids(tok_text)
    
    # Mark each of the tokens as belonging to sentence "1".
    #segments_ids = [1] * len(tok_text) # same as tok_code[1:-1]
    segments_ids = [1] * len(tok_code)
    
    # Convert inputs to PyTorch tensors and output embedding as last hidden state
    #return model(torch.tensor([indexed_tokens]), torch.tensor([segments_ids])).last_hidden_state[0]
    vecs = model(torch.tensor([tok_code]), torch.tensor([segments_ids])).last_hidden_state[0]
    return toks, vecs, torch.mean(vecs, 0)

#device = "cuda:0" if torch.cuda.is_available() else "cpu"
#print(f"Using device: {device}")

model_names = ['fasttext', 'bert-base-uncased', 'dmis-lab/biobert-base-cased-v1.2']

for model_name in model_names:
  tokens, _, mean_vec = get_embeddings(orig_text, model_name)
  print('model:', model_name)
  print('tokens:', tokens)
  print('mean vector:', mean_vec)
  print()

