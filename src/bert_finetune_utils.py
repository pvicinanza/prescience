'''
Author: Paul Vicinanza
Date: 08/26/2020
Purpose:
    Implement code needed to run finetune_bert.py

TODO:
    Modify generate_dev_data
'''

import pandas as pd
import numpy as np
import random
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast
from tqdm.auto import tqdm

class BertDataset(Dataset):
    '''
    Convert list of files to Dataset object for training
    '''
    def __init__(self, docs, tokenizer_vocab_path=None, parallelize=False, min_doc_len=12,
                 max_doc_len=102):
        '''
        @param docs (list[str]) - list of documents
            Note: Documents should be shuffled ahead of time to ease fine-tuning
        @param tokenizer_path (str) - Path to custom vocabulary file
            If None, use default tokenizer ('bert-base-uncased')
        @param parallelize (bool) - Whether to parallelize document tokenization  
            Not Implemented
        @param min_doc_len (int) - minimum document length 
            including special characters after tokenization
        @param max_doc_len (int) - maximum document length
            including special characters after tokenization
        '''
        if tokenizer_vocab_path is None:
            tokenizer_vocab_path = 'bert-base-uncased'

        self.tokenizer = BertTokenizerFast.from_pretrained(tokenizer_vocab_path,
                                                           do_lower_case=True,
                                                           clean_text=True)

        self.docs = []
        if parallelize:
            raise NotImplementedError("Seems like a useful feature, eh?")
        else:
            for doc in tqdm((docs)):
                tokens = self.tokenizer.encode(doc)

                # Drop documents outside of document range
                if ((len(tokens) >= min_doc_len) and
                   (len(tokens) <= max_doc_len)):
                    self.docs.append(tokens)

    def __len__(self):
        return len(self.docs)

    def __getitem__(self, i):
        return torch.tensor(self.docs[i])


def expandDF(df, col_name):
    '''
    Expand dataframe along a column, holding other rows at current values
    e.g. 
        a, [1,2,3], d
        b, [2], f
    becomes
        a, 1, d
        a, 2, d
        a, 3, d
        b, 2, f

    @param df (DataFrame) - Dataframe to expand
    @param col_name (str) - column name to expand on
        Must be a column of lists

    @return Expanded df
    '''
    df = df.copy(deep=True)  # If suffering memory issues remove this line

    # Expand dataframe to the sentence level
    df.reset_index(inplace=True, drop=True)

    # Flatten columns of lists
    col_flat = [sentence for doc in df[col_name] for sentence in doc] 

    #Row numbers to repeat
    lens = df[col_name].apply(len)
    vals = range(df.shape[0])
    ilocations = np.repeat(vals, lens)

    # Replicate rows and add flattened column of lists
    cols = [i for i,c in enumerate(df.columns) if c != col_name]
    df = df.iloc[ilocations, cols].copy()
    df[col_name] = col_flat

    return df.reset_index(drop=True)

def setGPU():
    '''
    Establish on GPU
    Prints device name for confirmation
    '''
    try: 
        torch.cuda.set_device(0)  # Default to first GPU
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        print(torch.cuda.get_device_name(0))
    except:
        print("No cuda compatible GPU found")
