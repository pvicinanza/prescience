'''
Author: Paul Vicinanza
Date: 09/21/2020
Purpose: Implement functions to compute prescience
'''

import pandas as pd
import numpy as np
import torch
import math
import os
from collections import defaultdict, Counter
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence
from transformers import (
    BertTokenizerFast,
    BertForMaskedLM,
)

# Import custom fuctions
from bert_finetune_utils import *

def declareTokenizer(vocab_path=None, do_lower_case=True, clean_text=True):
    '''
    Declare tokenizer to use
    @param vocab_path - Path to custom BERT vocabulary
        If None use default bert-base-uncased

    @return tokenizer - BERT fast tokenizer object
    '''
    if vocab_path is None:
        vocab_path = 'bert-base-uncased'

    return BertTokenizerFast.from_pretrained(vocab_path,
                                              do_lower_case=do_lower_case,
                                              clean_text=clean_text)

def filterSentences(df, sent_col, min_doc_length=12, max_doc_length=102,
                    sort_lengths=True):
    '''
    Filter sentences by minimum/maximum length

    @param df (dataFrame) - Dataframe of documents
    @param sent_col (str) - Column name to filter and sort by
    @param min_doc_length (int) - Minimum length of document
    @param max_doc_length (int) - Maximum length of document
    @param sort_lengths (Bool) - Whether to sort the dataframe by length ascending

    @return 
    '''
    df['len'] = df[sent_col].apply(len)

    df = df[(df['len'] >= min_doc_length) & 
        (df['len'] <= max_doc_length)]

    if sort_lengths:
        df = df.sort_values(by='len')

    return df.reset_index(drop=True)

def computePerplexity(sents, model_path, batch_size=50, num_digits=None,
                      compute_sent_perp=False):
    '''
    Compute perplexity for a list of tokenized sentences
    Varies batch size by the length of the sentences

    @param sents (list[list[int]]) - list of BERT tokenizer encoded sentences
    @param model_path (str) - Path to finetuned BERT model
    @param batch_size (int) - Batch size for forward pass through the model
        Defaults to 65 - Appropriate for 2080ti with max sequence length of 100 tokens
    @param num_digits (int) - Number of digits to round by
        If None, no rounding
        Note: inefficient implementation, better to vectorize
    @param compute_sent_perp (bool) - Whether to return sentence-level perplexity
        If False, return word-level perplexity

    @return perps (list[list[float]]) - List of the word-level perplexities from MLM task
        Note: Returned at the word-level, rather than the sentence level, to enable
              word-level analyses and adjustments 

    Dependencies:
        computePerplexityHelper
    '''

    # Load model on gpu in evalulation model if gpu is available
    # Note: Not sure eval is needed for BERT, but won't hurt anything
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForMaskedLM.from_pretrained(model_path).to(device).eval()

    # Load loss function assuming BERT tokenizer pad token idx
    loss_fct = torch.nn.CrossEntropyLoss(ignore_index=0, reduction='none')

    # Compute and return perplexity
    perps = computePerplexityHelper(sents, batch_size, model, loss_fct)

    if compute_sent_perp:  # Reduce to sentence perplexity
        return [sentencePerplexity(sent) for sent in perps]
    elif num_digits is not None: # Round items in list while retaining list structure
        raise ValueError
        #return perps = [[round(perp, num_digits) for perp in sent] for sent in perps]
    else:
        return perps

def computePerplexityHelper(encoded_sents, batch_size, model, loss_func):
    '''
    Compute perplexity for a batch of sentences by minibatching
    @param encoded_sents (list[list[int]]) - list of BERT tokenizer encoded sentences
    @param batch_size (int) - size to batch by
    @param model - BERT model for forward pass
    @param loss_func - torch loss function

    @return perp (list[list[float]]) - List of the word-level perplexities
    Called in computePerplexity
    '''

    # Load tensors on GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    perps = []  # Array to hold sequence MLM perplexities
    encoded_sents = list(encoded_sents)  # Cast as list to remove potential indexing

    tqdm_desc = "Batch Size: {}".format(batch_size)

    n_minibatches = math.ceil(len(encoded_sents) / batch_size)
    for batch in tqdm(np.array_split(encoded_sents, n_minibatches), desc=tqdm_desc):

        # Get sequence lengths
        lens = [len(x) for x in batch]

        # Batch the tokens and forward pass through the langauge model
        batch_tokens = [torch.tensor(s, requires_grad=False, device=device) for s in batch]
        batch_tokens = pad_sequence(batch_tokens, batch_first=True, padding_value=0)
        pred = model(batch_tokens)[0].detach()

        # Collapse batch compute loss
        pred = pred.view(batch_tokens.shape[0] * batch_tokens.shape[1], -1)
        batch_reshaped = batch_tokens.view(-1)
        loss = loss_func(pred, batch_reshaped).view(batch_tokens.shape[0], -1)

        # Exponentiate to obtain perplexity
        perp = torch.exp(loss).cpu().numpy()

        for i, l in enumerate(lens):
            # Append perplexity, excluding pad tokens 
            perps.append(perp[i, :l])

    torch.cuda.empty_cache() 
    return perps

def sentencePerplexity(perp, trim_special_tokens=True):
    '''
    Compute sentence-level perplexity given a list of word-level perplexities
    @param perp (list[float]) - list of word-level perplexities
    @param trim_special_tokens (bool) - whether to crop [CLS] and [SEP] tokens from start and end of list

    @return float - sentence-level-perplexity
    '''

    if trim_special_tokens:
        vals = perp[1:-1]

    # Take word-level product and normalize by # of words
    return np.prod(np.float64(vals)) ** (1.0 / len(vals))

def computePresciencePairwise(perp1_list, perp2_list, trim_special_tokens=True):
    '''
    NOTE: depreciated

    Compute prescience between two lists, returning average prescience
    e.g. if perp1_list has 1970, 1971, and 1972 model losses, while perp2_list
        has 1980 and 1981 losses, then compute prescience for the following pairs:
        [(1970, 1980),
        (1971, 1980),
        (1972, 1980),
        (1970, 1981),
        (1971, 1981),
        (1972, 1981)]

    @param perp1_list - list[list[float]] - list of early perplexities
    @param perp2_list - list[list[float]] - list of late perplexities
    @param trim_special_tokens (bool) - Whether to remove [CLS] & [SEP]

    @return presicence (float) - Statement prescience
        Average of individual prescience values
    Dependencies:
        computePrescience
    '''
    raise Exception("Deprecaited Function!!!")

    pres = []  # List to store presciences
    for p1 in perp1_list:
        for p2 in perp2_list:
            pres.append(computePrescience(p1, p2, trim_special_tokens))

    return round(np.mean(pres), 5)

def computeTokenLikelihoods(token_lists, token_perp_list, as_dataframe=True):
    '''
    Compute relative token likelihoods for a list of sequences 
    @param token_lists (list[list[int]]) - list of list of token ids
    @param token_perp_list (list[list[float]]) - list of list of token perplexities from computePerplexity
    @param as_dataframe (bool) - whether to return the average token perplexities as dataframe (otherwise defaultdict)

    @return tokenDict - Dictionary of token likelihoods indexed by token_id with value mean token perplexity
    '''

    token_dict = defaultdict(list)

    # Append token perplexities to token_dict
    for (seq_tokens, seq_perps) in tqdm(zip(token_lists, token_perp_list), total=len(token_lists), desc="Processing Sents:"):
        for token, perp in zip(seq_tokens, seq_perps):
            token_dict[token].append(perp)  # Exponentiate to obtain perplexity

    # Consolidate by taking mean perplexity
    for token_id, token_perps in tqdm(token_dict.items(), total=len(token_dict), desc="Computing Avg Perplexity:"):
        token_dict[token_id] = np.mean(np.log(token_perps))

    if as_dataframe:
        return pd.DataFrame.from_dict(token_dict, orient='index').rename(columns={0 : 'avg_perplexity'})

    return token_dict


def wordLevelPrescience(df, early_model_name, late_model_name, isEncoded=True,
                        sentence_col='sentence_encoded', vocab_path=None):
    '''
    Compute average word-level prescience for words between two models
    @param df (DataFrame) - pandas dataframe of documents with sentence perplexities
    @param early_model_name (str) - column name for early model perplexities
    @param late_model_name (str) - column name for late model perplexities
    @param isEncoded (bool) - Whether the sentences are encoded by the tokenizer
        If false, assume the sentences are tokenized strings
    @param sentence_col (str) - column name for sentences
    @param vocab_path (str) - path to vocabulary file for bert tokenizer
        If None, use BERT default. Only relevant if sentences are encoded

    @dependencies declareTokenizer, computeTokenLikelihoods
    
    @return word_perp (DataFrame) - DataFrame holding word-level perplexities
    '''

    if vocab_path is None and isEncoded:
        print('No vocabulary specificed. Using default BERT vocabulary')

    # Token likelihoods in early and late model
    mEarly = computeTokenLikelihoods(df[sentence_col], df[early_model_name])
    mLate = computeTokenLikelihoods(df[sentence_col], df[late_model_name])

    if isEncoded:
        tokenizer = declareTokenizer(vocab_path)
        mEarly['word'] = tokenizer.convert_ids_to_tokens(mEarly.index.values)
        mLate['word'] = tokenizer.convert_ids_to_tokens(mLate.index.values)
    else:
        mEarly.index.name = 'word'
        mLate.index.name = 'word'
        mEarly.reset_index(inplace=True)
        mLate.reset_index(inplace=True)

    word_perp =  mEarly.merge(mLate, on='word', suffixes=['_current', '_future'])
    word_perp['diff'] = word_perp['avg_perplexity_current'] - word_perp['avg_perplexity_future']
    word_perp['diff_percent'] = (word_perp['avg_perplexity_current'] - word_perp['avg_perplexity_future']) / word_perp['avg_perplexity_current']

    return word_perp

def computePerplexitiesForPrescience(df, text_col, model_names, vocab='bert-base-uncased',
                                     min_doc_length=12, batch_size=50, compute_sent_perp=True):
    '''
    Calculate perplexities across models
        Main function to call when computing perplexity

    @param df (DataFrame) - Pandas dataframe holding the text
    @param text_col (str) - Name of column holding text to evaluate
    @param model_names (list[str]) - List of paths/names to models for evaluation
    @param vocab (str) - Path to custom tokenizer vocabulary
        Otherwise default to bert-base-uncased
    @param min_doc_length (int) - Minimum document length post-tokenization
        Including special characters [CLS] and [SEP]
        Prescience computation is more accurate on longer documents.
        Do not go below the minimum.
    @param batch_size (int) - Batch size for forward pass through BERT. 
        Larger batch size --> quicker computation but more memory intensive
    @param compute_sent_perp (bool) - Whether to compute average sentence perplexity
        If False, return word-level perplexity

    @return df (DataFrame) - Pandas dataframe holding prescience columns

    @dependencies - filterSentences, declareTokenizer, computePerplexity
    '''

    # Remove na
    df = df[df[text_col].notnull()]

    # Declare BERT tokenizer
    tokenizer = declareTokenizer(vocab_path=vocab)

    # Tokenize in parallel
    print(f'Tokenizing sentences using {vocab} vocabulary')
    df['sentence_encoded'] = (df[text_col].swifter
                               .allow_dask_on_strings()
                               .apply(tokenizer.encode))

    # Filter sentences below threshold
    df = filterSentences(df, 'sentence_encoded', sort_lengths=True, min_doc_length=min_doc_length)

    # Iterate over models to compute perplexity
    for model in model_names:
        model_name = os.path.split(model)[-1]

        print("\tComputing perplexity for model {}".format(model_name))

        # Get last checkpoint in model checkpoint files
        # Note: Assumes that the file with 'checkpoint-' followed by the
            # largest number is the correct model to use
        model_dir = model + '/' + sorted([i for i in os.listdir(model) if i.startswith('checkpoint-')])[-1]

        # Compute loss and append to dataframe
        col_name = 'perp_{}'.format(model_name)
        df[col_name] = computePerplexity(df['sentence_encoded'], model_dir,
                                         compute_sent_perp=compute_sent_perp,
                                         batch_size=batch_size)

    df['len'] = df['len'] - 2  # Adjust length for [CLS] and [SEP] tokens

    # Return dataframe
    return df


#### Functions to compute word-level prescience in context
# Primary function - wordPrescienceInContext
# Helper functions - wordPrescience, focalWordPrescience

def wordPrescienceInContext(df, word, m1_col, m2_col, sent_col, tokenizer):
    '''
    Identify for a single token its prescience in a sentence
    @param df (DataFrame) - dataframe holding data
    @param m1_col (str) - Column holding word-level perplexities for model 1
    @param m2_col (str) - Column holding word-level perplexities for model 2
    @param sent_col (str) - Column holding the sentences in encoded form
    @param tokenizer (Bert Tokenizer) = Tokenizer to convert tokens to ids

    @return filtered_df (DataFrame) - Pandas dataframe holding sentences with word-level prescience

    @dependencies wordPerp focalWordPrescience
    '''

    ind = tokenizer.convert_tokens_to_ids([word])[0]

    # Filter sentences which contain focal word
    tqdm.pandas(desc="Filtering sentences for {}".format(word))
    mask = df[sent_col].progress_apply(lambda x: ind in x)
    filtered_df = df[mask].copy(deep=True).reset_index(drop=True)

    # Compute word-level prescience
    filtered_df['word_prescience'] = filtered_df.apply(lambda x : wordPrescience(x[m1_col], x[m2_col]), axis=1)

    # Extract prescience for target word
    filtered_df['focal_word_prescience'] = filtered_df.apply(lambda x : focalWordPrescience(x[sent_col], x['word_prescience'], ind), axis=1)
    
    # Get average word prescience
    filtered_df['average_word_prescience'] = [np.mean(i) if isinstance(i, list) else i for i in filtered_df['focal_word_prescience']]
    
    return (filtered_df.sort_values(by=['average_word_prescience'], ascending=False)
                       .reset_index(drop=True)
                       .drop(columns=['focal_word_prescience']))

def wordPrescience(m1, m2):
    '''
    Compute word-level perplexities
    @param s1 (list[float]) - word-level perplexities for model 1
    @param s2 (list[float]) = word-level perplexities for model 2
    
    Helper function for wordPrescienceInContext
    '''

    return [(w1 - w2) / w1 for w1, w2 in zip(m1, m2)]

def focalWordPrescience(sent, prescience, token_ind):
    '''
    Identify prescience for target words
    @param sent (list[int]) - List of token idx
    @param prescience (list[float]) - List of word-level prescience
    @param token_ind (int) - Token idx to filter for 

    @return float or list[float] - Prescience in context for focal word
        - Note: List[float] rather than float because word can occur multiple times

    Helper function for wordPrescienceInContext
    '''

    # If item is not in list return null
    if token_ind not in sent:
        return

    # Get index locations
    inds = np.where(np.isin(sent, token_ind))[0]

    # Return int if a single occurance
    if len(inds) == 1:
        return prescience[inds[0]]
    else:
        return [prescience[i] for i in inds]

# Vectorize percentage decrease
def percentageDecrease(p1, p2):
    return (p1 - p2) / p1

percentDiffVec = np.vectorize(percentageDecrease)