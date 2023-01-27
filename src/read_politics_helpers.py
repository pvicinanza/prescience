'''
Author: Paul Vicinanza
Date: 01/18/2023
Purpose: Helper functions to read and process politics data
'''

import numpy as np
import pandas as pd
import os
import swifter
from nltk.tokenize import sent_tokenize
from bert_finetune_utils import expandDF

###
### Reading Political Speech Data ###
###

def splitSents(s):
    '''
    Split sentences with nltk sent_tokenize and join sentences that are 
        improperly split by the tokenizer due to data issues (lots of additional)
        punctuation
    @param s (list[str]) - list of tokenized sentences

    @return (list[str]) - list of tokenized sentences
    '''

    s = sent_tokenize(s)

    new_sents = [s[0]]

    for sent in s[1:]:
        if sent[0].islower():
            if new_sents[-1][-1] == '.':  # Remove period from improper split
                new_sents[-1] = new_sents[-1][:-1] + ' ' + sent
            else:
                new_sents[-1] = new_sents[-1] + ' ' + sent
        else:
            new_sents.append(sent)
    return new_sents

def readCongress(file):
    '''
    Read in and progress congressional data
    @param file (str) - File of congressional speeches to process

    @return df (DataFrame) - Dataframe holding political speeches

    @dependencies splitSents
    ''' 
    df = pd.read_csv(file, sep='\n', encoding='latin-1')
    df = [x[0].split('|')[:2] for x in df.values]   # Split on | - text after second | is dropped - extremely rare and inconsequential 
    df = pd.DataFrame(df, columns=['speech_id', 'speech'])

    # Split dataframe on sentences
    df = df[df['speech'] != ''] # Drop empty strings
    df['speech'] = df['speech'].swifter.allow_dask_on_strings().apply(lambda x : splitSents(x))

    # Expand dataframe so that each sentence is a unique row
    df = expandDF(df, 'speech')

    return df

