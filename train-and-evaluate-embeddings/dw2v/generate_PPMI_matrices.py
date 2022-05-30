from collections import defaultdict
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import pickle
import os

def main():
    #Parameters
    MOVING_WINDOW = 5
    AGGREGATION_INTERVAL = 5

    key_to_index = load_key_to_index()
    for year in tqdm(range(1867,2022,AGGREGATION_INTERVAL)):
        sentences=[]
        for constant in range(AGGREGATION_INTERVAL):
            sentences.extend(load_sentences(year+constant))
        cooccurrence_matrix = co_occurrence(sentences,MOVING_WINDOW, key_to_index)
        ppmi_matrix = ppmi(cooccurrence_matrix)
        ppmi_long = ppmi_to_long_format(ppmi_matrix)
        ppmi_df = pd.DataFrame(ppmi_long,columns=["pmi","word","context"]).loc[:,["word","context","pmi"]]
        store_pmi(ppmi_df,f"PPMI/{AGGREGATION_INTERVAL}_year_interval/",year)
    return True

def load_key_to_index():
    key_to_index = pd.read_csv("wordIDHash_ufr.csv")
    key_to_index = key_to_index.set_index("word")
    return key_to_index["index"].to_dict()

def co_occurrence(sentences, window_size, key_to_index):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple(sorted([t, token]) )
                d[key] += 1
    # formulate the dictionary into dataframe
    vocab = sorted(vocab) # sort vocab
    vocab = [key_to_index[v] for v in vocab if v in key_to_index.keys()]
    cooccurence = np.zeros([len(key_to_index),len(key_to_index)],dtype=np.int16)
    for key, value in d.items():
        if all(subkeys in key_to_index.keys() for subkeys in key):
            cooccurence[key_to_index[key[0]], key_to_index[key[1]]] = value
            cooccurence[key_to_index[key[1]], key_to_index[key[0]]] = value
    return cooccurence

def ppmi(m, positive=True):
    col_totals = m.sum(axis=0)
    total = col_totals.sum()
    row_totals = m.sum(axis=1)
    # Silence distracting warnings about log(0) and dividing with zero:
    with np.errstate(invalid="ignore",divide='ignore'):
        expected = np.outer(row_totals, col_totals) 
        m = (m * total) / expected
        m = np.log(m)
    m[np.isinf(m)] = 0.0  # log(0) = 0
    if positive:
        m[m < 0] = 0.0
    return m

def ppmi_to_long_format(ppmi):
    XX,YY = np.meshgrid(np.arange(ppmi.shape[1]),np.arange(ppmi.shape[0]))
    ppmi_1d = ppmi.ravel()
    ppmi_nonzero_or_nan = ~np.logical_or(ppmi_1d==0,np.isnan(ppmi_1d))
    ppmi_long = np.vstack((ppmi_1d[ppmi_nonzero_or_nan],XX.ravel()[ppmi_nonzero_or_nan],YY.ravel()[ppmi_nonzero_or_nan])).transpose()
    return ppmi_long

def load_sentences(year):
    with open(f"C:/Users/espen/Documents/SDS/thesis/data/processed/sentences/year/{year}_sentences.pkl", "rb") as fp:
        sentences = pickle.load(fp)
    return sentences

def store_pmi(df,path,time,prefix="wordPairPMI_"):
    if not os.path.exists(path):
        os.makedirs(path)
    df.to_csv(f"{path}{prefix}{time}.csv",encoding="UTF-8")

if __name__== "__main__":
    main()
