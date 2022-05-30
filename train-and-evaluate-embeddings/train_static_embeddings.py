import logging
from utils import *

import gensim
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
tqdm.pandas()


def main():
    EMBEDDING_DIMENSIONS = 200
    MOVING_WINDOW = 5
    SEED = 1
    MIN_COUNT = 200
    NEGATIVE_DRAWS = 10
    NEGATIVE_EXPONENT = 0.75
    SUBSAMPLE_THRESHOLD = 7500
    
    sentences = gensim.models.word2vec.PathLineSentences("C:/Users/espen/Documents/SDS/thesis/data/processed/sentences/year_txt")

    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = gensim.models.Word2Vec(vector_size=EMBEDDING_DIMENSIONS, 
                                   window=MOVING_WINDOW, 
                                   seed=SEED, 
                                   min_count=MIN_COUNT, 
                                   workers=6, 
                                   sg=1,
                                   negative=NEGATIVE_DRAWS,
                                   ns_exponent=NEGATIVE_EXPONENT,
                                   sample = SUBSAMPLE_THRESHOLD, 
                                   shrink_windows=True)
    model.build_vocab(sentences)
    model.train(sentences,total_examples=model.corpus_count, epochs=model.epochs)
    model.save(f"emb_static_ufr_EMB_{EMBEDDING_DIMENSIONS}.pkl")

if __name__=="__main__":
    main()