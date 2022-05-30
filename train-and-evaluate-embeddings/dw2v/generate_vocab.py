import pandas as pd
from gensim.utils import tokenize
from sklearn.feature_extraction.text import CountVectorizer

WORD_COUNT_MIN = 200

def main():
    df = pd.read_parquet("C:/Users/espen/Documents/SDS/thesis/data/processed/pyarrow/UfR_text.parquet")
    cv = CountVectorizer(tokenizer=lambda x: tokenize(x, lowercase=True))
    cv.fit(df['verdict_text'])
    results = cv.transform(df['verdict_text'])

    word_count =(results.sum(axis=0)).tolist()[0]
    df_index = pd.DataFrame(cv.vocabulary_,index=[0]).transpose()
    df_index = df_index.sort_values(by=0).reset_index().drop(columns=0).rename(columns={"index":"word"})
    df_word_count = pd.merge(df_index,pd.DataFrame(word_count).rename(columns={0:"count"}),left_index=True,right_index=True)

    key_to_index = df_word_count.loc[(df_word_count["count"]>=WORD_COUNT_MIN)] \
                                    .reset_index()  \
                                    .reset_index()  \
                                    .set_index("word")  \
                                    .drop(["index"],axis=1) \
                                    .rename(columns={"level_0":"index"})


    key_to_index.to_csv("wordIDHash_ufr.csv")

if __name__=="__main__":
    main()