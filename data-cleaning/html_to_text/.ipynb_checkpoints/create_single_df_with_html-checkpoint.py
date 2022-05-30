"""
Contains code to open files and store html in DataFrame 
"""
import os, codecs

import pyarrow as pa
import pyarrow.parquet as pq

import pandas as pd
from tqdm.auto import tqdm

def main():
    #Config
    os.chdir('C:/Users/espen/Documents/SDS/thesis') 
    tqdm.pandas()

    #Read unhydrated dataset
    df = pd.read_pickle("data/index_of_verdicts_in_UfR_only_articles")
    
    #Create directory path for each file (observation) in the unhydrated dataset
    for year in tqdm(reversed(df['year'].unique())):
        df.loc[df["year"]==year, "directory"]=df["id_dir_compliant"].apply(lambda x: f"data/UfR_html/{year}/{x}.html")
    
    #Open each html file and store the html as text in df
    df['html_data'] = df['directory'].progress_apply(lambda x: open_html_file(x))

    #Save df as .parquet file
    storeDataframe(df,"data/processed/pyarrow/verdicts_in_UfR_only_articles_with_html")
    return 1

def open_html_file(path):
    """Opens html file and returns the html as a string. 
    If the file cannot be found it returns pd.NA.

    Args:
        path (str): path to file

    Returns:
        str: html code OR pd.NA (if missing file)
    """

    try: 
        r = codecs.open(path,"r","utf-8")
        r = r.read()
    except FileNotFoundError:
        r = pd.NA
    return r

def storeDataframe(df, path):
    """Store DF using the Apache Arrow package

    Args:
        df (pd.DataFrame): Pandas DataFrame
        path (str): Where to store .parquet file (including name of file) 
    """
    table = pa.Table.from_pandas(df)
    pq.write_table(table,path)

if __name__ == "__main__":
    main()