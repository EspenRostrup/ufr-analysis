""" This script saves a dataframe as parquet containing the text between a "thi kendes for ret" part 
    and the next header in the document.
"""

import pyarrow as pa
from pyarrow import parquet as pq
import pandas as pd
from tqdm.auto import tqdm
from bs4 import BeautifulSoup, NavigableString, Tag
import re

def main():
    #Config
    tqdm.pandas()
    
    #Load data
    table = pq.read_table("./data/processed/pyarrow/UfR_text.parquet")
    df = table.to_pandas()

    # Count thi kendes for ret appearences
    df["thi_kendes_for_ret_antal"] = df["verdict_text"].progress_apply(lambda x: x.lower().count("thi kendes for ret"))
    
    # Only consider texts that contain "thi kendes for ret"
    df_sample_thi_kendes_for_ret = df.loc[df["thi_kendes_for_ret_antal"]>0,:].copy()
    df_sample_thi_kendes_for_ret["thi_kendes_for_ret_text"] = df_sample_thi_kendes_for_ret["html_concat"].progress_apply(find_thi_kendes_for_ret_parts)
    df["headers"] = df["html_concat"].progress_apply(retrieve_headers)
    
    # Merge the thi_kendes_for_ret sample onto the original df
    df_merged = pd.merge(df,df_sample_thi_kendes_for_ret[["id_verdict","thi_kendes_for_ret_text"]],left_on="id_verdict",right_on="id_verdict",how="left")
    # Remove the most spaceconsuming columns from the df in order to store it. 
    # Can be recreated by merging on original df
    df_merged = df_merged.drop(["html_concat","verdict_text"], axis=1)
    
    # Save dataframe as .parquet 
    Table = pa.Table.from_pandas(df_merged)
    pq.write_table(Table, "data/processed/pyarrow/UfR_thi_kendes_for_ret_partition.parquet")
    
    return 1

def retrieve_headers(string):
    soup = BeautifulSoup(string)
    headers = soup.find_all(["h1","h2","h3","h4"])
    headers = [x.text for x in headers]
    return headers 

def find_thi_kendes_for_ret_parts(string):
    soup = BeautifulSoup(string)
    list_of_strings = []
    for header in soup.find_all(["h1","h2","h3","h4"],string=re.compile(r"thi kendes for ret",re.IGNORECASE)):
        string = ""
        nextNode = header
        while True:
            nextNode = nextNode.nextSibling
            if nextNode is None:
                break
            if isinstance(nextNode, NavigableString):
                string += nextNode.strip()
            if isinstance(nextNode, Tag):
                if any(x in nextNode.name for x in ["h1","h2","h3","h4"]):
                    break
                string += nextNode.get_text(strip=True).strip()
        list_of_strings.append(string)
    return list_of_strings

if __name__=="__main__":
    main()