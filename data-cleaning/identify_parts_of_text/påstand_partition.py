"""This script produces a pd.DataFrame consisting of UfR id and all paragraphs of 
the text in the UfR that includes the substrings "paast" or "påst". 
The paragraphs are orderd as a list."""

import pandas as pd
from tqdm.auto import tqdm
from bs4 import BeautifulSoup
import re

def main():
    #config
    tqdm.pandas()
    #Load UfR html
    df = pd.read_parquet("./data/processed/pyarrow/UfR_text.parquet")    
    #Partition
    df["claims"]= df["html_concat"].progress_apply(get_claims)
    #Save 
    df[["id_verdict","claims"]].to_parquet("./data/processed/pyarrow/UfR_claims.parquet")
    
    return 1

def get_claims(string):
    soup = BeautifulSoup(string)
    paragraphs = soup.find_all("p")
    claims = [claim.text for claim in paragraphs if re.search("(paast|påst)", claim.text)]
    return claims 

if __name__=="__main__":
    main()
