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
    #config
    tqdm.pandas()
    #Load UfR html
    df = pd.read_parquet("./data/processed/pyarrow/UfR_text.parquet")    
    #Partition
    df["sagsomkostning"]= df["html_concat"].progress_apply(get_cost_of_trial)
    #Save 
    df[["id_verdict","sagsomkostning"]].to_parquet("./data/processed/pyarrow/UfR_sagsomkostning.parquet")
    
    return 1

def get_cost_of_trial(string):
    soup = BeautifulSoup(string)
    paragraphs = soup.find_all("p")
    
    claims = [claim.text for claim in paragraphs if re.search("(sagsomkostning|omkostrninger)", claim.text)]
    return claims 
if __name__=="__main__":
    main()



    