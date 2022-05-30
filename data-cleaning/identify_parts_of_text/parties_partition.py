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
    df = df[["id_verdict","html_concat"]]
    
    #Partition
    df["parties"] = df["html_concat"].progress_apply(get_parties)
    ##Expand parties into columns (and count of parties)
    df_merged = pd.merge(df,df["parties"].apply(pd.Series), left_index=True,right_index=True)
    
    #Save 
    df_merged.to_parquet("./data/processed/pyarrow/UfR_parties.parquet")        
    

def get_parties(string):
    s = BeautifulSoup(string, features="lxml") 
    #Remove abstract to find parties
    abstract = s.find("div", {"class":"abstract"})
    sagsbesk = s.find("div", {"class":"SAGSBESK"})
    N2_ann_zoomable_clippable_notable = s.find("div", {"class":"N2 ann zoomable clippable notable"})
    if abstract: abstract.extract()
    if sagsbesk: sagsbesk.extract()
    if N2_ann_zoomable_clippable_notable: N2_ann_zoomable_clippable_notable.extract()
    
    paragraphs = s.find_all("p")
    
    dic = {}
    if paragraphs:
        paragraphs_containing_word = [paragraph.text for paragraph in paragraphs if re.search("( mod | contra )", paragraph.text, flags=re.IGNORECASE)]
        if paragraphs_containing_word:
            parties = paragraphs_containing_word[0].split("mod")
            if len(parties)<2:
                parties = paragraphs_containing_word[0].split("contra")
            parties = [party.strip() for party in parties]
            dic["count_of_conflicts"] = len(parties) - 1
            if len(parties)==2:
                dic["prosecutor"] = parties[0]
                dic["defendant"] = parties[1]
                
    return dic


if __name__=="__main__":
    main()
