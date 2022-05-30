from bs4 import BeautifulSoup
import re
import pandas as pd
from tqdm.auto import tqdm
from nltk import sent_tokenize

def main():
    #config
    tqdm.pandas()
    #Load UfR html
    df = pd.read_parquet("./data/processed/pyarrow/UfR_text.parquet")    
    #Partition
    df["omkostninger"]= df["verdict_text"].progress_apply(retrieve_omkostninger)
    #Save 
    df[["id_verdict","omkostninger"]].to_parquet("./data/processed/pyarrow/UfR_omkostninger.parquet")

    return 1

def retrieve_omkostninger(string):
    pattern_omkost = re.compile("(omkost)", re.IGNORECASE)
    pattern_sag = re.compile("(sag)", re.IGNORECASE)
    omkostning_list = [paragraph for paragraph in sent_tokenize(string, language="danish")[-10:] if re.search(pattern_omkost,paragraph)]  
    sagsomkostning_list = [paragraph for paragraph in omkostning_list if re.search(pattern_sag,paragraph)]
    return sagsomkostning_list

if __name__=="__main__":
    main()