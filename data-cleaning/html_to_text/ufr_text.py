"""NOT IN USE"""

import os
from tqdm.auto import tqdm
import pandas as pd
import re
import pandas as pd
from bs4 import BeautifulSoup
os.chdir("C:/Users/espen/Documents/SDS/thesis")

tqdm.pandas()

def main():
    #Load data
    df_html = pd.read_parquet("data/processed/pyarrow/verdicts_in_UfR_only_articles_with_html")
    df_html["id_verdict"] = df_html["id"].apply(lambda x: re.sub("\$[0-9]*","",x))
    #Get text from html
    df_html["verdict_text"] = df_html["html_data"].progress_apply(lambda x: retrieve_html_data(x))
    #Concatenate multiple-page judgements
    df = df_html.copy()
    func = lambda x: "\n".join(x)
    df = df.loc[:,["verdict_text","id_verdict"]].groupby("id_verdict").agg(func)
    df = pd.merge(df,df_html, left_on="id_verdict",right_on="id_verdict")
    df = df.drop_duplicates("id_verdict")
    #Remove irrelevant data
    df = df.drop(["id","karnov_pagenation","id_dir_compliant","next_page","prev_page", "url","html_data", "directory"], axis=1)
    #Save UfR_text.parquet
    df.to_parquet("data/processed/pyarrow/UfR_text_keep_linebreaks.parquet")

def retrieve_html_data(string):
    if type(string) == str:
        soup = BeautifulSoup(string, features="lxml")
        s = soup.find("div", {"class":"maincontent"})
        if s is None: return ""
        s = s.get_text(separator=' ')
        s = re.sub(" +"," ",s)
        s = s.split(" Tidsskrifter Ugeskrift for Retsvæsen",1)[0]
    else:
        s = ""
    return s 

if __name__ == "__main__":
    main()