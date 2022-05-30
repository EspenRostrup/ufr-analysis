import re
import pandas as pd

def main():
    df_headers = pd.read_parquet("./data/processed/pyarrow/UfR_thi_kendes_for_ret_partition.parquet") 
    df_headers = df_headers.loc[:,["id_verdict","headers"]].copy()
    df_headers = pd.merge(df_headers["id_verdict"], df_headers["headers"].apply(ikkeKendelse), left_index=True, right_index=True)
    df_headers.rename(columns={"headers":"not_kendelse"}, inplace=True)
    df_headers.loc[:,["id_verdict","not_kendelse"]].to_parquet("./data/processed/pyarrow/UfR_kendelse.parquet")


def ikkeKendelse(li):
    for header in li:
        if re.search("\.?K\.",header):
            return False
    if len(li)!=0:
        return True
    else:
        return None

if __name__ == "__main__":
    main()
