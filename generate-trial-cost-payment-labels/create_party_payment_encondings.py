import pandas as pd
import regex as re
import string 

from collections import Counter
from functools import reduce
from tqdm.auto import tqdm

def main():
    # Config
    tqdm.pandas() 
    # Load data
    df_omkostninger = pd.read_parquet("data/processed/pyarrow/UfR_omkostninger.parquet")
    df_parties = pd.read_parquet("data/processed/pyarrow/UfR_parties.parquet")
    df_year = pd.read_parquet("data/processed/pyarrow/UfR_text.parquet").loc[:,["id_verdict","year"]]
    df_kendelse = pd.read_parquet("data/processed/pyarrow/UfR_kendelse.parquet")
    # Count brackets in defendants
    df_parties["defendant_representation_count"] = df_parties["defendant"].apply(lambda x: (Counter(x)["("]+Counter(x)[")"])/2)
    # Include relevant variables
    dfs = [df_year, df_parties.loc[:,["id_verdict","prosecutor","defendant", "count_of_conflicts","defendant_representation_count"]],df_omkostninger, df_kendelse]
    df = reduce(lambda x,y: pd.merge(x,y, left_on = "id_verdict", right_on = "id_verdict"), dfs)
    # Count how many sentences the substrings "sag" and "omkost" appears in together
    df["sagsomkostning_sætning_count"] = df["omkostninger"].apply(len)
    # Only consider cases from 1950 until today
    df["year"] = df["year"].astype(int) #formatting
    df = df.loc[df["year"]> 1949].copy() 

    ## Identify names/alias of party who recieves or pays the cost of trial
    # Identify party that pay cost of trials according to regex patterns OR identify if cost has been waived according to regex patterns
    df[["party_who_pays","party_who_pays_pattern"]] = df.loc[df["sagsomkostning_sætning_count"]>0].progress_apply(party_who_pays_list, axis=1, result_type="expand")
    # Identify party that recieves cost of trials from opponent according to regex patterns
    df[["party_who_recieves","party_who_recieves_pattern"]] = df.loc[df["sagsomkostning_sætning_count"]>0].progress_apply(party_who_recieves_list, axis=1, result_type="expand")

    ## Encode party names identified just above into DEF, PROC or NO_COST
    # List of alias and patterns
    defendant_alias_list = ["indstævnte", "sagsøgte", "tiltalte","indkærede"]
    prosecutor_alias_list = ["appellant","sagsøger","kærende","anmelderen"]
    no_cost_patterns = ["ophæve",
            "ingen\saf\sparterne",
            "hver\s(af\s)?(part(.*)?ære(r)?|ære(r)?part(.*)?)\s(sin|egn)(e)?\s(sags)?omkostninger",
            "hver\saf\sparterne",
            "hver\spart",
            "ingen\saf\ssagens\sparter",
            "ingen\spart\s"]
    som_nedenfor_anført_patterns = ["(som\snedenfor\sanført)","(som\snedenfor\snævnt)","som\snedenfor\sbestemt"]

    # Apply logical rules to identify who pays the cost of the trial
    df["party_who_pays_encoded"] = df.progress_apply(lambda x: encode_using_party_identified(x,"party_who_pays", defendant_alias_list,prosecutor_alias_list,anklagemyndigheds_conditional=True),axis=1)
    df["party_who_recieves_encoded"] = df.progress_apply(lambda x: encode_using_party_identified(x,"party_who_recieves", defendant_alias_list,prosecutor_alias_list,anklagemyndigheds_conditional=False),axis=1)
    df["no_cost"] = df.progress_apply(lambda x: encode_using_pattern(x,"party_who_pays_pattern",no_cost_patterns,"NO_COST"),axis=1)
    df["som_nedenfor_anført"] = df.progress_apply(lambda x: encode_using_pattern(x,"party_who_pays_pattern",som_nedenfor_anført_patterns,"NEDENFOR_ANFØRT"),axis=1)
    df["LABEL_party_who_pays"] = df.apply(encode_columns, axis=1)
    #remove  data
    df = df.loc[(df["count_of_conflicts"]<=1)&
        (df["defendant_representation_count"]<=1)&
        (df["sagsomkostning_sætning_count"]>0)&
        (df["not_kendelse"]==True)]

    df.to_excel("data/code_sagsomkostnigner.xlsx")
    df.to_parquet("data/processed/pyarrow/code_sagsomkostninger.parquet")
    
    return 1


# Regex patterns to match parties that pays
def party_who_pays(s):
    list_of_patterns = [
         #som nedenfor beskrevet
        "(som\snedenfor\sanført)",
        "(som\snedenfor\snævnt)",
        "som\snedenfor\sbestemt",
        #ophæve, hver part bære omkostninger
        "ophæve",
        "ingen\saf\sparterne",
        "hver\s(af\s)?(part(.*)?ære(r)?|ære(r)?part(.*)?)\s(sin|egn)(e)?\s(sags)?omkostninger",
        "hver\saf\sparterne",
        "hver\spart",
        "ingen\saf\ssagens\sparter",
        "ingen\spart\s",
        #betaler
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbetaler\si\ssagsomkostninger)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=betaler\stil)",
        "(?<=.*((sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostninger)).*betaler\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
        "(?<=(sagsomkostninger).*skal\s)(?!betale)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=.*betale)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sskal\sbetale\s)",
        "(?<!(.*)?til\s.*)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbetaler.*((sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostninger)))",
        "(?<=\sbetales\saf\s)([a-zA-Z0-9ÆØÅæøå]+)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\stilpligtedes\sat\sbetale)",
        "(?<=Det\sblev\spålagt).*(at\sbetale)",
        ".*(?=((\sdømmes\stil\sat)|\sdømtes|(\sbør)).*(betale).*((sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostninger)))",
        "(?<=efter\ssagens\sudfald\sskal\s).*(?=\s((betale\ssagens\somkostninger)|(betale\somkostninger\sfor\sagen)|(i\ssagsomkostninger)))",
        #udreder   
        "(?<=udredes\saf\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=(vil\shave\sat\sudrede|udreder)\s((sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostning)))",
        "(?<=(sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostninger)\sudreder\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
        #godtgøre
        "(?<=\svil\s).*(?=\shave\sat\sgodtgøre\s)",
        #findes at burde
        "((sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostninger)).*findes\s\K.*(?=\sat\sburde)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sfindes\sat\sburde\stilsvare)",
        "(?<=\svil\s).*(?=\shave\sat\sbetale)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=findes\sat\sburde\stilsvare)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sfindes\sat\sburde\sbetale)",
        #bør
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbør.*erstatte)",
        "(?<=i\ssagsomkostninger\s.*bør\s).*(?=\sbetale)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbør\sbetale\ssagens\somkostninger)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbør\si\ssagsomkostninger)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbør\s(til|derhos)\s.*(betale|godtgøre))",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\sbør(.*betale.*til|.*til.*betale))", 
        "(?<=((sagens\somkostninger)|(omkostninger\sfor\sagen)|(sagsomkostninger))\sbør\s).*(?=\sbetale)",
        #pålagdes
        "(?<=sagens\somkostninger.*\spålagdes\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
        "(?<=det\s(pålagdes|pålægges)\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=.*til)",
        "(?<=det\s(pålagdes|pålægges)\s).*(?=\sat\sbetale)"
        #udledes af
        "(?<=Sagens\somkostninger\s.*udledes\saf\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
        
        #vil have at
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\svil\shave\sat\stilsvare)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\svil\shave\sat\sbetale)",
        "((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)(?=\svil\shave\sat\sgodtgøre\s)"
        
    ]
    for pattern in list_of_patterns:
        s = re.sub("(-\s|--|–\s|––)?","",s)
        response = re.search(pattern, s, re.IGNORECASE)
        if response: 
            return response.group(), pattern
    return None, None

def party_who_recieves(s):
    list_of_patterns = [
         "(?<=(?!(ind))til\s(?!(dækning|betaling|følge|at)))((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
         #Tillagdes
         "(?<=\stillagdes\sder\s).*",
         #tilkendes
        "(?<=sagsomkostninger\stilkendtes\sder\s)((det\s|de\s)?[a-zA-Z0-9ÆØÅæøåü]+)",
    ]
    for pattern in list_of_patterns:
        s = re.sub("(-\s|--|–\s|––)?","",s)
        response = re.search(pattern, s, re.IGNORECASE)
        if response: 
            return response.group(), pattern
    return None, None


def party_who_pays_list(df):
    if len(df["omkostninger"])>0:
        for s in reversed(df["omkostninger"][-2:]):
            response, pattern = party_who_pays(s)
            if response:
                return response, pattern
            elif not any(w in s for w in ["dag","forrentes"]):
                return None, None
    else: 
        return None, None

def party_who_recieves_list(df):
    if len(df["omkostninger"])>0:
        for s in reversed(df["omkostninger"][-3:]):
            response, pattern = party_who_recieves(s)
            if response:
                return response, pattern
            elif not any(w in s for w in ["dag","forrentes"]):
                return None, None
    else: 
        return None, None

def clean_party(s):
    """Formatting.

    Args:
        s (string): Name entity that should be cleaned

    Returns:
        String: Cleaned string
    """
    if type(s)==str: 
        s = re.sub("\(.*\)","",s)   \
            .strip(" ")                \
            .translate(str.maketrans('', '', string.punctuation)) 
        return s
    else:
        return None

def encode_using_pattern(df,party_identified_pattern_column_name,list_of_patterns, encoding):
    """Encode the party identified in the sentence as the "prosecutor", "defendant" or "no cost"
    to pay the fees of the trial by using the pattern that was used for the match.

    Args:
        df (pandas.DataFrame): The dataframe containing the patterns used for matching
        party_identified_pattern_column_name (str): Name of the column that contains the matching pattern
        list_of_patterns (list(str)): List of patterns that are to be encoded 
        encoding (str): The value of the encoding, either "PROC", "DEF" or "NO_COST"

    Returns:
        encoding: returns the encoding if match or None if no match
    """
    pattern = df[party_identified_pattern_column_name]
    if pattern in list_of_patterns:
        return encoding
    else :
        return None
    
def encode_using_party_identified(df,party_identified_column_name,
                defendant_alias_list,
                prosecutor_alias_list,
                anklagemyndigheds_conditional):
    """Encode the party identified in either "PROC" or "DEF" using a set of logic rules.

    Args:
        df (_type_): DataFrame containing the name or alias party identified.
        party_identified_column_name (_type_): Column name.
        defendant_alias_list (_type_): List of alias that a defending party has.
        prosecutor_alias_list (_type_): List of alias that a prosecuting party has.

    Returns:
        str: DEF or PROC or NONE 
    """
    
    party_identified = clean_party(df[party_identified_column_name])
    defendant_name = clean_party(df["defendant"])
    prosecutor_name = clean_party(df["prosecutor"])
    if party_identified is None:
        return None
 
    #CHECK IF PART IS ANKLAGEMYNDIGHEDEN/RIGSADVOKATEN
    
    if anklagemyndigheds_conditional and type(prosecutor_name) == str:
        if re.search(("rigsadvokaten|anklagemyndigheden"),prosecutor_name,re.IGNORECASE):
            if re.search("offentlige|statskassen|stats\skassen",party_identified,re.IGNORECASE):
                return "PROC"
            else:
                return "DEF" 

    #CHECK IF PARTY IDENTIFIED AS-IS MATCHES ALIAS 
    if any(re.match(alias,party_identified, re.IGNORECASE) for alias in defendant_alias_list):
        return "DEF"
    if any(re.match(alias,party_identified, re.IGNORECASE) for alias in prosecutor_alias_list):
        return "PROC"

    #CHECK IF THE PARTY IDENTIFIED IS THE SAME AS THE DEFENDANTS OR PROSECUTOR NAME
    if not defendant_name: return None 
    if party_identified == defendant_name:
        return "DEF"
    if not prosecutor_name: return None    
    if party_identified == prosecutor_name:
        return "PROC"

    #CHECK EACH WORD IN PARTY IDENFIED IF THERE ARE MULTIPLE WORDS
    if len(party_identified_split:=party_identified.lower().split(" "))>1:
        for word in party_identified_split:
            if word in defendant_alias_list:
                return "DEF"
            if word in prosecutor_alias_list:
                return "PROC"

    #CHECK EACH WORD IN PROSECUT IF THERE ARE MULTIPLE WORDS
    if len(party_identified.split(" "))>1:
        pi = party_identified.split(" ")
    else: 
        pi = [party_identified]
    for word in pi:
        if len(word)>3:
            if re.search(word,defendant_name, re.IGNORECASE):
                return "DEF"    
            if re.search(word,prosecutor_name, re.IGNORECASE):
                return "PROC"
        elif word.isupper():
            if re.search(f"{word}[^a-zæøåA-ZÆØÅ]",defendant_name):
                return "DEF"    
            if re.search(f"{word}[^a-zæøåA-ZÆØÅ]",prosecutor_name):
                return "PROC"

def encode_columns(df):
    pays = df["party_who_pays_encoded"]
    receives = df["party_who_recieves_encoded"]
    no_cost = df["no_cost"]
    som_nedenfor_anført = df["som_nedenfor_anført"]
    
    if no_cost!= None:
        df["LABEL_party_who_pays"] = "NO_COST"
    elif som_nedenfor_anført != None:
        df["LABEL_party_who_pays"] = None
    elif pays == None and receives!=None:
        if receives == "PROC":
            df["LABEL_party_who_pays"] = "DEF"
        elif receives == "DEF":
            df["LABEL_party_who_pays"] = "PROC"    
    elif pays != None and receives==None:
        df["LABEL_party_who_pays"] = pays
    elif pays != None and receives!=None and receives!=pays:
        df["LABEL_party_who_pays"] = pays
    else:
        df["LABEL_party_who_pays"] = None
    
    return df["LABEL_party_who_pays"]
    

if __name__ == "__main__":
    main()