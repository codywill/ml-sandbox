import numpy as np
import pandas as pd

# Update specific hypothesis s using sample of training data d
def update_s(s, d):
    for key in s.keys():
        if s[key] == '?':
            continue
        elif pd.isna(s[key]):
            s[key] = d[key]
        elif d[key] != s[key]:
            s[key] = '?'
    return s    

# Compose specific hypothesis s from training data df
def find_s(df):
    # Specific hypothesis initialization
    s = {key:np.nan for key in df.iloc[:, :-1].columns}
    
    # Update s for each row where the last column is 1
    for i, row in df.iterrows():
        if row.iloc[-1] == 1:
            s = update_s(s, row)
    return s

# Remove general hypotheses from G if they are less general than new specific hypothesis
def G_positive(G, s):
    for key in s.keys():
        for i, g in enumerate(G):
            if g[key] != '?' and g[key] != s[key]:
                del G[i]
    return G

# Recompile list of most general hypotheses given negative sample
def G_negative(G, s, d):
    new_G = []
    for key in s.keys():
        if s[key] != '?' and s[key] != d[key]:
            g = {key:'?' for key in s.keys()}
            g[key] = s[key]
            new_G.append(g)
    return new_G

def candidate_elimination(df):
    # Specific hypothesis s initialization
    s = {key:np.nan for key in df.iloc[:, :-1].columns}
    
    # General hypothesis g and set G initialization
    g = {key:'?' for key in df.iloc[:, :-1].columns}
    G = [g]
    
    # Update s for each row where the last column is 1, G for either case
    for i, row in df.iterrows():
        if row.iloc[-1] == 1:
            s = update_s(s, row)
            G = G_positive(G, s)
        else:
            G = G_negative(G, s, row)
    return G, s