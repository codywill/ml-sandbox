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
