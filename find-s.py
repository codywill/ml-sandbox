import numpy as np
import pandas as pd

# Arbitrary training data
df_a = pd.DataFrame(np.random.randint(0, 2, size=(5, 5)), columns=list('ABCDE'))

# EnjoySport example
columns = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast', 'EnjoySport']
samples = [
    ['Sunny','Warm','Normal','Strong','Warm','Same',1],
    ['Sunny','Warm','High','Strong','Warm','Same',1],
    ['Rainy','Cold','High','Strong','Warm','Change',0],
    ['Sunny','Warm','High','Strong','Cool','Change',1]
]
df_b = pd.DataFrame(samples, columns=columns)

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

print(df_a)
print(find_s(df_a))
print('\n')
print(df_b)
print(find_s(df_b))