import algorithms as ag
import pandas as pd
import numpy as np

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

#print(df_a)
#print(ag.find_s(df_a))
#print('\n')
#print(df_b)
#print(ag.find_s(df_b))

print(ag.candidate_elimination(df_b))