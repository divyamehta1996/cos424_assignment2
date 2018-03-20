import pandas as pd
df = pd.read_csv('output.csv', header=0, low_memory=False)

with open('constantVariables.txt', 'r') as file:
    for line in file:
        word = line.split()[0]
        df.drop(word, 1)

df.to_csv('noConstants.csv', index=False)
