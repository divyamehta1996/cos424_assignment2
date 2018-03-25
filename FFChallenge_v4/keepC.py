import pandas as pd
df = pd.read_csv('noConstants.csv', header=0, low_memory=False)
f = df.columns.tolist()
print len(f)
keepWords = []
for each in f:
    if each[0] == 'c':
    	if each[1] == 'f' or each == "challengeID":
        	keepWords.append(each)
print keepWords
keep = df[keepWords]
print len(keep.columns)
keep.to_csv('onlyFather.csv', index=False)
