import pandas as pd
df = pd.read_csv('output.csv', header=0, low_memory=False)
f = df.columns.tolist()
print len(f)
with open('constantVariables.txt', 'r') as file:
    for line in file:
        word = unicode(line.split()[0], "utf-8")
        f.remove(word)
	#print len(f)
#	df.drop(word, 1)
keep = df[f]
print len(keep.columns)
keep.to_csv('noConstants.csv', index=False)
