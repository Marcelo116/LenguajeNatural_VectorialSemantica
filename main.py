import pandas as pd

df = pd.read_csv("Restaurant_Reviews.tsv", sep="\t") #Solo incorporamos la separación \t
df.head()