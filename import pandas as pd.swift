import pandas as pd


df = pd.read_csv("autotagging_genre.tsv", sep="\t")
print(df.columns)
print(df.head(5))


