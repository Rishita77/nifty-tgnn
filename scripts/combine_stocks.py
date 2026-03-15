import pandas as pd
import glob

files = glob.glob("data/raw/stocks/*.csv")

dfs = []

for file in files:
    df = pd.read_csv(file)
    dfs.append(df)

allstocks = pd.concat(dfs, ignore_index=True)
allstocks.to_csv("allstock.csv", index=False)