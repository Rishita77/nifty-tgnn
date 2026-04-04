import pandas as pd
import glob

# files = glob.glob("data/raw/stocks/*.csv")

# dfs = []

# for file in files:
#     df = pd.read_csv(file)
#     dfs.append(df)

# allstocks = pd.concat(dfs, ignore_index=True)
# allstocks.to_csv("allstock.csv", index=False)

df = pd.read_csv("allstock.csv")
df = df.drop(columns=["Unnamed: 0"])
df = df.drop(columns=["Adj Close"])

df.to_csv("allstock.csv", index=False)

print(df.head())