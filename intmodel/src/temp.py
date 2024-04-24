import pandas as pd

t = pd.read_pickle("trade_data.pkl")
p = pd.read_pickle("production_data.pkl")
n = pd.read_csv("intmodel/data/nuclear_winter_csv.csv", index_col=0).index.unique()
t = set(t["Reporter ISO3"].unique()) | set(t["Partner ISO3"].unique())
p = set(p["iso3"].unique())
print(len(t))
print(len(p))
print(len(n))
print(n.difference(p))
print(n.difference(t))
