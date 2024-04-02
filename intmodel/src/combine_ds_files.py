import pandas as pd
from os import listdir
from os.path import isfile, join
from read_nutrition_data import read_nutrition_data

ds_dir = "intmodel/data/prod_trade"
ds_files = [f for f in listdir(ds_dir) if isfile(join(ds_dir, f))]
items = set([f[: f.find("_")] for f in ds_files])
nd = read_nutrition_data()
print(sorted(items.intersection(nd.index)))

# for f in ds_files:
#     x = pd.read_csv(ds_dir + "/" + f, header=0, names=["Country", "Domestic Supply"])
#     print(x)
#     break
