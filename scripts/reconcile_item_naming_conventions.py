from src.utils import jaccard_index
import os
import pandas as pd

prefix = f"data{os.sep}preprocessed_data{os.sep}integrated_model"
path = prefix + f"{os.sep}prod_trade"
suffix = "_Y2020_Global_production.csv"  # TODO: generalise for any year/region
supply_items = set(
    [
        f[: -len(suffix)]
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and "production" in f
    ]
)
allowed_items = set(
    pd.read_csv(prefix + f"{os.sep}from_IM{os.sep}FAOSTAT_food_production_2020.csv")[
        "Item"
    ].values
)
missing_items = allowed_items.difference(supply_items)
guessed_matches = {
    mis_item: max(supply_items, key=lambda it: jaccard_index(it, mis_item))
    for mis_item in missing_items
}
# wrongle guessed found manually with:
# for mi, gm in guessed_matches.items():
#     print(mi, "==", gm)
#     if input("yay? ") == "n":
#         wrongly_guessed.append(mi)
# print(wrongly_guessed)
wrongly_guessed = [
    "Yautia (cocoyam)",
    "Fruit, stone nes",
    "Hemp tow waste",
    "Lentils",
    "Bastfibres, other",
    "Cloves",
    "Fruit, citrus nes",
    "Chillies and peppers, dry",
    "Hops",
    "Cabbages and other brassicas",
    "Spices nes",
    "Coconuts",
    "Vegetables, fresh nes",
    "Rubber, natural",
    "Anise, badian, fennel, coriander",
    "Garlic",
    "Seed cotton",
    "Rice, paddy (rice milled equivalent)",
    "Beans, green",
    "Flax fibre and tow",
    "Ramie",
    "Fibre crops nes",
    "Cinnamon (cannella)",
    "Berries nes",
    "Pulses nes",
    "Onions, dry",
    "Maize, green",
    "Fruit, tropical fresh nes",
    "Maize",
    "Vegetables, leguminous nes",
    "Sugar crops nes",
    "Chillies and peppers, green",
    "Taro (cocoyam)",
    "Rapeseed",
    "Jojoba seed",
    "Fruit, pome nes",
    "Grapefruit (inc. pomelos)",
    "Nuts nes",
    "Oilseeds nes",
    "Tallowtree seed",
    "Ginger",
    "Fruit, fresh nes",
    "Plantains and others",
    "Melons, other (inc.cantaloupes)",
    "Groundnuts, with shell",
    "Manila fibre (abaca)",
    "Rice, paddy",
    "Carobs",
]
guessed_matches = {k: v for k, v in guessed_matches.items() if k not in wrongly_guessed}
print(guessed_matches)
