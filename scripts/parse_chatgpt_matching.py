import os

with open(
    f"data{os.sep}preprocessed_data{os.sep}integrated_model{os.sep}chatgpt_item_matching.txt",
    "r",
) as f:
    matching_dict = {}
    found_first_line = False
    for line in f:
        if line[:2] == "1.":
            found_first_line = True
        if found_first_line is False:
            continue

        line = line[line.find(".") + 2 : -1].split(" - ")
        if len(line) != 2:
            print(line)
            break

        matching_dict[line[0]] = line[1]

prefix = f"data{os.sep}preprocessed_data{os.sep}integrated_model"
path = prefix + f"{os.sep}prod_trade"
suffix = "_Y2020_Global_production.csv"
supply_items = set(
    [
        f[: -len(suffix)]
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and "production" in f
    ]
)

assert len(supply_items.intersection(matching_dict.keys())) == len(supply_items)
