import pandas as pd


def read_nutrition_data():
    # read data from Mike, values are per kg
    nutri_data = pd.read_excel(
        "intmodel/data/ALLFED Food consumption, supplies and balances.xlsx",
        sheet_name="Nutrition data from FAOSTAT",
        usecols="A:E",
        skiprows=1,
    )
    # drop NaNs
    nutri_data = nutri_data.dropna()
    # keep outdoor growing only;
    # WARNING/TODO: this was done manually and might need double checking
    nutri_data = nutri_data[nutri_data["Outdoor growing?"] == 1.0].reset_index(
        drop=True
    )
    # set Item to be the index
    nutri_data = nutri_data.set_index("Item")
    # drop the outdoor growing columns since it isn't needed anymore
    nutri_data = nutri_data.drop(columns="Outdoor growing?")
    # TODO: check if we're not double-counting, e.g.,
    # does "Fruit, primary" or "Citrus Fruit, Total" include other fruits
    # we count separately?
    return nutri_data


if __name__ == "__main__":
    nd = read_nutrition_data()
    print(nd)
    print(nd.index.is_unique)
