# Preprocessed data for the integrated model

This sub-modul produces a table of combined domestic supply of macronutrients from all food items for each country before nuclear winter and every month during said winter.

The necessary input is:
1) Production and trade data from FAO ("All Data Normalized"):
https://www.fao.org/faostat/en/#data/QCL
https://www.fao.org/faostat/en/#data/TM

Put those in data/data_raw directory (note: *not here*, the main ```data``` folder, just like you would do for the PyTradeShifts model).

2) Crop reduction data due to nuclear winter (or whatever other scenario you're interested in).

Put this in ```intmodel/data``` directory.

An example is provided: ```intmodel/data/crop_reduction_by_year.csv```.

3) Nutrition information for each product in the FAO data.

The assumed format is an ```*.xlsx``` file with a sheet titled "Nutrition data from FAOSTAT" in which columns "A:E" contain the item name, amount of Calories, protein, and fat (per kg) and whether it is an outdoor growing crop or not (a 0/1 boolean; 1=outdoor growing crop).
Put that in ```intmodel/data```. Currently we do not provide this file but this should be subject to change once we're out of the prototype phase.


The output is going to be a lot of files you don't need to worry about unless
something goes wrong, and ```intmodel/data/macros_csv.csv``` which is the main result that we will then want to use in the [integrated model](https://github.com/allfed/allfed-integrated-model).

Considering the amount of data they are mostly #```.gitingored```. 

The whole procedures takes around 3.5 hours on Intel Xeon E3-1200 and takes 16+ GB RAM, and about 0.5GB of disk space.
It also produces tens of thousands of intermidiary files so there's quite a lot of IO happening.
If you don't have an SSD it will take longer.
We use IO instead of putting everything into RAM for two reasons:
- the author of the code doesn't have that much RAM
- it takes several hours to compute and if something goes wrong we would lose all progress; with IO we keep track item by item

In the output logs of the domestic supply script (```intmodel/src/domestic_supply.py```) not all messages are critical errors.
For example, "Almonds, with shell production/trade or crop_reduction_month_74.csv data not found, skipping." is listed as an error *but* the item "Almonds, with shell" is simply no longer in the FAO data (now it is called "Almonds, in shell" instead), so not all errors need intervention.

# Running

Make sure you follow the main README first. 
You will need your environment set up correctly such that you can use PyTradeShifts.

Create these directories:
- ```intmodel/data/domestic_supply```
- ```intmodel/data/domestic_supply_combined```
- ```intmodel/data/macros```
- ```intmodel/data/prod_trade```
- ```intmodel/data/scenario_files```

Then, the simplest way is to run :

```python inmodel/src/compute_domestic_supply.py```,

but be warned: this can take 3+ hours.

See the script to learn more, it has each computation stage described, and you can just comment out the parts you've already done or simply want to skip.

# TODO
- scan repo for TODOs
- check ds gen output for errors
- integrate into the integrated model
- crop_reduction_by_month.csv is not correct right now; it is a placeholder file.
- nutrition data needs verification (which is outdoor crop and which isn't)
- some tests would be nice
- some scripts have hard-coded inputs which is fine in the prototype phase but eventually we should refactor that into proper parametrisation
