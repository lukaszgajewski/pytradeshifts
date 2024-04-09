"""This is a convenience script that runs all stages of the computation.
The goal is to produce a table of combined domestic supply of macronutrients
from all food items for each country before nuclear winter and every year during the winter.

The necessary input is:
1) Production and trade data from FAO ("All Data Normalized"):
https://www.fao.org/faostat/en/#data/QCL
https://www.fao.org/faostat/en/#data/TM
Put those in data/data_raw directory.
2) Crop reduction data due to nuclear winter (or whatever other scenario),
an example is given in: intmodel/data/crop_reduction_by_year.csv
Put this in intmodel/data/ directory.
3) Nutrition information for each product in the FAO data.
The assumed format is an ```*.xlsx``` file with a sheet titled
"Nutrition data from FAOSTAT" in which columns "A:E" contain the item name,
amount of Calories, protein, and fat (per kg) and whether it is an
outdoor growing crop or not (a 0/1 boolean; 1=outdoor growing crop).
Put that in ```intmodel/data```.

The output is going to be a lot of files you don't need to worry about unless
something goes wrong and intmodel/data/macros_csv.csv which is the main result.
"""

from src.preprocessing import main as format_production_trade
from create_scenario_files import main as convert_reduction_to_scenarios
from domestic_supply import main as compute_domestic_supply
from combine_ds_files import main as combine_domestic_supply_files
from create_crop_macros_csv import main as create_macros_csv
from combine_macros import main as combine_macros_files
from convert_macros_yearly_to_monthly import main as convert_yearly_to_monthly

# Step 1.
# WARNING: takes 2-3 hours; uses over 16GB of RAM
# This will produce two files for each food item in the FAOSTAT data set
# Those files will be put in the intmodel/data/prod_trade directory
# One file will have production data, the other trading matrix for an item
format_production_trade(
    "All_Data",
    year="Y2020",
)

# Step 2.
# The way the PyTradeShifts model is set up it requires a specific file format
# to apply the crop reduction to the data; this function produces the required
# files from the crop reduction input file.
# This should be rather quick.
convert_reduction_to_scenarios()

# Step 3.
# Note: takes ~15 min.
# this is "the number of FAO food items times number of years times number of countries"
# of operations. Some of it is vectorised but not all so it takes a while.
# It ends up being around *thousands* of files with 10 years of nuclear winter
# that we then also have to combine (next step).
# The files wil lbe in intmodel/data/domestic_supply directory.
# TODO: For now, this assumes "All_Data" and "Y2020" was passed in Step 1.
compute_domestic_supply()

# Step 4.
# This should be much quicker than previous stages.
# The result will be a combined domestic supply file for each "time step",
# i.e., before nuclear winter and at every year of the winter.
# The files will be in intmodel/data/domesti_supply_combined directory
combine_domestic_supply_files()

# Step 5.
# This should also be rather fast; from the combined DS files we now create a csv
# with columns: iso3, country name, kcals, fat, protein and each row is a country.
# There will be one file per each scenario (+ base / no scenario case) in the
# intmodel/data/macros directory
create_macros_csv()

# Step 6.
# The final stage (also quick); this combines all the files from the macros directory
# into one.
# WARNING: This assumes that the data is "yearly" as it does name filtering,
# so, e.g., in case of monthly data in this script replace all "year_" with
# "month_" and it should work. Other steps are agnostic to this.
combine_macros_files()

# Step 7. Optional.
# This will convert yearly data to monthly by simply dividing the values by 12
# and propagating columns such that instead of 10+1 columns we have 120+1
# (+1 is for the before scenario columns)
convert_yearly_to_monthly()
