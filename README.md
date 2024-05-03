# Caloric domestic supply
Code in this repo computes the domestic supply of calories, with potential for other nutrients, for a list of countries, food items, and in a specified yield reduction scenario.

This is for the purposes of the ALLFED integrated model and will eventually get absorbed into that repostitory. 
While this is using core ideas of the pytradeshifts project it is not part of it, and this branch shall be deleted in the future. 

Runs in under 100[s] on an Intel Core i7-8550U @ 1.80GHz with 16GB of RAM

## Data
All paths can be modified easily in ```src/input_output.py```.
### Input
These files are raw inputs that must be provided for the code to work.
- ```data/input/Trade_DetailedTradeMatrix_E_All_Data.zip```; this is the data from: https://www.fao.org/faostat/en/#data/TM, bulk download -> all data.
- ```data/input/Production_Crops_Livestock_E_All_Data.zip```; this data is from: https://www.fao.org/faostat/en/#data/QCL.
- ```data/input/country_codes.csv```; country codes as specified by the FAO. We need conversion from FAO code to ISO3. This file can be obtained by going to either of the datasets above,and clicking "Definitions and standards", then "Country/Region", and finally the download button. There's probably an easier way but I haven't found it.
- ```data/input/primary_crop_nutritional_data.csv```; this is a list of food items and their nutritional value per [kg]. The list provided in this repo is thanks to the hard work of Mike Hinge. Based on FAO nutrition data: https://web.archive.org/web/20190215000000*/https://www.fao.org/3/X9892E/X9892e05.htm.
- ```data/input/nuclear_winter_csv.csv```; yield reduction data -- we consider a nuclear winter scenario, and the data is from Xia et al., https://www.nature.com/articles/s43016-022-00573-0.
- ```data/input/seasonality_csv.csv```; this is the fraction of yield per country at each month. Also the result of the hard work of Mike Hinge; based on data from: https://apps.fas.usda.gov/psdonline/app/index.html#/app/home.

### Intermidiary
These are intermidiary files created during running the whole procedure. The idea is that, e.g., if we change the scenario (yield reduction) file we don't have to recompute everything.
- ```data/intermediary/trade_data.pkl```; this is trading data of food items as per the nutrional data for each country that is present in the yield reduction data, with all unnecessary data removed.
- ```data/intermediary/production_data.pkl```; like the trade data but for production.
- ```data/intermediary/total_caloric_trade.csv```; this is the trade data above, converted to calories, summed up and put into a matrix.
- ```data/intermediary/total_caloric_production.csv```; as above but for production, and put into a vector.

### Output
The final output files, i.e., domestic supply of dry caloric tonnes per country per year/month.
- ```data/output/domestic_supply_kcals.csv```
- ```data/output/domestic_supply_kcals_monthly.csv```

## Code
- ```src/input_output.py```; controls data paths and loading the zips.
- ```src/fao_trade_zip_to_pkl.py```; reads in the FAO zip trade file and serialises it to a pickle.
- ```src/fao_production_zip_to_pkl.py```; reads in the FAO zip production file and serialises it to a pickle.
- ```src/fao_pkl_to_total_caloric_trade_and_production.py```; from said pickles, computes total caloric trade and production for each country (and saves results to CSVs).
- ```src/compute_domestic_supply.py```; from the above, computes the domestic supply of dry caloric tonnes for each country yearly, in the context of the specified yield reduction scenario.
- ```src/convert_supply_yearly_to_monthly.py```, from the above, computes the domestic supply of dry coloric tonnes monthly, in the context of the specified seasonality data.
- ```src/main.py```; runs all of the above scripts in the proper order.

## Launch
All data in the ```input``` subsection must be put in the right places (see Data section).

Then, run ```python src/main.py```. 

That's it. The results will be in ```data/output/```.

## Requirements
- pandas
- tqdm
- pytests

## TODO
- env/req files
- update this readme with installation instructions
- tests