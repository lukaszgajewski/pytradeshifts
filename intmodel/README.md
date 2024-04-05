# Preprocessed data for the integrated model

Considering the amount of data they are ```.gitingored```. 

Running preprocessing will produce them (warning: takes several hours [2-3] and a lot of RAM).

Other steps are pretty quick (seconds or minutes).

2-3hrs again for calculating domestic supply for all scenarios, this is the number of FAO food items times number of months times number of countries of operations.
Some of it is vectorised but not all so it takes a while.
It ends up being around 20 thousand files that we then also have to combine.

In the output of the domestic supply script not all messages like "Almonds, with shell production/trade or crop_reduction_month_74.csv data not found, skipping." are errors. Almonds, with shell are simply no longer in the FAO data (now it is "in shell"), so such logs need careful considerations.

# TODO
- write code for generating domestic supply in a scenario (basic functionality is there, only handling file names is left to do)
- write a script automating the procedure
- write docs and instructions on how to use
- scan repo for TODOs
- check ds gen output for errors
- integrate into intmodel repo somehow, write now we use importutilities and create_crop_macro from intmodel which is a bit messy
- crop_reduction_by_month.csv is not correct right now it is a placeholder file.
- nutrition data needs verification

# Running

1. PyTradeShifts.preprocessing; produces production and trade data (prod_trade directory)
2. domestic_supply.py; produces domestic supply values for all items (with or without a reduction factor, this can also in the future incorporate trade difficulty factor due to gravity model of trade built into PyTradeShifts)
3. combine_ds_files.py; produces domesti_supply_combined.csv
4. create_crop_macros_csv.py; produces domestic supply of *calories* for each country, just like the OG script in integrated model repo did for production values