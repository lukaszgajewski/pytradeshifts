# Preprocessed data for the integrated model

Considering the amount of data they are ```.gitingored```. 

Running preprocessing will produce them (warning: takes several hours and a lot of RAM).

Generating domesctic supply is pretty quick (seconds).

# TODO
- combine all domestic supply files into one
- modify creat_crop_macros code fron intmodel to work with domestic supply and new nutrition table
- write code for generating domestic supply in a scenario (basic functionality is there, only handling file names is left to do)
- write docs and instructions on how to use
- scan repo for TODOs
- check ds gen output for errors
- integrate into intmodel repo somehow, write now we use importutilities and create_crop_macro from intmodel which is a bit messy

# Running

1. PyTradeShifts.preprocessing; produces production and trade data (prod_trade directory)
2. domestic_supply.py; produces domestic supply values for all items (with or without a reduction factor, this can also in the future incorporate trade difficulty factor due to gravity model of trade built into PyTradeShifts)
3. combine_ds_files.py; produces domesti_supply_combined.csv
4. create_crop_macros_csv.py; produces domestic supply of *calories* for each country, just like the OG script in integrated model repo did for production values