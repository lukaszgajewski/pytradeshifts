#


### rename_item
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L14)
```python
.rename_item(
   item
)
```

---
Renames specific item entries for readability.


**Args**

* **item** (str) : The item name.


**Returns**

* **str**  : The renamed item name.


----


### read_faostat_bulk
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L31)
```python
.read_faostat_bulk(
   faostat_zip: str
)
```

---
Return pandas.DataFrame containing FAOSTAT data extracted from a bulk
download zip file.
This is based on the following R implementation:
https://rdrr.io/cran/FAOSTAT/src/R/faostat_bulk_download.R#sym-read_faostat_bulk


**Arguments**

* **faostat_zip** (str) : Path to the FAOSTAT zip file.


**Returns**

* **DataFrame**  : The FAOSTAT data.


----


### serialise_faostat_bulk
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L52)
```python
.serialise_faostat_bulk(
   faostat_zip: str
)
```

---
Read FAOSTAT data from a bulk download zip file as a pandas.DataFrame,
and save it as a pickle to allow for faster loading in the future.


**Arguments**

* **faostat_zip** (str) : Path to the FAOSTAT zip file.


**Returns**

None

----


### _melt_year_cols
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L72)
```python
._melt_year_cols(
   data: (pd.Series|pd.DataFrame)
)
```

---
Filter out unnecessary columns from the data and melt the year columns.


**Arguments**

* **data** (pd.Series | pd.DataFrame) : The data to be melted.


**Returns**

* **DataFrame**  : The melted data.


----


### _prep_trade_matrix
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L97)
```python
._prep_trade_matrix(
   trade_pkl: str, item: str, unit = 'tonnes', element = 'ExportQuantity',
   year = 'Y2021'
)
```

---
Return properly formatted trade matrix.


**Arguments**

* **trade_pkl** (str) : Path to the trade matrix pickle file.
* **item** (str) : Item to filter for.
* **unit** (str) : Unit to filter for.
* **element** (str) : Element to filter for.
* **year** (str) : Year to filter for.


**Returns**

* **DataFrame**  : The trade matrix.

---
Notes:
    The optional arguments must be determined semi-manually as their allowed values
    depend on particular datasets. E.g., unit can be "tonnes" in one file and "t"
    in another.

----


### _prep_production_vector
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L139)
```python
._prep_production_vector(
   production_pkl: str, item = 'Wheat', unit = 't', year = 'Y2021'
)
```

---
Return properly formatted production vector.


**Arguments**

* **production_pkl** (str) : Path to the production vector pickle file.
* **item** (str) : Item to filter for.
* **unit** (str) : Unit to filter for.
* **year** (str) : Year to filter for.


**Returns**

* **DataFrame**  : The production vector.

---
Notes:
    The optional arguments must be determined semi-manually as their allowed values
    depend on particular datasets. E.g., unit can be "tonnes" in one file and "t"
    in another.

----


### _unify_indices
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L169)
```python
._unify_indices(
   production_vector: pd.DataFrame, trade_matrix: pd.DataFrame
)
```

---
Return the production (as a Series) and trade matrix (DataFrame) with
unified (i.e., such that they match each other),
and sorted indices/columns.
Missing values are replaced by 0.


**Arguments**

* **production_vector** (pd.DataFrame) : The production vector.
* **trade_matrix** (pd.DataFrame) : The trade matrix.


**Returns**

* The production vector and trade matrix
    with unified indices/columns.


----


### format_prod_trad_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L194)
```python
.format_prod_trad_data(
   production_pkl: str, trade_pkl: str, item: str, production_unit = 't',
   trade_unit = 'tonnes', element = 'ExportQuantity', year = 'Y2021'
)
```

---
Return properly formatted production vector (as a Series),
and trade matrix (DataFrame).


**Arguments**

* **production_pkl** (str) : Path to the production vector pickle file.
* **trade_pkl** (str) : Path to the trade matrix pickle file.
* **item** (str) : Item to filter for.
* **production_unit** (str) : Unit to filter for in the production vector.
* **trade_unit** (str) : Unit to filter for in the trade matrix.
* **element** (str) : Element to filter for in the trade matrix.
* **year** (str) : Year to filter for.


**Returns**

* The production vector and trade matrix.

---
Notes:
    The optional arguments must be determined semi-manually as their allowed values
    depend on particular datasets. E.g., unit can be "tonnes" in one file and "t"
    in another.
