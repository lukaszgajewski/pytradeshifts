#


## PyTradeShifts
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L6)
```python 
PyTradeShifts(
   crop, base_year, percentile = 0.75
)
```


---
Class to build the trade matrix, calculate the trade shifts
and plot them. This combines all the methods that are needed
to easily run this from a jupyter notebook.


**Arguments**

* **crop** (str) : The crop to build the trade matrix for.
* **base_year** (int) : The base_year to extract data for. The trade communities
    are built relative to this year.
* **percentile** (float) : The percentile to use for removing countries with
    low trade.


**Returns**

None


**Methods:**


### .load_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L33)
```python
.load_data(
   crop: str, base_year: int
)
```

---
Loads the data into a pandas dataframe and cleans it
of countries with trade below a certain percentile.


**Arguments**

* **crop** (str) : The crop to build the trade matrix for.
* **base_year** (int) : The base_year to extract data for.


**Returns**

* **DataFrame**  : The trade data with countries with low trade removed
    and only the relevant crop.


### .remove_above_percentile
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L75)
```python
.remove_above_percentile(
   trade_matrix: pd.DataFrame, percentile: float = 0.75
)
```

---
Removes countries with trade below a certain percentile.


**Arguments**

* **crop_trade_data** (pd.DataFrame) : The trade data with countries with low trade removed
    and only the relevant crop.
* **percentile** (float) : The percentile to use for removing countries with
    low trade.


**Returns**

* **DataFrame**  : The trade data with countries with low trade removed
    and only the relevant crop.


### .remove_re_exports
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L105)
```python
.remove_re_exports()
```

---
Removes re-exports from the trade matrix.
This is a Python implementation of the R/Matlab code from:
Croft, S. A., West, C. D., & Green, J. M. H. (2018).
"Capturing the heterogeneity of sub-national production
in global trade flows."

Journal of Cleaner Production, 203, 1106–1118.

https://doi.org/10.1016/j.jclepro.2018.08.267

This implementation also includes pre-balancing to ensure that countries don't
export more than they produce and import.


**Arguments**

None


**Returns**

* **DataFrame**  : The trade matrix without re-exports.


### .prebalance
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L147)
```python
.prebalance(
   production_data: pd.Series, trade_matrix: pd.DataFrame, precision = 10**-3
)
```

---
Return prebalaced trading data.


**Arguments**

* **production_data** (pd.Series) : Vector of production data.
* **trade_matrix** (pd.DataFrame) : Trade matrix of the crop specified
* **precision** (float, optional) : Specifies precision of the prebalancing.


**Returns**

* **DataFrame**  : Prebalanced trade matrix.


### .remove_net_zero_countries
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L177)
```python
.remove_net_zero_countries(
   production_data: pd.Series, trade_matrix: pd.DataFrame
)
```

---
Return production and trading data with "all zero" countries removed.
"All zero" countries are such states that has production = 0 and sum of rows
in trade matrix = 0, and sum of columns = 0.


**Arguments**

* **production_data** (pd.Series) : Vector of production data.
* **trade_matrix** (pd.DataFrame) : Trade matrix of the crop specified


**Returns**

* Production data and trade matrix
without "all zero" countries.

### .correct_reexports
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L202)
```python
.correct_reexports(
   production_data: pd.Series, trade_matrix: pd.DataFrame
)
```

---
Return trading data after correcting for re-exports.

Input to this function should be prebalanced and have countries with all zeroes
removed.


**Arguments**

* **production_data** (pd.Series) : Vector of production data.
* **trade_matrix** (pd.DataFrame) : Trade matrix of the crop specified


**Returns**

* **DataFrame**  : Trade matrix without re-exports.

