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
   crop, base_year, percentile
)
```

---
Loads the data into a pandas dataframe and cleans it
of countries with trade below a certain percentile.


**Arguments**

* **crop** (str) : The crop to build the trade matrix for.
* **base_year** (int) : The base_year to extract data for.
* **percentile** (float) : The percentile to use for removing countries with
    low trade.


**Returns**

* **DataFrame**  : The trade data with countries with low trade removed
    and only the relevant crop.


### .remove_above_percentile
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L75)
```python
.remove_above_percentile(
   crop_trade_data, percentile
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


### .build_trade_matrix
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L93)
```python
.build_trade_matrix()
```

---
Builds the trade matrix for the given crop and base_year.


**Arguments**

None


**Returns**

* **DataFrame**  : The trade matrix.


### .remove_re_exports
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L118)
```python
.remove_re_exports()
```

---
Removes re-exports from the trade matrix.
This is a Python implementation of the Matlab code from:
Croft, S. A., West, C. D., & Green, J. M. H. (2018).
"Capturing the heterogeneity of sub-national production
in global trade flows."

Journal of Cleaner Production, 203, 1106–1118.

https://doi.org/10.1016/j.jclepro.2018.08.267


**Arguments**

None


**Returns**

* **DataFrame**  : The trade matrix without re-exports.


### .balance_trade_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L150)
```python
.balance_trade_data(
   trade_matrix, production_data, tolerance = 0.001, max_iterations = 10000
)
```

---
Balance the trade data using an iterative approach.


**Args**

* **trade_matrix** (numpy.ndarray) : The bilateral trade matrix.
* **production_data** (numpy.ndarray) : A vector of production values for each country.
* **tolerance** (float, optional) : A tolerance threshold for trade imbalances. Defaults to 0.001.
* **max_iterations** (int, optional) : The maximum number of iterations. Defaults to 100.


**Returns**

* **ndarray**  : The balanced bilateral trade matrix.

