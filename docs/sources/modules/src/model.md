#


## PyTradeShifts
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L6)
```python 
PyTradeShifts(
   crop, base_year, percentile = 0.75, region = 'Global', testing = False
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
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L35)
```python
.load_data()
```

---
Loads the data into a pandas dataframe and cleans it
of countries with trade below a certain percentile.


**Arguments**

None


**Returns**

None

### .remove_above_percentile
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L79)
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


### .prebalance
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L110)
```python
.prebalance(
   precision = 10**-3
)
```

---
This implementation also includes pre-balancing to ensure that countries don't
export more than they produce and import.


**Arguments**

* **precision** (float, optional) : Specifies precision of the prebalancing.


**Returns**

None

### .remove_net_zero_countries
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L144)
```python
.remove_net_zero_countries()
```

---
Return production and trading data with "all zero" countries removed.
"All zero" countries are such states that has production = 0 and sum of rows
in trade matrix = 0, and sum of columns = 0.


**Arguments**

None


**Returns**

None

### .correct_reexports
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/model.py/#L172)
```python
.correct_reexports()
```

---
Removes re-exports from the trade matrix.
This is a Python implementation of the R/Matlab code from:
Croft, S. A., West, C. D., & Green, J. M. H. (2018).
"Capturing the heterogeneity of sub-national production
in global trade flows."

Journal of Cleaner Production, 203, 1106–1118.

https://doi.org/10.1016/j.jclepro.2018.08.267


Input to this function should be prebalanced and have countries with all zeroes
removed.


**Arguments**

None


**Returns**

None
