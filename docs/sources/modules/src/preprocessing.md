#


### read_in_raw_trade_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L5)
```python
.read_in_raw_trade_data(
   testing = False
)
```

---
Reads in the raw trade matrix from the data folder and returns it as a pandas dataframe.
The raw trade matrix can be found at: https://www.fao.org/faostat/en/#data/TM
Select "All Data" for download.


**Arguments**

* **testing** (bool) : Checks to only use a subset of the data for testing purposes.


**Returns**

* **trade_data** (pd.DataFrame) : The raw trade matrix as a pandas dataframe.


----


### read_in_raw_production_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L39)
```python
.read_in_raw_production_data()
```

---
Reads in the raw food production to be used later for the
re-export algorithm.


**Returns**

* **DataFrame**  : The raw food production data.


----


### extract_relevant_trade_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L59)
```python
.extract_relevant_trade_data(
   trade_data, items, year = 2021
)
```

---
Extracts only the relevant data needed for building the trade model.


**Args**

* **trade_data** (pd.DataFrame) : The raw trade matrix.
* **year** (int) : The year to extract data for.
* **items** (list) : The items of interest, i.e., trade goods.


**Returns**

* **DataFrame**  : The cleaned trade matrix.


----


### extract_relevant_production_data
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L105)
```python
.extract_relevant_production_data(
   production_data, items, year = 2021
)
```

---
Extracts only the relevant data for the re-export algorithm.


**Args**

* **production_data** (pd.DataFrame) : The raw production data.
* **items** (list) : The items of interest, i.e., trade goods.
* **year** (int) : The year to extract data for.


**Returns**

* **DataFrame**  : The cleaned production data.


----


### rename_item
[source](https://github.com/allfed/My-Super-Cool-Respository/blob/master/src/preprocessing.py/#L150)
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

