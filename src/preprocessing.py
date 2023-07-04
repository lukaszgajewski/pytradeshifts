import pandas as pd
import os

trade_matrix = pd.read_csv("." + os.sep + "data" + os.sep + "FAO_trade_matrix" + os.sep + "Trade_DetailedTradeMatrix_E_Oceania"+ os.sep + "Trade_DetailedTradeMatrix_E_Oceania.csv")
print(trade_matrix.head())