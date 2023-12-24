import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class StockData:
  """Class to generate panel data using each columns; ret, prc, shrout

  Args:
    data: DataFrame (<- crsp-2.csv)

  Attributes:
    data: Dataframe of initial data
    ret : Dataframe of monthly return data (panel)
    prc : Dataframe of price data (panel)
    shrout : Dataframe of share outstanding data (panel)
    mktcap : Datafroma of market capitalization data (panel)
    date_index : index of datetime
  """
  def __init__(self, data):
    self.data = data
    self.ret = pd.pivot(self.data, values="ret", index="date", columns="permno")
    self.prc = pd.pivot(self.data, values="prc", index="date", columns="permno")
    self.shrout = pd.pivot(self.data, values="shrout", index="date", columns="permno")
    self.mktcap = self.ret * self.shrout
    self.date_index = self.get_datetime_index()

  def get_datetime_index(self):
    """Get datetime index list used in converting index into datetime

    Returns:
      list of datetime index
    """
    return self.ret.index.copy().tolist()
  
  def get_ret_til(self, when : int):
    """Get returns until the input time.

    Args:
      when: int of the index which indictaes time.

    Returns:
      return dataframe until the input time.
    """
    return self.ret.loc[:self.date_index[when]]

  def gen_weight_zeros(self):
    """Generate dataframe filled with zeros which has same size of return

    Returns:
      weight dataframe filled with zeros which has same size of self.ret
    """
    return pd.DataFrame(np.zeros_like(self.ret), index=self.ret.index, columns=self.ret.columns)

  def gen_return_zeros(self):
    """Generate dataframe filled with zeros which has same size of return

    Returns:
      portfolio return dataframe filled with zeros which has same size of self.ret
    """
    return pd.DataFrame(np.zeros(len(self.ret)), index=self.ret.index, columns=["pf_value"])
  
  def rebalance(self, portfolio_function, start : str, transaction_cost : float = 0):
    """Rebalance Portfolio by expanding window

    Args:
      portfolio_function: portfolio function which returns portfolio weight.
      start: start timestamp of the window. str : datetime
      transaction_cost: transaction cost. float; base is 0 for no cost

    Returns:
      pf_ret: dataframe of portfolio returns for backtesting period
    """
    start_condition = self.ret.index > pd.to_datetime(start)
    df_weight = self.gen_weight_zeros()
    pf_ret = self.gen_return_zeros()

    for cnt,idx in enumerate(self.ret.index):
      if cnt == 0:
          continue
      
      elif idx < pd.to_datetime(start):
        continue

      else:
        target_weight = portfolio_function(cnt-1)
        current_weight = (1+self.ret.loc[self.date_index[cnt-1]]) * df_weight.loc[self.date_index[cnt-1]]
        current_weight = current_weight / current_weight.sum()
        weight_difference = np.abs(target_weight - current_weight)

        df_weight.loc[self.date_index[cnt]] = target_weight
        pf_ret.loc[self.date_index[cnt],"pf_value"]  = (self.ret.loc[self.date_index[cnt]] * df_weight.loc[self.date_index[cnt]]).sum() 
        pf_ret.loc[self.date_index[cnt],"pf_value"] -= (weight_difference.sum() * transaction_cost)
        
    return pf_ret[start_condition]