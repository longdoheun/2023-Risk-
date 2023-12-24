from .stockdata import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp

class Portfolio(StockData):
  """Class for each portfolio weight according to window sizes

  Type of portfolios
  (1) value_weight: value-weight portfolio
  (2) equal_weight: equal-weight portfolio 
  (3) mean_variance: mean-variance tangent portfolio
  (4) mean_variance_short_constraint: mean-variance tangent portfolio with short-sale constraints
  (5) MinVar: minimum-variance portfolio
  (6) Robust: mean-variance portfolio with box uncertainty in mean

  Args:
    data: DataFrame : crsp-2.csv

  Attributes:
  data: Dataframe of initial data
  date_index : index of datetime
  ret: Dataframe of return panel data
  prc: Dataframe of price panel data
  shrout : Dataframe of shrout panel data
  mktcap : Datafroma of market capitalization data
  """
  def __init__(self, data):
    super().__init__(data)
    
  def value_weight(self, when :int):
    """value-weight portfolio
    Weight is proportional to the size (= prc x shrout)

    Args:
      when: the last timestamp of the window

    Return:
      dataframe of value-weight
    """
    return self.mktcap.loc[self.date_index[when]] / self.mktcap.loc[self.date_index[when]].sum()

  def equal_weight(self, when : int):
    """equal-weight portfolio

    Args:
      when: the last timestamp of the window

    Return:
      dataframe of equal-weight
    """
    return np.ones_like(self.ret.loc[self.date_index[when]]) / self.ret.shape[1]
  
  def mean_variance(self, when : int):
    """mean-variance (tangent) portfolio

    Args:
      when: the last timestamp of the window

    Return:
      dataframe of weight of mean-variance portfolio
    """
    ret_window = self.get_ret_til(when)
    returns = ret_window.mean(axis=0).values
    covariance = ret_window.cov().values

    e = np.ones_like(returns)

    # Objective function
    def fobj(w, mu, C): return -(w @ mu) /  np.sqrt(w.T @ C @ w)

    # Budget constraint (equality)
    def fcon_budget(w): return w.sum() - 1 # = 0

    cons = [dict(type='eq', fun=fcon_budget)]

    w0 = e / len(e) # Initial guess
    
    res = sp.optimize.minimize(fobj, w0, args=(returns, covariance),  constraints = cons, options={'maxiter':5000})

    weight = res.x

    return weight
  
  def mean_variance_short_constraint(self, when : int):
    """mean-variance (tangent) portfolio with short-sale constraint

    Args:
      when: the last timestamp of the window

    Return:
      dataframe of weight of mean-variance portfolio with short-sale constraint weight
    """
    ret_window = self.get_ret_til(when)
    returns = ret_window.mean(axis=0).values
    covariance = ret_window.cov().values

    e = np.ones_like(returns)

    # Objective function
    def fobj(w, mu, C): return -(w @ mu) /  np.sqrt(w.T @ C @ w)

    # Budget constraint (equality)
    def fcon_budget(w): return w.sum() - 1 # = 0

    cons = [dict(type='eq', fun=fcon_budget)]

    w0 = e / len(e) # Initial guess

    # Bounds
    bounds = sp.optimize.Bounds(0 * e)
    
    res = sp.optimize.minimize(fobj, w0, args=(returns, covariance),  constraints = cons, bounds=bounds, options={'maxiter':5000})

    weight = res.x

    return weight
  
  def min_var(self, when : int):
    """minimun-variance portfolio

    Args:
      when: the last timestamp of the window

    Return:
      dataframe of weight of minimum-variance portfolio
    """
    ret_window = self.get_ret_til(when)
    returns = ret_window.mean(axis=0).values
    covariance = ret_window.cov().values

    e = np.ones_like(returns)
    weight = np.linalg.solve(covariance, e)
    weight = weight / np.sum(weight)

    return weight
  
  def robust_optimization(self, when : int):
    """mean-variance (tangent) portfolio with box uncertainty in mean

    Args:
      when: the last timestamp of the window

    Return:
      dataframe of weight of minimum-variance portfolio
    """
    ret_window = self.get_ret_til(when)
    returns = ret_window.mean(axis=0).values
    covariance = ret_window.cov().values

    e = np.ones_like(returns)

    # Objective function - Sharpe Ratio Maximisation with box uncertainty
    def fobj(w, mu, C): return -(w @ mu - np.abs(w) @ np.std(ret_window, axis=0)) /  np.sqrt(w.T @ C @ w)

    # Budget constraint (equality)
    def fcon_budget(w): return w.sum() - 1 # = 0

    cons = [dict(type='eq', fun=fcon_budget)]
    
    w0 = e / len(e) # Initial guess
    
    res = sp.optimize.minimize(fobj, w0, args=(returns, covariance),  constraints = cons, options={'maxiter':5000})

    weight = res.x
    
    return weight