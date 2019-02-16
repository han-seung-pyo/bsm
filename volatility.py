# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 12:32:49 2018

@author: 한승표
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import root, fsolve, newton


s= 100;
k = 100;
r = 0.02;
q = 0.01;
sigma = 0.2;
t = 0.25;
option_type = 'call'
optionprice = 5.3
def d1(s,k,r,q,t,sigma):
    return (np.log(s/k) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))

def d2(s,k,r,q,t,sigma):
    return (np.log(s/k) + (r-q-0.5*sigma**2)*t)/(sigma*np.sqrt(t))

def bs_price(s,k,r,q,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    d_1 = d1(s,k,r,q,t,sigma)
    d_2 = d2(s,k,r,q,t,sigma)
    #d_1 = (np.log(s/k) + (r-q+0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    #d_2 = (np.log(s/k) + (r-q-0.5*sigma**2)*t)/(sigma*np.sqrt(t))
    option_price = x * s * np.exp(-q*t) * norm.cdf(x*d_1) -x*k*np.exp(-r*t) *norm.cdf(x*d_2);
    return option_price;

def bs_vega(s,k,r,q,t,sigma):
    vega = s * np.exp(-q*t) * norm.pdf(d1(s,k,r,t,q,sigma))*np.sqrt(t)
    return vega

def implied_vol2(s,k,r,q,t,optionprice,option_type,init=0.1,tol=1e-6):
    err = 1
    vol = init
    while abs(err) > tol:
        err = optionprice -bs_price(s,k,r,q,t,vol,option_type)
        vol = vol + err/bs_vega(s,k,r,q,t,vol)
    return vol.round(3)       


def implied_vol(s,k,r,q,t,optionprice,option_type):
    f = lambda x : bs_price(s,k,r,q,t,x,option_type) - optionprice
    return scipy.optimize.brentq(f,0,5)

def bsm_implied_vol(s0,k,T,t,r,c0,option_type,sigma_est,it=100):
    for i in range(it):
        sigma_est = sigma_est - (bs_price(s,k,r,q,t,sigma_est,option_type)-c0)/bsm_vega(s0,k,T,t,r,sigma_est)
    return sigma_est

#%%
print(implied_vol2(s,k,r,q,t,optionprice,option_type))
print( '%0.3f' %(implied_vol(s,k,r,q,t,optionprice,option_type)))
#%%
#bisection 너무 느리다
#코딩 다시하기
    
def bisection(s,k,r,q,t,optionprice,option_type,low_vol, high_vol,tol=1e-6):
    iv = 0.5 * (low_vol +high_vol)
    est_price = bs_price(s,k,r,q,t,iv,option_type)
    err = est_price - optionprice
    while abs(err)>tol:
        if bs_price(s,k,r,q,t,sigma,option_type) < optionprice:
            low_vol = iv
        else:
            high_vol = iv
        iv  = 0.5 * (low_vol+high_vol)
        est_price = bs_price(s,k,r,q,t,iv,option_type)
        err = est_price - optionprice
    return iv
