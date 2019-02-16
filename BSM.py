# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 10:29:30 2018

@author: 한승표
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import matplotlib.pyplot as plt
from scipy.optimize import root, fsolve, newton
import time ## 시간 측정하는 library. time.time(측정할 것)

s= 100;
k = 100;
r = 0.02;
q = 0.01;
sigma = 0.2;
t = 0.25;
option_type = 'call'
#%%
#BLACK SHOLES PRICE
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
    return option_price.round(3);
#%%
#greek 계산
def bs_delta(s,k,r,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    delta = x*np.exp(-q*t) *norm.cdf(x*d1(s,k,r,t,q,sigma))
    return delta

def bs_vega(s,k,r,q,t,sigma):
    vega = s * np.exp(-q*t) * norm.pdf(d1(s,k,r,t,q,sigma))*np.sqrt(t)
    return vega

def bs_theta(s,k,r,q,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    theta = (-np.exp(-q*t)*s*norm.pdf(d1(s,k,r,t,q,sigma))*sigma)/(2*np.sqrt(t))-x*k*np.exp(-r*t)*norm.cdf(x*d2(s,k,r,t,q,sigma))+x*q*s*np.exp(-q*t)*norm.cdf(x*d1(s,k,r,t,q,sigma))
    return theta

def bs_rho(s,k,r,q,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1; 
    rho = x * k * t * np.exp(-r*t)* norm.cdf(x*d2(s,k,r,t,q,sigma))
    return rho

def bs_psi(s,k,r,q,t,sigma,option_type):
    if option_type == 'call':
        x = 1;
    if option_type == 'put':
        x = -1;
    psi = x * s * t * np.exp(-q*t)* norm.cdf(x*d1(s,k,r,t,q,sigma))
    return psi

def bs_gamma(s,k,r,q,t,sigma):
    gamma = (np.exp(-q*t) * norm.pdf(d1(s,k,r,t,q,sigma)))/(s*sigma*np.sqrt(t))
    return gamma
#%%
#뉴턴랩슨 방식으로 implied vol 구하기
def bsm_implied_vol(s0,k,T,t,r,c0,option_type,sigma_est,it=100):
    for i in range(it):
        sigma_est = sigma_est - (bs_price(s,k,r,q,T,sigma_est,option_type)-c0)/bsm_vega(s0,k,T,t,r,sigma_est)
    
    return sigma_est
       

#%%    
#bs_prise and delta
x= []
for i in range(20):
    s = 90 + i
    x.append([bs_price(s,k,r,t,q,sigma,'call'),bs_delta(s,k,r,t,sigma,option_type)])
xx = pd.DataFrame(x,columns = ['bs_price','bs_delta'], index = np.arange(90,90+20)).round(3)
plt.plot(xx['bs_delta'])
plt.show()

#vega
y = []
for i in range(30):
    sigma = sigma + (i/100);
    y.append(bs_vega(s,k,r,t,q,sigma))
vega = pd.DataFrame(y, columns = ['bs_vega'], index = ((np.arange(0,30)/100)+0.2).round(2)).round(3)
plt.plot(vega)
plt.show()
#%%
#몬테카를로 시뮬레이
def montecarlo(s,k,r,q,t,sigma,option_type,M):    
    data = pd.DataFrame(norm.ppf(np.random.uniform(size = M)), columns = ['epsilon'])
    data['sT'] = data.apply(lambda x : s * np.exp((r-q-0.5*sigma**2)*t + sigma * np.sqrt(t) * x['epsilon']),axis =1)
    data['option_price'] = data.apply(lambda x: max(x['sT']-k,0) if option_type =='call' else max(k-x['sT'],0), axis = 1)
    return data['option_price'].sum()/M * np.exp(-r*t)
    
#Montecarlo simulation_ more efficient
def Montecarlo_sim(s,T,r,sigma,sim_num,n_steps):
    delta_t = T/n_steps
    z_matrix = np.random.standard_normal(size =(sim_num,n_steps))
    st_matrix = np.zeros((sim_num,n_steps))
    st_matrix[:,0] = s0
    for i in range(n_steps-1):
        st_matrix[:,i+1] = st_matrix[:,i]*np.exp((r-0.5*sigma**2)*delta_t + sigma*np.sqrt(delta_t)*z_matrix[:,i])
    
    return st_matrix

def stratifed(s,k,r,q,t,sigma,option_tpye,M):
    x = 0
    for i in range(M):
        epsilon = np.nan_to_num(norm.ppf((i-0.5)/M))
        s_T = s * np.exp((r-q-0.5*sigma**2)*t + sigma * np.sqrt(t) * epsilon)
        if option_tpye == 'call':
            xx = max(s_T-k,0)
        else:
            xx = max(k-s_T,0)
        x = x+ xx
        stratifed = (x / M) *np.exp(-r*t)
    return stratifed


s = 100
k =100
r = 0.02
q = 0.01
t = 0.25
vol = 0.2
option_type = 'call'

start = time.time()
option_value = montecarlo(s,k,r,q,t,vol,option_type,50000).round(4)
print((time.time() - start))


















