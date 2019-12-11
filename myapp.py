### IMPORT :

# python packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

# mypackages
import myfinutils as fin
import myblackscholes as bs
import myautocallable as acl
import mygreeks as grk


### INPUTS :

print('\nINPUT AND PARAMETERS ESTIMATION...\n')

# parameters
n_simu = int(1e6) # 10^6 simu

# data
N = 1 # NOMINAL
S_0 = 3042 # SPOT PRICE
kickout = 1.10 # KICKOUT BARRIER PERCENTAGE
protection = 0.95 # PROTECTION_BARRIER
I = 0.04 # INTEREST
T = 2020 # MATURITY
t_0 = 2015

# other variables
S_p = S_0 * protection # PROTECTION BARRIER
S_k = S_0 * kickout # KICKOUT BARRIER
t_steps = range(t_0+1, T+1)

# input from excel file
xls_data = pd.ExcelFile('res\\Data_Pricing.xlsx') # read pathfile
maturity_yrs = np.arange(2016, 2025+1, 1)

# strike prices
Strk = pd.read_excel(xls_data, sheet_name='Strike_Price', header=None, names=maturity_yrs)
# time to maturities
TtM = pd.read_excel(xls_data, sheet_name='Maturity', header=None, names=maturity_yrs)
# call options
Call = pd.read_excel(xls_data, sheet_name='Call_Price', header=None, names=maturity_yrs)
# put options
Put = pd.read_excel(xls_data, sheet_name='Put_Price', header=None, names=maturity_yrs)
# spot prices
Spot = pd.read_excel(xls_data, sheet_name='Underlying_Price', header=None, names=maturity_yrs)
# (daily) risk-free rate
d_RfR = pd.read_excel(xls_data, sheet_name='Risk_Free_Rate_EONIA', header=None, names=maturity_yrs)


### VARS ESTIMATION :

starting_time = time.time()

# interest rate
InR = fin.interestRate(Call, Put, Spot, Strk, TtM)
# implied volatility
flag = 'C' # call option is used for implied vol
ImV = fin.impliedVolatilitySurface(Call, flag, Spot, Strk, TtM, InR)
# risk-free rate -> discount rate
DsR = d_RfR.iloc[0] * np.sqrt(365)
# debug
elapsed_time = time.time() - starting_time
print('Interest rates, implied volatility, discount rates estimated in', elapsed_time, 's\n')

# find key-values to interpolate from dataframes
S = Strk[t_steps[0]]
i = 0
while (S_p > S[i]):  
    i += 1
# interpolation: dataframes -> series
k_Drift = (InR.iloc[i] * (S_p - S[i-1]) + InR.iloc[i-1] * (S[i] - S_p)) / (S[i] - S[i-1])
k_Vol = (ImV.iloc[i] * (S_p - S[i-1]) + ImV.iloc[i-1] * (S[i] - S_p)) / (S[i] - S[i-1])
k_TtM = TtM.iloc[0] # constant over strike value


### PLOT GREEKS BY MONTECARLO SIMULATION :

# initialize spark
sc = acl.startDistribuitedEnvironment()

## VEGA :
print('\n\nSIMULATING VEGA...\n')
starting_time = time.time()
vol_param = np.arange(0, 5, 0.5)

# simulate range of prices
priceForVega, vol = grk.computePricesForGreek('V', t_steps, k_TtM, k_Drift, k_Vol, DsR, S_0, S_k, S_p, N, I, vol_param, n_simu, sc)
# compute discrete derivatives
vega, vol_mid = grk.derivateGreek(priceForVega, vol)
# debug
print('\nElapsed time =', time.time() - starting_time, 's')
# plot vega
fig1, ax11, ax12 = grk.plot(vega, vol_mid, priceForVega, vol, 'VEGA', 'PRICE', 'VOLATILITY')


## DELTA :
print('\n\nSIMULATING DELTA...\n')
starting_time = time.time()
vol_param = np.arange(0.5, 1.5, 0.1)

# simulate range of prices
priceForDelta, spot = grk.computePricesForGreek('D', t_steps, k_TtM, k_Drift, k_Vol, DsR, S_0, S_k, S_p, N, I, vol_param, n_simu, sc)
# compute discrete derivatives
delta, spot_mid = grk.derivateGreek(priceForDelta, spot)
# debug
print('\nElapsed time =', time.time() - starting_time, 's')
# plot delta
fig2, ax21, ax22 = grk.plot(delta, spot_mid, priceForDelta, spot, 'DELTA', 'PRICE', 'SPOT PRICE') 


## RHO :
print('\n\nSIMULATING RHO...\n')
starting_time = time.time()
vol_param = np.arange(-5, 5, 1)

# simulate range of prices
priceForRho, inR = grk.computePricesForGreek('R', t_steps, k_TtM, k_Drift, k_Vol, DsR, S_0, S_k, S_p, N, I, vol_param, n_simu, sc)
# compute discrete derivatives
rho, inR_mid = grk.derivateGreek(priceForRho, inR)
# debug
print('\nElapsed time =', time.time() - starting_time, 's')
# plot delta
fig3, ax31, ax32 = grk.plot(rho, inR_mid, priceForRho, inR, 'RHO', 'PRICE', 'INTEREST RATE')
# ax32.set_ylim([0.2, 0.8]) # correction for the 'rho' ax


## GAMMA :
print('\n\nSIMULATING GAMMA...\n')
starting_time = time.time()
# compute discrete derivatives
# (gamma is the second derivative of price over volatility, i.e. the first derivative of delta)
gamma, vol_midmid = grk.derivateGreek(delta, vol_mid)
# debug
print('Elapsed time =', time.time() - starting_time, 's')
# plot gamma
fig4, ax41, ax42 = grk.plot(gamma, vol_midmid, priceForDelta, vol, 'GAMMA', 'PRICE', 'VOLATILITY')

# show plots
plt.show() 