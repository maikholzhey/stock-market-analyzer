# pip install datareader
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
# quandl api explore
import quandl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# do fitting
import numpy as np
from scipy.optimize import curve_fit

# paths
import os

# Plot grayscale
plt.style.use('grayscale')

# api instructions
quandl.ApiConfig.api_key = "YOUR_API_KEY"
end = datetime.now()
start = end - timedelta(days=245)

# frankfurt stock exchange
mydata = quandl.get('FSE/SIX2_X', start_date = start, end_date = end)
f = mydata.reset_index()

# timeseries
plt.figure(1)
plt.title("FSE/SIX2_X")
f = pd.Series(f.Close.values,f.Date)
f.plot()


# extract last data and do linear regression
y = f.values
x = f.index.date #datetime index
	
#calc std
sigma = np.std(y)
d = np.linspace(1,len(x),len(x))

#lin reg
def linear_func(x, a, b):
	return a*x +b

popt, pcov = curve_fit(linear_func, d, y)
print(str(popt))

xx = d#-np.max(d)
yy= linear_func(xx, *popt)

mue = 1 + popt[0]
x0 = yy[0]

	
days= xx-np.max(d)
	
plt.figure(2)
plt.plot(days, y,label='FSE/SIX2_X')
plt.plot(days, yy,label='linear reg')
plt.plot(days, yy + sigma,label='mean + std',linestyle='dashed')
plt.plot(days, yy - sigma,label='mean - std',linestyle='dashed')
plt.title('Linear Fit')
plt.ylabel('price in [EUR]')

##################################
### Brownian Bridge
##################################

n=len(x)  # full year in trading days
dt=0.00001 # to be somewhat stable in time

CC=pd.DataFrame()

def GBM(x0, mue, sigma, n, dt):
	x=pd.DataFrame()
	def local(x0,mue,sigma):
		step=np.exp((mue-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
		return [x0*step.cumprod()]
	return pd.DataFrame(local(x0,mue,sigma))

def sumup(x0,mue, sigma, n, dt):
	x=pd.DataFrame()
	temp = GBM(x0, mue, sigma, n, dt)
	x=pd.concat([x,temp],axis=0)	
	return x

for s in range(1000):	
	if np.mod(s,200) == 0:
		print('called')
	CC = pd.concat([CC,sumup(x0,mue, sigma, n, dt).sum(level=0)],axis=0)

# drift correction
CC = CC + linear_func(xx, popt[0],0)
q = [5,95]
CoInt = np.percentile(CC,q,axis=0)

plt.figure(3)
plt.plot(days, y,label='FSE/SIX2_X')
plt.plot(days,np.mean(CC,0) ,label='mean')
plt.plot(days,CoInt[0].flatten(),color='grey',label='perc5',linestyle='dashed')
plt.plot(days,CoInt[1].flatten(),color='grey',label='perc95',linestyle='dashed')
plt.legend()
plt.title('Browian Bridge')
plt.ylabel('price in [EUR]')

plt.show()