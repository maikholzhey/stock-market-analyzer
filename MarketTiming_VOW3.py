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

# general purpose
import sys, os
import time

# clock the execution
startTime = time.time()

# Plot grayscale
plt.style.use('grayscale')

# api instructions
quandl.ApiConfig.api_key = "YOUR_API_KEY"
end = datetime.now()
start = end - timedelta(days=365)

wkn = 'FSE/EON_X'
# frankfurt stock exchange
mydata = quandl.get(wkn, start_date = start, end_date = end)
f = mydata.reset_index()

# timeseries
plt.figure(1)
plt.title("FSE/WAC_X")
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
plt.plot(days, y,label=wkn)
plt.plot(days, yy,label='linear reg')
plt.plot(days, yy + sigma,label='mean + std',linestyle='dashed')
plt.plot(days, yy - sigma,label='mean - std',linestyle='dashed')
plt.title('Linear Fit')
plt.ylabel('price in [EUR]')

##################################
### Brownian Bridge
##################################

n=len(x)  # full year in trading days
dt=0.00001 # to be somewhat stable in time [Euler Discretization :: Variation of the Wiener]

CC=pd.DataFrame()

def GBM(x0, mue, sigma, n, dt):
	x=pd.DataFrame()
	def local(x0,mue,sigma):
		step=np.exp((mue-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
		return [x0*step.cumprod()]
	return pd.DataFrame(local(x0,mue,sigma))

for s in range(1000):	# candidate for GPU massive parallelization!!! (numba.cuda.random.)
	if np.mod(s,200) == 0:
		print('called')
	CC = pd.concat([CC,GBM(x0,mue, sigma, n, dt)],axis=0)

# drift correction
CC = CC + linear_func(xx, popt[0],0)
q = [10,90]
CoInt = np.percentile(CC,q,axis=0)

plt.figure(3)
plt.plot(days, y,label=wkn)
plt.plot(days,np.mean(CC,0) ,label='mean')
plt.plot(days,CoInt[0].flatten(),color='grey',label='perc10',linestyle='dashed')
plt.plot(days,CoInt[1].flatten(),color='grey',label='perc90',linestyle='dashed')
plt.legend()
plt.title('Brownian Bridge')
plt.ylabel('price in [EUR]')

print("GBM worst")
GBMworst = CoInt[0].flatten()[-1]
print(GBMworst)
print("GBM best")
GBMbest = CoInt[1].flatten()[-1]
print(GBMbest)
print("end price yesterday")
endPrice = y[-1]
print(endPrice)

##########################
### Buy-Keep-Drop ########
##########################

def DK(worst,best,endPrice):
	if endPrice > np.multiply(best,1):
		return u"<font color="+u"red"+"><b>"+u"SELL"+u"</b></font>"
	if endPrice < np.multiply(worst,1):
		return u"<font color="+u"green"+"><b>"+u"BUY"+u"</b></font>"
	else:
		return u"<font color="+u"yellow"+"><b>"+u"KEEP"+u"</b></font>"
		
print(DK(GBMworst,GBMbest,endPrice))

print("overall time")
print(str(time.time() - startTime) + u"sec")

# popt[0] in relation to total stock price to have a relative profitability

# run GBMbest and GBMworst case analysis for 1y prediction of entire portfolio

### MPI setting for each stock
### ==========================
# from multiprocessing import Pool
# MPIbase = list()

# for i in range(len(samples.T)):
	# if np.mod(i,NoProcess) == 0:
		# if i+NoProcess<len(samples.T):
			# MPIbase.append(samples.T[i:i+NoProcess])
		# else:
			# MPIbase.append(samples.T[i:len(samples.T)])

# u_mc = list()

# for i in range(len(MPIbase)):
	# # start asynchronous child processes
	# result = dict()
	# resKey = 0
	# for s in MPIbase[i]:
		# result[str(resKey)] = pool.apply_async(u, (s[0], LyMax, sig, tdt, fsig, dftM, window, dtmax, nx , ny , nz, s[1] , boundaryConditions,))
		# resKey += 1
	# # collect results
	# resKey = 0
	# for s in MPIbase[i]:			
		# u_mc.append(result[str(resKey)].get())
		# resKey += 1

plt.show()