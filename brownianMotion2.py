# simulate brownian motion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drawnow import drawnow


n=252 # full year in trading days
dt=0.000001 # to be somewhat stable in time

x=pd.DataFrame()
np.random.seed(1)

def makeFig():
	x.columns=['SAP','JPM','MSFT', 'AAPL']
	plt.plot(x)
	plt.legend(x.columns)
	plt.xlabel('t')
	plt.ylabel('X')
	plt.title('brownian motion finance model with average inputs')
	plt.show()

	
def GBM(x0, mu, sigma):
	step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
	temp=pd.DataFrame(x0*step.cumprod())
	global x
	x=pd.concat([x,temp],axis=1)
	
# SAP
x0 = 113.21# start value
mu = 1.03# estimate
sigma = 5.16# volatility

GBM(x0, mu, sigma)

# JPM
x0 = 95.224# start value
mu = 1.083# estimate
sigma = 7.88# volatility

GBM(x0, mu, sigma)

# MSFT
x0 = 74.72# start value
mu = 1.14# estimate
sigma =10.469# volatility

GBM(x0, mu, sigma)

# AAPL
x0 = 173.332# start value
mu = 1.184# estimate
sigma = 15.6109# volatility

GBM(x0, mu, sigma)

# plot some shit
makeFig()
	
	


