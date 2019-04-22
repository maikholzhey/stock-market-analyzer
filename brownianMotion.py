# simulate brownian motion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

n=252 # full year in trading days
dt=0.000001 # to be somewhat stable in time dt=0.000001

x=pd.DataFrame()
np.random.seed(1)

def makeFig():
	x.columns=['V','SAP','JPM','MSFT', 'AAPL','INTC','MITT']
	#stock=['V','SAP','JPM','MSFT','AAPL','INTC','MITT']
	#colstock=['blue','orange','green','red', 'pink','yellow', 'olive']
	columncols = ['blue','orange','green','red','pink','yellow', 'olive']
	x.plot(color = columncols)
	plt.legend(x.columns)
	plt.xlabel('t')
	plt.ylabel('X')
	plt.title('brownian motion finance model with averaged inputs')
	plt.show()

	
def GBM(x0, mu, sigma):
	step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
	temp=pd.DataFrame(x0*step.cumprod())
	global x
	x=pd.concat([x,temp],axis=1)

	
# task: GBM with stochastic volatility
def GMBstochVolal(x0,mu,sigma,sigmasigma):
	"""drive sigma with own GBM using sigmasigma as sigma"""
	return NotImplementedError

# VISA
x0 = 107.4# start value
mu = 0.15# estimate
sigma = 11.79# volatility

GBM(x0, mu, sigma)
	
# SAP
x0 = 113.21# start value
mu = 0.03# estimate
sigma = 5.16# volatility

GBM(x0, mu, sigma)

# JPM
x0 = 95.224# start value
mu = 0.083# estimate
sigma = 7.88# volatility

GBM(x0, mu, sigma)

# MSFT
x0 = 74.72# start value
mu = 0.14# estimate
sigma =10.469# volatility

GBM(x0, mu, sigma)

# AAPL
x0 = 173.332# start value
mu = 0.184# estimate
sigma = 15.6109# volatility

GBM(x0, mu, sigma)

# INTC
x0 = 32.4168# start value
mu = 0.0641# estimate
sigma = 5.6348# volatility

GBM(x0, mu, sigma)

# MITT
x0 = 17.23028# start value
mu = 0.006119# estimate
sigma = 0.9179# volatility

GBM(x0, mu, sigma)

# plot some shit
makeFig()
	
	


