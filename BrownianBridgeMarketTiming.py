# simulate brownian motion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Plot grayscale
plt.style.use('grayscale')

n=252  # full year in trading days
dt=0.00001 # to be somewhat stable in time

x=pd.DataFrame()

#np.random.seed(1)

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
	x=pd.DataFrame()
	def local(x0,mu,sigma):
		step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
		return [x0*step.cumprod()]
	# for s in range(0,3):
		# if s == 0:
			# tmp = np.concatenate((local(x0,mu,sigma),local(x0,mu,sigma)),axis = 0)
		# else:
			# tmp = np.concatenate((tmp,local(x0,mu,sigma)),axis = 0)
	# ttt = np.mean(tmp,0)
	return pd.DataFrame(local(x0,mu,sigma))
	
# task: GBM with stochastic volatility
def GMBstochVolal(x0,mu,sigma,sigmasigma):
	"""drive sigma with own GBM using sigmasigma as sigma"""
	return NotImplementedError

def sumup():
	x=pd.DataFrame()
	# VISA
	x0 = 107.4# start value
	mu = 1.15# estimate
	sigma = 11.79# volatility

	temp = GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)
		
	# SAP
	x0 = 113.21# start value
	mu = 1.03# estimate
	sigma = 5.16# volatility

	temp = GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)
	
	# JPM
	x0 = 95.224# start value
	mu = 1.083# estimate
	sigma = 7.88# volatility

	temp = GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)

	# MSFT
	x0 = 74.72# start value
	mu = 1.14# estimate
	sigma =10.469# volatility

	temp = GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)

	# # AAPL
	# x0 = 173.332# start value
	# mu = 1.184# estimate
	# sigma = 15.6109# volatility

	temp = GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)

	# INTC
	x0 = 32.4168# start value
	mu = 1.0641# estimate
	sigma = 5.6348# volatility

	temp = GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)

	# MITT
	x0 = 17.23028# start value
	mu = 1.006119# estimate
	sigma = 0.9179# volatility

	GBM(x0, mu, sigma)
	x=pd.concat([x,temp],axis=0)
	
	return x

for s in range(1000):	
	if np.mod(s,50) == 0:
		print('called')
	x = pd.concat([x,sumup().sum(level=0)],axis=0)

	
# # plot some shit
# makeFig()
	
plt.plot(np.mean(x,0))
plt.plot(np.mean(x,0)+np.std(x,0))
plt.plot(np.mean(x,0)-np.std(x,0))
plt.show()
	


