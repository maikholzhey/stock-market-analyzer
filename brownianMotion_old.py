# simulate brownian motion

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from drawnow import drawnow

mu=1
n=300
dt=0.0001
x0=100
x=pd.DataFrame()
np.random.seed(1)

def makeFig():
	x.columns=np.arange(0.92,1,0.02)
	plt.plot(x)
	plt.legend(x.columns)
	plt.xlabel('t')
	plt.ylabel('X')
	plt.title('Realizations of Geometric Brownian Motion with different variances\n mu=1')
	plt.show()

	
for sigma in np.arange(0.92,1,0.02):
	step=np.exp((mu-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
	temp=pd.DataFrame(x0*step.cumprod())
	x=pd.concat([x,temp],axis=1)

makeFig()
	
	


