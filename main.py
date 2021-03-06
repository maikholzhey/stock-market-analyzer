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
from multiprocessing import Pool

# write to historic log file
doc = True

def linear_func(x, a, b):
	return a*x +b
	
def GBM(x0, mue, sigma, n, dt):
	x=pd.DataFrame()
	def local(x0,mue,sigma):
		step=np.exp((mue-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
		return [x0*step.cumprod()]
	return pd.DataFrame(local(x0,mue,sigma))
	
def DK(worst,best,endPrice):
	if endPrice > best:
		return u"<font color="+u"red"+"><b>"+u"SELL"+u"</b></font>"
	if endPrice < worst:
		return u"<font color="+u"green"+"><b>"+u"BUY"+u"</b></font>"
	else:
		return u"<font color="+u"gold"+"><b>"+u"KEEP"+u"</b></font>"
		
def red(a):
	if a < 0.:
		return u"<font color="+u"red"+"><b>"+str(a)+u"</b></font>"
	else:
		return str(a)
		
def PortfolioAnalysis(x0,mue, sigma, n, dt):
	p = pd.DataFrame()
	# practical Monte Carlo with >10k = 10 * 1k samples
	for s in range(1000):	# candidate for GPU massive parallelization!!! (numba.cuda.random.)
		p = pd.concat([p,GBM(x0,mue, sigma, n, dt)],axis=0)
	return p
		
def StockAnalysis(StockOfInterest,stock, prep, TimeFrame, iter):
	y = StockOfInterest.values[len(StockOfInterest.values)-TimeFrame:len(StockOfInterest.values)]
	x = StockOfInterest.index.date[len(StockOfInterest.values)-TimeFrame:len(StockOfInterest.values)] #datetime index
	
	data = list()
	
	#calc std
	sigma = np.std(y)
	d = np.linspace(1,len(x),len(x))
	# lin reg
	popt, pcov = curve_fit(linear_func, d, y)

	yy= linear_func(d, *popt)
	
	days= d-np.max(d)
	
	# GBM
	mue = 1 + popt[0]
	x0 = yy[0]
	
	# dump analysis results
	data.append(stock[iter]) # label
	data.append(red(popt[0])) # av daily revenue
		
	data.append(red(sigma)) # volatility
	
	data.append(red(np.round_((y[-1]*0.025)/popt[0],0)))	#trading penalty
	
	##################################
	### Brownian Bridge
	##################################

	n=len(x)  # full year in trading days
	dt=0.00001 # to be somewhat stable in time [Euler Discretization :: Variation of the Wiener]

	CC=pd.DataFrame()
	
	# practical Monte Carlo with >10k samples
	for s in range(1000):	# candidate for GPU massive parallelization!!! (numba.cuda.random.)
		CC = pd.concat([CC,GBM(x0,mue, sigma, n, dt)],axis=0)

	# drift correction
	CC = CC + linear_func(d, popt[0],0)
	q = [10,90]
	CoInt = np.percentile(CC,q,axis=0)

	
	GBMworst = CoInt[0].flatten()[-1]
	GBMbest = CoInt[1].flatten()[-1]
	endPrice = y[-1]
	
	data.append(red(GBMworst))
	data.append(red(GBMbest))
	data.append(red(endPrice))
	
	# drop or keep analysis
	data.append(DK(GBMworst,GBMbest,endPrice))
	# portfolio rep
	data.append(str(prep[iter]))
	
	return data
		
def generate_html_with_table(data, columns_or_rows = 1, \
                             column_name_prefix = 'Column', \
                             span_axis = 1):
    # Calculate number of elements in the list: data
    elements = len(data)
    # Calculate the number of rows/columns needed
    if (span_axis == 0): # if spanning along a column
      rows = columns_or_rows
      columns = int(np.ceil(elements/rows))    
    else: #(span_axis = 1)
      columns = columns_or_rows
      rows = int(np.ceil(elements/columns))
    # Generate Column Names
    column_names = [u'Stock of Interest',u'Profit/Day [€]',u'Volatility [€]', \
					u'TradingPenalty [days] (2.5[%])', u'GBM worst [€]', u'GBM best [€]', \
					u'End of Day Price [€]', 'Sell-Keep-Buy', 'QTY']
    # Convert the data into a numpy array    
    data_array = np.array(data + ['']*(columns*rows - elements))		
    if (span_axis == 0):
      data_array = data_array.reshape(columns,rows).T  
    else: #(span_axis == 0)
      data_array = data_array.reshape(rows,columns)  
    
    # Convert the numpy array into a pandas DataFrame
    data_df = pd.DataFrame(data_array, columns = column_names) 
    # Create HTML from the DataFrame
    data_html = data_df.to_html()
    return (data_html, data_df)
	
	
###########################################
######## Analysis #########################
###########################################

# main frame protected execution
if __name__ == '__main__':
	# ==========================
	# -- MAIN ------------------
	# ==========================
		
	#stock of interest
	stock=['FSE/ADS_X','FSE/ALV_X','FSE/BAS_X','FSE/BAYN_X','FSE/BEI_X',\
			'FSE/BMW_X','FSE/CBK_X','FSE/CON_X','FSE/DAI_X','FSE/DBK_X',\
			'FSE/DPW_X','FSE/EON_X','FSE/FRE_X','FSE/HEI_X','FSE/HEN3_X',\
			'FSE/LIN_X','FSE/ALV_X','FSE/BAS_X','FSE/LHA_X','FSE/MUV2_X',\
			'FSE/SIE_X','FSE/TKA_X','FSE/VOW3_X','FSE/VOS_X','FSE/VIB3_X',\
			'FSE/SAP_X','FSE/MAN_X','FSE/MEO_X','FSE/LEO_X','FSE/WAC_X',\
			]#DAX
	#colstock=['blue','orange','red', 'pink','yellow', 'olive']
	stockquantity = [3,6,6,6,2,\
					2,4,2,1,3,\
					1,5,1,2,3,\
					5,2,3,2,2,\
					2,1,5,1,1,\
					9,1,1,1,1,\
					]

	# some containers
	flist = [] # stock data series
	data1 = [] # html dump
	data2 = [] # html dump medium long
	data3 = [] # html dump short
	data4 = [] # html dump very short

	prep = stockquantity	# portfolio representaion

	# clock the execution
	startTime = time.time()

	# Plot grayscale
	plt.style.use('grayscale')

	# api instructions
	quandl.ApiConfig.api_key = "YOUR_API_KEY"
	end = datetime.now()
	start = end - timedelta(days=385)

	# Download
	plt.figure(1)
	print("Downloading data...")
	for i in range(len(stock)):
		# frankfurt stock exchange
		mydata = quandl.get(stock[i], start_date = start, end_date = end)
		f = mydata.reset_index()
		# timeseries
		f = pd.Series(f.Close.values,f.Date)
		f.plot(label=stock[i])
		plt.legend()
		plt.ylabel('price in [USD]')
		# dump to container
		flist.append(f)

	# Data FSE is someteimes unreliable... 
	# so better check in detail, or pay for a reliable data source
	#plt.show()
		
	## MPI setting for each stock
	## ==========================
	NoProcess = 10 # adjust for your machine!
	# --------------------------
	# MPI split
	pool = Pool(processes=NoProcess)         # start worker processes
	
	MPIbase = list()
	
	iterate = [i for i in range(len(stock))]
	TimeFrame = [252,150,50,20]
	DataDump = [data1, data2, data3, data4]
	
	for i in range(len(stock)):
		if np.mod(i,NoProcess) == 0:
			if i+NoProcess<len(stock):
				MPIbase.append(iterate[i:i+NoProcess])
			else:
				MPIbase.append(iterate[i:len(stock)])
	
	for k in range(len(TimeFrame)):
		print(u"Analysis on " + str(TimeFrame[k]) + u" trading days...")
		for i in range(len(MPIbase)):
			# start asynchronous child processes
			result = dict()
			resKey = 0
			for s in MPIbase[i]:
				result[str(resKey)] = pool.apply_async(StockAnalysis, (flist[s],stock, prep, TimeFrame[k], s))
				resKey += 1
			# collect results
			resKey = 0
			for s in MPIbase[i]:			
				DataDump[k] += result[str(resKey)].get()
				resKey += 1
	
	######################################
	### Portfolio Prediction #############
	######################################
	
	print("Portfolio Prediction...")
	
	dataX0 = list()
	dataMue = list()
	dataSigma = list()
	
	for i in range(len(stock)):
		y = flist[i].values[len(flist[i].values)-252:len(flist[i].values)]
		x = flist[i].index.date[len(flist[i].values)-252:len(flist[i].values)] #datetime index
		
		d = np.linspace(1,len(x),len(x))
		# lin reg
		popt, pcov = curve_fit(linear_func, d, y)
		
		#  - weighted by prep
		for j in range(prep[i]):
			dataSigma.append(np.std(y))
			dataMue.append(1 + popt[0])
			dataX0.append(y[-1])
		
	##################################
	### Brownian Bridge
	##################################
	
	n  = 252  # full year in trading days
	dt = 0.00001 # to be somewhat stable in time [Euler Discretization :: Variation of the Wiener]
	
	mue = np.mean(dataMue)
	sigma = np.mean(dataSigma)
	x0 = np.sum(dataX0)
	
	# maximum threads on the machine: i7-8700
	NoProcess = 12
	pool = Pool(processes=NoProcess)         # start worker processes
		
	# start asynchronous child processes
	result = dict()
	resKey = 0
	for i in range(NoProcess):
		result[str(resKey)] = pool.apply_async(PortfolioAnalysis, (x0,mue, sigma, n, dt))
		resKey += 1
		
	p = pd.DataFrame()
	# collect results
	resKey = 0
	for i in range(NoProcess):		
		p = pd.concat([p,result[str(resKey)].get()],axis=0)
		resKey += 1

	# drift correction
	p = p + linear_func(d, mue-1 ,0)
	q = [10,90]
	CoInt = np.percentile(p,q,axis=0)

	
	GBMworst = CoInt[0].flatten()[-1]
	GBMbest = CoInt[1].flatten()[-1]
	endPrice = y[-1]
	
	plt.figure(2)
	plt.plot(np.mean(p,0))
	plt.plot(np.mean(p,0)+np.std(p,0))
	plt.plot(np.mean(p,0)-np.std(p,0))
	
	iterate = [i for i in range(len(stock))]
	# todays networth
	networth = 0
	for i in iterate:
		networth += np.round_(np.multiply(flist[i].values[-1],prep[i]),2)
	
	# LRM estimate
	LRM = np.round_(np.mean(p,0).values[-1],2)
		
	# GBM best
	BMbest = np.round_(LRM+np.std(p,0).values[-1],2)
	
	# GBM worst
	BMworst = np.round_(LRM-np.std(p,0).values[-1],2)
	
	# interest rates
	iLRM = np.round_(np.multiply(np.divide(LRM,networth)-1,100),2) 
	iBMbest = np.round_(np.multiply(np.divide(BMbest,networth)-1,100),2)
	iBMworst = np.round_(np.multiply(np.divide(BMworst,networth)-1,100),2)
	
	print("Writing results to file...")
	
	#data = ['one','two','three','four','five','six','seven','eight','nine']
	columns = 9                   # Number of Columns
	columns_or_rows = columns
	column_name_prefix = '' # Prefix for Column headers
	span_axis = 1                 # Span along a row (1) or a column (0) first
	showOutput = False            # Use False to suppress printing output

	# Generate HTML
	data_html1, data_df1 = generate_html_with_table(DataDump[0], columns_or_rows, column_name_prefix, span_axis)
	data_html2, data_df1 = generate_html_with_table(DataDump[1], columns_or_rows, column_name_prefix, span_axis)
	data_html3, data_df1 = generate_html_with_table(DataDump[2], columns_or_rows, column_name_prefix, span_axis)
	data_html4, data_df1 = generate_html_with_table(DataDump[3], columns_or_rows, column_name_prefix, span_axis)

	html_header = u"<h1>Stock market analysis</h1> <p><i> Frankfurt Stock Exchange</p></i>"
	
	html_trailer = u"<h2>Networth</h2> <p> Todays networth (in EUR): <strong>"+red(networth)+ u"</strong> </p>"+"<p> Linear Regression Model estimate in 252 trading days (in EUR): <strong>"+red(LRM)+ u"</strong> ("+red(iLRM)+ u"%) </p>"+"<p> Geometric Brownian Motion estimate in 252 trading days (in EUR): <strong>"+red(BMbest)+ u"</strong> ("+red(iBMbest)+ u"%) --- best case</p>"+"<p> Geometric Brownian Motion estimate in 252 trading days (in EUR): <strong>"+red(BMworst)+ u"</strong> ("+red(iBMworst)+ u"%) --- worst case</p>"
	
	content = html_header +u"<p> Analysis performed on 252 trading days period </p>" +data_html1 + u"<p> Analysis performed on 150 trading days period </p>" +data_html2 +u"<p> Analysis performed on 50 trading days period </p>" +data_html3 + u"<p> Analysis performed on 10 trading days period </p>" +data_html4 + html_trailer

	#redmarker = u"<font color="+u"red"+"><b>!-</b></font>"
	contentr = content.replace(u"&lt;",u"<")
	contentr = contentr.replace(u"&gt;",u">")
	contentr = contentr.replace(u"<...",u"")

	# save data html to file
	patht = os.path.abspath("table.html")	
	Html_file= open(patht,"w")
	Html_file.write(contentr)
	
	# add date
	dateInfo = u"<p><i>" + "{:%B %d, %Y}".format(datetime.now()) + u"</i></p><p> &copy mh </p>"
	Html_file.write(dateInfo)
	Html_file.close()

	######################################
	### Networth Logfile #################
	######################################
	
	historicdata = str(networth) + u"\t\t" + str(LRM) + u"\t\t" +str(BMbest) + u"\t" +str(BMworst) + u"\t\t" + "{:%B %d, %Y}".format(datetime.now()) + u"\n"

	if doc:
		# save data to historic analysis
		pathh = os.path.abspath("historicNetworth.txt")
			
		txt_file = open(pathh,"a")
		txt_file.write(historicdata)
		txt_file.close()
		print(historicdata)

	print("finished in overall time")
	print(str(time.time() - startTime) + u"sec")
	
	#plt.show()
	sys.exit(0)