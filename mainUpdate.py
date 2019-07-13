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

def linear_func(x, a, b):
	return a*x +b
	
def GBM(x0, mue, sigma, n, dt):
	x=pd.DataFrame()
	def local(x0,mue,sigma):
		step=np.exp((mue-sigma**2/2)*dt)*np.exp(sigma*np.random.normal(0,np.sqrt(dt),(1,n)))
		return [x0*step.cumprod()]
	return pd.DataFrame(local(x0,mue,sigma))
	
def DK(worst,best,endPrice):
	if endPrice > np.multiply(best,1):
		return u"<font color="+u"red"+"><b>"+u"SELL"+u"</b></font>"
	if endPrice < np.multiply(worst,1):
		return u"<font color="+u"green"+"><b>"+u"BUY"+u"</b></font>"
	else:
		return u"<font color="+u"gold"+"><b>"+u"KEEP"+u"</b></font>"
		
def red(a):
	if a < 0.:
		return u"<font color="+u"red"+"><b>"+str(a)+u"</b></font>"
	else:
		return str(a)
		
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
	data.append(red(np.multiply(np.divide(popt[0],y[-1]),100))) # av daily revenue rel to end price
	
	#anrev.append(popt[0]) # anual estimate revenue
	
	data.append(red(np.multiply(np.divide(sigma,y[-1]),100))) # volatility
	
	#netsigma.append(sigma)# GBM save
	
	#networth.append(y[-1])
	
	data.append(red(np.round_((y[-1]*0.025)/popt[0],0)))	#trading penalty
	
	##################################
	### Brownian Bridge
	##################################

	n=len(x)  # full year in trading days
	dt=0.00001 # to be somewhat stable in time [Euler Discretization :: Variation of the Wiener]

	CC=pd.DataFrame()

	for s in range(10000):	# candidate for GPU massive parallelization!!! (numba.cuda.random.)
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
    column_names = [u'Stock of Interest',u'Profit/Day/EndPrice [%]',u'Volatility/EndPrice [%]', \
					u'TradingPenalty [days] (2.5[%])', u'GBM worst [€]', u'GBM best [€]', \
					u'End of Day Price [€]', 'Drop-Keep-Buy', 'QTY']
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
	stock=['FSE/VOW3_X','FSE/WAC_X','FSE/SIX2_X','FSE/ZO1_X','FSE/EON_X','FSE/SKB_X']#,'SAP','JPM','MSFT','AAPL','INTC','MITT']
	#colstock=['blue','orange','red', 'pink','yellow', 'olive']
	stockquantity = [1,1,1,1,1,1]

	# some containers
	flist = [] # stock data series
	data1 = [] # html dump
	data2 = [] # html dump medium long
	data3 = [] # html dump short
	data4 = [] # html dump very short

	anrev = [] # annual revenue estimate
	networth = [] # updated portfolio networth
	netsigma = [] # netsigma for GBM
	prep = stockquantity	# portfolio representaion

	# clock the execution
	startTime = time.time()

	# Plot grayscale
	plt.style.use('grayscale')

	# api instructions
	quandl.ApiConfig.api_key = "DV8RpAAxoKayzstCQWyq"
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

	#plt.show()
		
	## MPI setting for each stock
	## ==========================
	NoProcess = 6
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
	
	#annualrev=np.sum(np.multiply(np.multiply(anrev,prep),252))
	#approxnetworth=np.sum(np.round_(np.multiply(networth,prep),2))

	#html_trailer = u"<h2>Estimated annual revenue</h2> <p> Assuming approximately 252 trading days (in USD): <strong>"+ red(np.round_(np.sum(np.multiply(np.multiply(anrev,prep),252)),2)) +u"</strong> chance to be correct: => 68.27%</p>" + u"<p> todays networth in USD:  <strong>"+ red(np.round_(np.sum(np.multiply(networth,prep)),2)) +u"</strong></p>" + u"<p> Linear Regression Model estimate networth in 252 trading days in USD :  <strong>"+ red(np.round_(annualrev+approxnetworth,2)) +u"</strong></p>"  + u"<p> annual interest rate in %:  <strong>"+ red(np.round_(np.multiply(np.divide(annualrev,approxnetworth),100),2)) +u"</strong> (no transaction costs assumed here & pre tax!)</p>"+ u"<p> Geometric Brownian Motion Model estimate networth in 252 trading days in USD:  <strong>"+ red(GBMestimate) +u"</strong></p>" + u"<p> annual interest rate in %:  <strong>"+ red(np.round_(np.multiply(np.divide(GBMestimate-approxnetworth,approxnetworth),100),2)) +u"</strong> (no transaction costs assumed here & pre tax!)</p>" 

	content = html_header +u"<p> Analysis performed on 252 trading days period </p>" +data_html1 + u"<p> Analysis performed on 150 trading days period </p>" +data_html2 +u"<p> Analysis performed on 50 trading days period </p>" +data_html3 + u"<p> Analysis performed on 10 trading days period </p>" +data_html4# + html_trailer

	#redmarker = u"<font color="+u"red"+"><b>!-</b></font>"
	contentr = content.replace(u"&lt;",u"<")
	contentr = contentr.replace(u"&gt;",u">")
	contentr = contentr.replace(u"<...",u"")

	# save data html to file
	patht = os.path.abspath("tableUpdate.html")
		
	Html_file= open(patht,"w")
	Html_file.write(contentr)
	# add date
	dateInfo = u"<p><i>" + "{:%B %d, %Y}".format(datetime.now()) + u"</i></p><p> &copy mh </p>"
	Html_file.write(dateInfo)
	Html_file.close()

	print("finished in overall time")
	print(str(time.time() - startTime) + u"sec")
	sys.exit(0)