# pip install datareader
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import numpy as np
import time

# do fitting
import plotly.plotly as py
import plotly.graph_objs as go

from matplotlib import pylab
from scipy.optimize import curve_fit
from datetime import datetime, timedelta

#stock of interest
stock=['V','SAP','JPM','MSFT','AAPL','INTC','MITT']
colstock=['blue','orange','green','red', 'pink','yellow', 'olive']
stockquantity = [1,1,1,1,1,1,1]

# period of analysis
end = datetime.now()
start = end - timedelta(days=300)

# some containers
flist = [] # stock data series
data = [] # html dump
data2 = [] # html dump medium long
data3 = [] # html dump short
data4 = [] # html dump very short
anrev = [] # annual revenue estimate
networth = [] # updated portfolio networth
prep = stockquantity	# portfolio representaion

import numpy as np
import pandas as pd

def generate_html_with_table(data, columns_or_rows = 1, \
                             column_name_prefix = 'Column', \
                             span_axis = 1, \
                             showOutput = True):
    """
    This function returns a pandas.DataFrame object and a 
    generated html from the data based on user specifications.
    
    #Example:
      data_html, data_df = generate_html_with_table(data, columns_or_rows, column_name_prefix, span_axis, showOutput)
      # To suppress output and skip the DataFrame:
      # span data along a row first
        columns = 4
        columns_or_rows = columns
        data_html, _ = generate_html_with_table(data, columns_or_rows, column_name_prefix, 1, False)  
      # span data along a column first
        rows = 4
        columns_or_rows = rows
        data_html, _ = generate_html_with_table(data, columns_or_rows, column_name_prefix, 0, False)   
      
    # Inputs: 
        1. data:               Data
           (dtype: list)
           
      **Optional Input Parameters:**
        2. columns_or_rows:            Number of Columns or rows
           (dtype: int)                columns: span_axis = 1
           (DEFAULT: 1)                rows:    span_axis = 0
        3. column_name_prefix: The Prefix for Column headers
           (dtype: string)
           (DEFAULT: 'Column')
        4. span_axis:          The direction along which the elements span.
           (dtype: int)        span_axis = 0 (span along 1st column, then 2nd column and so on)
           (DEFAULT: 1)        span_axis = 1 (span along 1st row, then 2nd row and so on)
        5. showOutput:         (True/False) Whether to show output or not. Use 
           (dtype: boolean)                   False to suppress printing output.
           (DEFAULT: True)
                                                              
    # Outputs:
        data_html: generated html
        data_df:   generated pandas.DataFrame object
        
    # Author: Sugato Ray 
    Github: https://github.com/sugatoray
    Repository/Project: CodeSnippets
    
    """
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
    column_names = [column_name_prefix + '_{}'.format(i) \
                    for i in np.arange(columns)]    
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
    if showOutput:
        print('Elements: {}\nColumns: {}\nRows: {}'.format(elements, \
                                                           columns, \
                                                           rows))
        print('Column Names: {}'.format(column_names))
        print('\nPandas DataFrame: ')
        #display(data_df)
        print('\nHTML Generated: \n\n' + data_html)
        
    return (data_html, data_df)
	
def DK(f):
	return u"KEEP"

########################################
### body ###############################
########################################

# long analysis
plt.figure(1)
for i in range(len(stock)):
	f = web.DataReader(stock[i], 'robinhood', start, end)

	# nice looking timeseries (DataFrame to panda Series)
	f = f.reset_index()
	#f = pd.Series(f.Close.values,f.Date)
	a = [float(f.close_price.values[ii].encode('ascii')) for ii in range(len(f.close_price.values))]
	f = pd.Series(a, f.begins_at.values)

	print "Start: Year, Month, Day, Time"
	print str(start)
	f.plot(label=stock[i],color=colstock[i]);
	plt.legend()
	plt.ylabel('price in [USD]')
	# dump to container
	flist.append(f)

plt.figure(2)
for i in range(len(stock)):
	# extract last data and do linear regression
	y = flist[i].values
	x = flist[i].index.date #datetime index
	
	#calc std
	sigma = np.std(y)

	d = np.linspace(1,len(x),len(x))

	#lin reg
	def linear_func(x, a, b):
		return a*x +b

	popt, pcov = curve_fit(linear_func, d, y)
	print str(popt)

	xx = d#-np.max(d)
	yy= linear_func(xx, *popt)
	
	days= xx-np.max(d)
	
	plt.plot(days, yy,label=stock[i], color=colstock[i])
	
	pylab.title('Linear Fit')
	plt.legend()
	plt.ylabel('price in [USD]')
	
	# dump analysis results
	data.append(stock[i]) # label
	data.append(str(popt[0])) # av daily revenue
	anrev.append(popt[0]) # anual estimate revenue
	data.append(str(sigma)) # volatility
	data.append(str(y[-1]))	# today value
	networth.append(y[-1])
	data.append(str(np.round_((y[-1]*0.025)/popt[0],0)))	#trading penalty
	
	# drop or keep analysis
	data.append(DK(flist[i]))
	
	# portfolio rep
	data.append(str(prep[i]))

#medium long analysis 150d	

plt.figure(3)
for i in range(len(stock)):
	# extract last data and do linear regression
	y = flist[i].values[len(flist[i].values)-150:len(flist[i].values)]
	x = flist[i].index.date[len(flist[i].values)-150:len(flist[i].values)] #datetime index
	
	#calc std
	sigma = np.std(y)

	d = np.linspace(1,len(x),len(x))

	#lin reg
	def linear_func(x, a, b):
		return a*x +b

	popt, pcov = curve_fit(linear_func, d, y)
	print str(popt)

	xx = d#-np.max(d)
	yy= linear_func(xx, *popt)
	
	days= xx-np.max(d)
	
	plt.plot(days, yy,label=stock[i], color=colstock[i])
	
	pylab.title('Linear Fit')
	plt.legend()
	plt.ylabel('price in [USD]')
	
	# dump analysis results
	data2.append(stock[i]) # label
	data2.append(str(popt[0])) # av daily revenue
	#anrev.append(popt[0]) # anual estimate revenue
	data2.append(str(sigma)) # volatility
	data2.append(str(y[-1]))	# today value
	#networth.append(y[-1])
	data2.append(str(np.round_((y[-1]*0.025)/popt[0],0)))	#trading penalty
	
	# drop or keep analysis
	data2.append(DK(flist[i]))
	
	# portfolio rep
	data2.append(str(prep[i]))

# short analysis	
# some containers

plt.figure(4)
for i in range(len(stock)):
	# extract last data and do linear regression
	y = flist[i].values[len(flist[i].values)-50:len(flist[i].values)]
	x = flist[i].index.date[len(flist[i].values)-50:len(flist[i].values)] #datetime index
	
	#calc std
	sigma = np.std(y)

	d = np.linspace(1,len(x),len(x))

	#lin reg
	def linear_func(x, a, b):
		return a*x +b

	popt, pcov = curve_fit(linear_func, d, y)
	print str(popt)

	xx = d#-np.max(d)
	yy= linear_func(xx, *popt)
	
	days= xx-np.max(d)
	
	plt.plot(days, yy,label=stock[i], color=colstock[i])
	
	pylab.title('Linear Fit')
	plt.legend()
	plt.ylabel('price in [USD]')
	
	# dump analysis results
	data3.append(stock[i]) # label
	data3.append(str(popt[0])) # av daily revenue
	#anrev.append(popt[0]) # anual estimate revenue
	data3.append(str(sigma)) # volatility
	data3.append(str(y[-1]))	# today value
	#networth.append(y[-1])
	data3.append(str(np.round_((y[-1]*0.025)/popt[0],0)))	#trading penalty
	
	# drop or keep analysis
	data3.append(DK(flist[i]))
	
	# portfolio rep
	data3.append(str(prep[i]))
	
plt.figure(5)
for i in range(len(stock)):
	# extract last data and do linear regression
	y = flist[i].values[len(flist[i].values)-10:len(flist[i].values)]
	x = flist[i].index.date[len(flist[i].values)-10:len(flist[i].values)] #datetime index
	
	#calc std
	sigma = np.std(y)

	d = np.linspace(1,len(x),len(x))

	#lin reg
	def linear_func(x, a, b):
		return a*x +b

	popt, pcov = curve_fit(linear_func, d, y)
	print str(popt)

	xx = d#-np.max(d)
	yy= linear_func(xx, *popt)
	
	days= xx-np.max(d)
	
	plt.plot(days, yy,label=stock[i], color=colstock[i])
	
	pylab.title('Linear Fit')
	plt.legend()
	plt.ylabel('price in [USD]')
	
	# dump analysis results
	data4.append(stock[i]) # label
	data4.append(str(popt[0])) # av daily revenue
	#anrev.append(popt[0]) # anual estimate revenue
	data4.append(str(sigma)) # volatility
	data4.append(str(y[-1]))	# today value
	#networth.append(y[-1])
	data4.append(str(np.round_((y[-1]*0.025)/popt[0],0)))	#trading penalty
	
	# drop or keep analysis
	data4.append(DK(flist[i]))
	
	# portfolio rep
	data4.append(str(prep[i]))
	
plt.figure(6)
for i in [0, 1, 3]:
	# extract last data and do nonlinear regression
	y = flist[i].values
	x = flist[i].index.date #datetime index
	
	#calc std
	sigma = np.std(y)

	d = np.linspace(1,len(x),len(x))

	#lin reg
	def linear_func(x, a, b):
		return a*x +b

	popt, pcov = curve_fit(linear_func, d, y)
	print str(popt)

	xx = d#-np.max(d)
	yy = linear_func(xx, *popt)
	
	days= xx-np.max(d)
	
	plt.plot(days, yy,label=stock[i], color=colstock[i])
	plt.plot(days, yy+ sigma ,linestyle='dashed', color=colstock[i])
	plt.plot(days, yy- sigma ,linestyle='dashed', color=colstock[i])
	
	pylab.title('Linear Fit')
	plt.legend()
	plt.ylabel('price in [USD]')
	
#plt.show()

#data = ['one','two','three','four','five','six','seven','eight','nine']
columns = 7                   # Number of Columns
columns_or_rows = columns
column_name_prefix = 'Column' # Prefix for Column headers
span_axis = 1                 # Span along a row (1) or a column (0) first
showOutput = False            # Use False to suppress printing output

# Generate HTML
data_html1, data_df1 = generate_html_with_table(data, columns_or_rows, column_name_prefix, span_axis, showOutput)
data_html2, data_df1 = generate_html_with_table(data2, columns_or_rows, column_name_prefix, span_axis, showOutput)
data_html3, data_df1 = generate_html_with_table(data3, columns_or_rows, column_name_prefix, span_axis, showOutput)
data_html4, data_df1 = generate_html_with_table(data4, columns_or_rows, column_name_prefix, span_axis, showOutput)

html_header = u"<h1>Stock market analysis</h1> <p> Legend to the market analysis result </p> <ul>  <li>Column_0 = stock of interest </li>  <li>Column_1 = average daily profit in USD</li> <li>Column_2 = volatility in USD (remains within bound: chance => 68.27%; 2 sigma => 95.45 %) </li><li>Column_3 = value today in USD </li><li>Column_4 = trading penalty in days (2.5% transaction cost assumed) </li><li>Column_5 = KEEP or DROP indication (NotImplemented yet) </li><li>Column_6 = portfolio representaion in QTY </li></ul>"

annualrev=np.sum(np.multiply(np.multiply(anrev,prep),252))
approxnetworth=np.sum(np.round_(np.multiply(networth,prep),2))

html_trailer = u"<h2>Estimated annual revenue</h2> <p> Assuming approximately 252 trading days (in USD): <strong>"+ str(np.round_(np.sum(np.multiply(np.multiply(anrev,prep),252)),2)) +u"</strong> chance to be correct: => 68.27%</p>" + u"<p> todays networth in USD:  <strong>"+ str(np.round_(np.sum(np.multiply(networth,prep)),2)) +u"</strong></p>" + u"<p> annual interest rate in %:  <strong>"+ str(np.multiply(np.round_(np.divide(annualrev,approxnetworth),2),100)) +u"</strong> (no transaction costs assumend here & pre tax!)</p>"

content = html_header +u"<p> Analysis performed on 252 trading days period </p>" +data_html1 + u"<p> Analysis performed on 150 trading days period </p>" +data_html2 +u"<p> Analysis performed on 50 trading days period </p>" +data_html3 + u"<p> Analysis performed on 10 trading days period </p>" +data_html4 + html_trailer

redmarker = u"<font color="+u"red"+"><b>!-</b></font>"
contentr = content.replace(u"-",redmarker)

# save data html to file
Html_file= open("table.html","w")
Html_file.write(contentr)
Html_file.close()
