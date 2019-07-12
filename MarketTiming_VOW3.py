# pip install datareader
import pandas as pd
pd.core.common.is_list_like = pd.api.types.is_list_like
# quandl api explore
import quandl
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# api instructions
quandl.ApiConfig.api_key = "DV8RpAAxoKayzstCQWyq"
end = datetime.now()
start = end - timedelta(days=365)

# frankfurt stock exchange
mydata = quandl.get('FSE/VOW3_X', start_date = start, end_date = end)
f = mydata.reset_index()

# timeseries
plt.figure(1)
f = pd.Series(f.Close.values,f.Date)
f.plot()
plt.show()