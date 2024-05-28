import streamlit as st
import plotly.express as px
import os
import numpy as np      # To use np.arrays
import pandas as pd     # To use dataframes
from pandas.plotting import autocorrelation_plot as auto_corr

# To plot
import matplotlib as mpl
import seaborn as sns
import matplotlib.pyplot as plt  
mpl.pyplot.ion()

#For date-time
import math
import datetime
import calendar
from datetime import datetime
from datetime import timedelta

# Another imports if needs
import itertools
import statsmodels.api as sm
import statsmodels.tsa.api as smt
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn import metrics
from sklearn.linear_model import LinearRegression 
from sklearn import preprocessing

from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.tsa.arima_model import ARIMA
from pmdarima.utils import decomposed_plot
from pmdarima.arima import decompose
from pmdarima import auto_arima

import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_columns=100 # to see columns 

df_store = pd.read_csv('stores.csv') #store data
df_train = pd.read_csv('train.csv') # train set
df_features = pd.read_csv('features.csv') #external information

# merging 3 different sets
df = df_train.merge(df_features, on=['Store', 'Date'], how='inner').merge(df_store, on=['Store'], how='inner')
#print(df.shape)

df.drop(['IsHoliday_y'], axis=1,inplace=True) # removing dublicated column
df.rename(columns={'IsHoliday_x':'IsHoliday'},inplace=True) # rename the column
#print(df.head()) # last ready data set
#print(df.shape)
#print(df['Store'].nunique()) # number of different values
#print(df['Dept'].nunique()) # number of different values
df = df.loc[df['Weekly_Sales'] > 0] #to exclude negative sales values
#print(df.shape)

first_and_last_5 = df['Date'].head(5)._append(df['Date'].tail(5))
#print(first_and_last_5)
#Our data is from 5th of February 2010 to 26th of October 2012. 

df.sort_values(by='Weekly_Sales',ascending=False).head(5) #5 highest weekly sales

#print(df.isna().sum())
df = df.fillna(0) # filling null's with 0
#print(df.isna().sum()) #Last null check

df["Date"] = pd.to_datetime(df["Date"]) # convert to datetime
#df['week'] =df['Date'].dt.week
df['week'] = df['Date'].dt.isocalendar().week
df['month'] =df['Date'].dt.month
#df['month'] = df['Date'].dt.to_period('M').dt.month_name()
#df['month'] = df['Date'].dt.to_period('M').map(lambda x: x.month())
df['year'] =df['Date'].dt.year

df.to_csv('clean_data.csv') # assign new data frame to csv for using in Streamlit webAPP

def mae_test(test, pred):
    error = np.mean(np.abs(test - pred))
    return error

pd.options.display.max_columns=100 # to see columns 
df = pd.read_csv('clean_data.csv')
df.drop(columns=['Unnamed: 0'],inplace=True)
df['Date'] = pd.to_datetime(df['Date']) # changing datetime to divide if needs

### Time Series Model ###

df["Date"] = pd.to_datetime(df["Date"]) #changing data to datetime for decomposing
df.set_index('Date', inplace=True) #seting date as index

df_week = df.select_dtypes(include=[np.number]).resample('W').mean()
#df_week = df.resample('W').mean() #resample data as weekly
#df_week = df['Weekly_Sales'].resample('W').mean()

### Train - Test Split of Weekly Data ###

train_data = df_week[:int(0.7*(len(df_week)))] 
test_data = df_week[int(0.7*(len(df_week))):]

#rmse_test = np.sqrt(mean_squared_error(test_data_diff, y_pred))
#print("Root Mean Squared Error:",rmse_test)

# print("Mean Absolute Error:",mae_test(test_data_diff, y_pred))
# rmse_test = np.sqrt(mean_squared_error(test_data_diff, y_pred))
# print("Root Mean Squared Error:",rmse_test)

### Streamlit Web APP ### 
st.set_page_config(page_title="Prediction!!!", page_icon=":chart_with_upwards_trend:",layout="wide")

st.title(":chart_with_upwards_trend: Walmart Prediction EDA")
st.markdown('<style>div.block-container{padding-top:2rem;}</style>',unsafe_allow_html=True)

os.chdir(r"C:\Python312\Walmart Sales")
df = pd.read_csv("clean_data.csv", encoding = "ISO-8859-1")

col1, col2 = st.columns((2))
df["Date"] = pd.to_datetime(df["Date"])

# Getting the min and max date 
startDate = pd.to_datetime(df["Date"]).min()
endDate = pd.to_datetime(df["Date"]).max()

with col1:
    date1 = pd.to_datetime(st.date_input("Start Date", startDate))

with col2:
    date2 = pd.to_datetime(st.date_input("End Date", endDate))

df = df[(df["Date"] >= date1) & (df["Date"] <= date2)].copy()

st.sidebar.header("Choose your filter: ")
# Create for Types
type_s = st.sidebar.multiselect("Choose type of Sales", df["Type"].unique())
if not type_s:
    df2 = df.copy()
else:
    df2 = df[df["Type"].isin(type_s)]

# Create for Month
mon_s = st.sidebar.multiselect("Choose Month of Sales", df2["month"].unique())
if not mon_s:
    df3 = df2.copy()
else:
    df3 = df2[df2["month"].isin(mon_s)]

year_s = st.sidebar.multiselect("Choose Year of Sales",df3["year"].unique())

# Filter the data based on Type, Month and Year

if not type_s and not mon_s and not year_s:
    filtered_df = df
elif not mon_s and not year_s:
    filtered_df = df[df["Type"].isin(type_s)]
elif not type_s and not year_s:
    filtered_df = df[df["month"].isin(mon_s)]
elif mon_s and year_s:
    filtered_df = df3[df["month"].isin(mon_s) & df3["year"].isin(year_s)]
elif type_s and year_s:
    filtered_df = df3[df["Type"].isin(type_s) & df3["year"].isin(year_s)]
elif type_s and mon_s:
    filtered_df = df3[df["Type"].isin(type_s) & df3["month"].isin(mon_s)]
elif year_s:
    filtered_df = df3[df3["year"].isin(year_s)]
else:
    filtered_df = df3[df3["Type"].isin(type_s) & df3["month"].isin(mon_s) & df3["year"].isin(year_s)]

dept_df = filtered_df.groupby(by = ["Dept"], as_index = False)["Weekly_Sales"].sum()

with col1:
    st.subheader("Department wise Sales")
    fig = px.bar(dept_df, x = "Dept", y = "Weekly_Sales", text = ['${:,.2f}'.format(x) for x in dept_df["Weekly_Sales"]],
                 template = "seaborn")
    st.plotly_chart(fig,use_container_width=True, height = 200)

with col2:
    st.subheader("Month wise Sales")
    fig = px.pie(filtered_df, values = "Weekly_Sales", names = "month", hole = 0.5)
    fig.update_traces(text = filtered_df["month"], textposition = "outside")
    st.plotly_chart(fig,use_container_width=True)

st.subheader('Store wise Sales')
store_df = filtered_df.groupby(by = ["Store"], as_index = False)["Weekly_Sales"].sum()
#linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Weekly_Sales"].sum()).reset_index()
figx = px.line(store_df, x = "Store", y="Weekly_Sales", labels = {"Weekly_Sales": "Sales"},height=500, width = 1000,template="gridon")
st.plotly_chart(figx,use_container_width=True)

filtered_df["month_year"] = filtered_df["Date"].dt.to_period("M")
st.subheader('Time Series Analysis')

linechart = pd.DataFrame(filtered_df.groupby(filtered_df["month_year"].dt.strftime("%Y : %b"))["Weekly_Sales"].sum()).reset_index()
fig2 = px.line(linechart, x = "month_year", y="Weekly_Sales", labels = {"Weekly_Sales": "Amount"},height=500, width = 1000,template="gridon")
st.plotly_chart(fig2,use_container_width=True)

st.subheader('Prediction of Weekly Sales Using Auto-ARIMA')

### Train Test Split For Auto-Arima Model ###
df_week_diff = df_week['Weekly_Sales'].diff().dropna() #creating difference values
train_data_diff = df_week_diff [:int(0.7*(len(df_week_diff )))]
test_data_diff = df_week_diff [int(0.7*(len(df_week_diff ))):]

model_auto_arima = auto_arima(train_data_diff, trace=True,start_p=0, start_q=0, start_P=0, start_Q=0,
                  max_p=20, max_q=20, max_P=20, max_Q=20, seasonal=True,maxiter=200,
                  information_criterion='aic',stepwise=False, suppress_warnings=True, D=1, max_D=10,
                  error_action='ignore',approximation = False)
model_auto_arima.fit(train_data_diff)

y_pred = model_auto_arima.predict(n_periods=len(test_data_diff))
y_pred = pd.DataFrame(y_pred,index = test_data.index,columns=['Prediction'])
# plt.figure(figsize=(20,6))
# plt.title('Prediction of Weekly Sales Using Auto-ARIMA', fontsize=20)
# plt.plot(train_data_diff, label='Train')
# plt.plot(test_data_diff, label='Test')
# plt.plot(y_pred, label='Prediction of ARIMA')
# plt.legend(loc='best')
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Weekly Sales', fontsize=14)
# plt.show()

# Create a figure and axis for the plot
st.set_option('deprecation.showPyplotGlobalUse', False)
fig, ax = plt.subplots(figsize=(20,6))
# Plot the train data
ax.plot(train_data_diff.index, train_data_diff.values, label='Train')
# Plot the test data
ax.plot(test_data_diff.index, test_data_diff.values, label='Test')
# Plot the predicted data
ax.plot(y_pred.index, y_pred.values, label='Prediction of ARIMA')
# Set the plot title and labels
ax.set_xlabel('Date', fontsize=14)
ax.set_ylabel('Weekly Sales', fontsize=14)
# Set the legend
ax.legend(loc='best')
# Convert the plot to a Streamlit figure
st.pyplot(fig)
# Set the plot height and width
#st.pyplot(height=500, width=1000)

#Create a line chart using Plotly Express
# fig = px.line(x=train_data_diff.index, y=train_data_diff.values, labels={'x': 'Date', 'y': 'Weekly Sales'}, height=500, width=1000, template="gridon")
# fig.add_scatter(x=train_data_diff.index, y=train_data_diff.values, mode='lines', name='Train', line=dict(color='blue'))
# fig.add_scatter(x=test_data_diff.index, y=test_data_diff.values, mode='lines', name='Test')
# fig.add_scatter(x=y_pred.index, y=y_pred.values, mode='lines', name='Prediction of ARIMA')
# # Display the chart in Streamlit
# st.plotly_chart(fig, use_container_width=True)

st.subheader('Prediction of Weekly Sales Using Exponential Smoothing')
### Exponential Smoothing Model ###

model_holt_winters = ExponentialSmoothing(train_data_diff, seasonal_periods=20, seasonal='additive',
                                           trend='additive',damped=True).fit() #Taking additive trend and seasonality.
y_pred = model_holt_winters.forecast(len(test_data_diff))# Predict the test data

# #Visualize train, test and predicted data.

# plt.figure(figsize=(20,6))
# plt.title('Prediction of Weekly Sales using ExponentialSmoothing', fontsize=20)
# plt.plot(train_data_diff, label='Train')
# plt.plot(test_data_diff, label='Test')
# plt.plot(y_predz, label='Prediction using ExponentialSmoothing')
# plt.legend(loc='best')
# plt.xlabel('Date', fontsize=14)
# plt.ylabel('Weekly Sales', fontsize=14)
# plt.show()

# Create a line chart using Plotly Express
fig = px.line(x=train_data_diff.index, y=train_data_diff.values, labels={'x': 'Date', 'y': 'Weekly Sales'}, height=500, width=1000, template="gridon")
fig.add_scatter(x=train_data_diff.index, y=train_data_diff.values, mode='lines', name='Train', line=dict(color='blue'))
fig.add_scatter(x=test_data_diff.index, y=test_data_diff.values, mode='lines', name='Test')
fig.add_scatter(x=y_pred.index, y=y_pred.values, mode='lines', name='Prediction of ExponentialSmoothing')
# Display the chart in Streamlit
st.plotly_chart(fig, use_container_width=True)
