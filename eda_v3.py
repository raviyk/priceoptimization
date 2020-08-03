# -*- coding: utf-8 -*-
"""
Created on Fri Jul 31 13:37:48 2020

@author: ravkiran
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

#Read datasets
loc = 'C:/Users/ravkiran/Downloads/'
df = pd.read_csv(loc+'ecom-elasticity-data.tsv', sep='\t', header=None)
df.columns = ['Item_ID', 'Category', 'Date', 'Average_Price', 'Units_Sold']

#Read holiday dataset
loc = 'C:/Users/ravkiran/Downloads/7476_10641_bundle_archive/'
df_holidays = pd.read_csv(loc+'usholidays.csv', header=0)
df_holidays.columns=['temp', 'Date', 'Holiday_name']
df_holidays.nunique() #13 holidays

df = pd.merge(df, 
              df_holidays[['Date', 'Holiday_name']], 
              on='Date',
              how='left')

# Add a new column named 'Holiday_flag' 
df['Holiday_flag'] = [0 if x.isna else 1 for x in df['Holiday_name']] 

df['Holiday_flag'] = ~df['Holiday_name'].isna()

#multivariate scatter plot
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_sample['Date'],df_sample['Units_Sold'])
ax.set_xlabel('Date')
ax.set_ylabel('Units sold')
plt.show()

#outlier treatment - z score
z = np.abs(stats.zscore(df))
print(z)

#outlier treatment - IQR
df_in = df_sample[['Units_Sold', 'Average_Price']]

Q1 = df_in.quantile(0.25)
Q3 = df_in.quantile(0.75)
IQR = Q3 - Q1
print(IQR)
print((df_in < (Q1 - 1.5 * IQR)) |(df_in > (Q3 + 1.5 * IQR)))
df_out = (df_in < (Q1 - 1.5 * IQR)) |(df_in > (Q3 + 1.5 * IQR))

df_sample[df_out['Average_Price']==True]
df_sample[df_out['Units_Sold']==True]
df_sample_o = df_sample[~((df_out['Units_Sold']==True) | (df_out['Average_Price']==True))]
print(df_sample.shape)
print(df_sample_o.shape)

#multivariate scatter plot after outlier treatment
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter((df_sample_o['Average_Price']),df_sample_o['Units_Sold'])
ax.set_xlabel('Avg. price per unit of item')
ax.set_ylabel('Units sold')
plt.show()

#
print(df_sample.Average_Price.describe())
print(df_sample_o.Average_Price.describe())

print(df_sample.Units_Sold.describe())
print(df_sample_o.Units_Sold.describe())


#count unique item ids in the data
df['Item_ID'].nunique()
#420

df[df['Category']=='Diapers']['Item_ID'].nunique()
#100 items of Diaper category exist
df.groupby('Category').nunique('Item_ID')

df_sample.groupby('Category').nunique('Item_ID')

df['Date'].nunique() 
#1100

df.info()
df.describe()

#Share of income from 
df_sample['total_income'] = df_sample['Average_Price']*df_sample['Units_Sold']
df['total_income'] = df['Average_Price']*df['Units_Sold']

df.groupby('Category')['total_income'].sum()
df[df['Category']=='Diapers'].groupby('Item_ID')['total_income'].sum()

#Summarize the quantity at Price points
df_demand_tbl=df_sample.groupby(['Date', 'Average_Price'])['Units_Sold'].sum().reset_index()
type(df_demand_tbl)
df_demand_tbl.head(20)

#multivariate scatter plot after outlier treatment
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_demand_tbl['Units_Sold'], (df_demand_tbl['Average_Price']))
ax.set_xlabel('Qty')
ax.set_ylabel('Price')
plt.show()

#multivariate scatter plot after outlier treatment, with log transformation
import numpy as np
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(np.log(df_demand_tbl['Units_Sold']), np.log(df_demand_tbl['Average_Price']))
ax.set_xlabel('log(Qty)')
ax.set_ylabel('log(Price)')
plt.show()

###Model
import scipy.stats
from sklearn.linear_model import LinearRegression
X = df_demand_tbl['Units_Sold']
X.shape #(1100,)
X = np.array(X).reshape(-1, 1)
y = df_demand_tbl['Average_Price']
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_

#Model using Monthy data
df_sample['year'] = pd.DatetimeIndex(df_sample['Date']).year
df_sample['month'] = pd.DatetimeIndex(df_sample['Date']).month
df_sample['year_month'] = pd.to_datetime(df_sample['Date']).dt.to_period('M')
#Summarize the quantity at Price points
df_demand_tbl=df_sample.groupby(['year_month', 'Average_Price'])['Units_Sold'].sum().reset_index()

#level-level
X = df_demand_tbl['Units_Sold']
X = np.array(X).reshape(-1, 1)
y = df_demand_tbl['Average_Price']
reg = LinearRegression().fit(X, y)
reg.score(X, y)
reg.coef_
reg.intercept_

#log-level
reg = LinearRegression().fit(X, np.log(y))
reg.score(X, y)
reg.coef_
reg.intercept_
y_pred = reg.predict(X)

from sklearn.metrics import r2_score
r2_score(y, y_pred)

#level-log
reg = LinearRegression().fit(np.log(X), y)
reg.score(np.log(X), y)
reg.coef_
reg.intercept_
y_pred = reg.predict(np.log(X))

from sklearn.metrics import r2_score
r2_score(y, y_pred)

#log-log
reg = LinearRegression().fit(np.log(X), np.log(y))
reg.score(np.log(X), np.log(y))
reg.coef_
reg.intercept_
y_pred = reg.predict(np.log(X))

from sklearn.metrics import r2_score
r2_score(np.log(y), y_pred)

#

#Model using Quarterly data
#df_sample['year'] = pd.DatetimeIndex(df_sample['Date']).year
#df_sample['month'] = pd.DatetimeIndex(df_sample['Date']).month
#df_sample['quarter'] = pd.to_datetime(df_sample['Date']).dt.to_period('Q')
#df_sample['year_month'] = pd.to_datetime(df_sample['Date']).dt.to_period('M')
df['quarter'] = pd.to_datetime(df['Date']).dt.to_period('Q')
df['year_month'] = pd.to_datetime(df['Date']).dt.to_period('M')

#Summarize the quantity at Price points

## Define a lambda function to compute the weighted mean:
#wm = lambda x: np.average(x, weights=df_sample.loc[x.index, "Units_Sold"])
wm = lambda x: np.average(x, weights=df.loc[x.index, "Units_Sold"])

# Groupby and [aggregate with namedAgg][1]:
#df_demand_tbl_mo = df_sample.groupby(["year_month", "Item_ID", "Category"]).agg(Units_Sold=("Units_Sold", "sum"),  
#                                                      price_weighted_mean=("Average_Price", wm)).reset_index()

df_demand_tbl_mo = df.groupby(["year_month", "Item_ID", "Category"]).agg(Units_Sold=("Units_Sold", "sum"),  
                                                      price_weighted_mean=("Average_Price", wm)).reset_index()
n_items = df_demand_tbl_mo['Item_ID'].nunique()
items = df_demand_tbl_mo['Item_ID'].unique()

score = {}
elasticity = {}
for item in items:
    df_ =df_demand_tbl_mo[df_demand_tbl_mo['Item_ID']==item]
    X = df_['Units_Sold']
    X = np.array(X).reshape(-1, 1)
    y = df_['price_weighted_mean']

    reg = LinearRegression().fit(np.log(X), np.log(y))
    score[item] = reg.score(np.log(X), np.log(y))
    elasticity[item] = reg.coef_

    
######################
X = df_demand_tbl_mo['Units_Sold']
X = np.array(X).reshape(-1, 1)
y = df_demand_tbl_mo['price_weighted_mean']

#log-log
reg = LinearRegression().fit(np.log(X), np.log(y))
print(reg.score(np.log(X), np.log(y)))
print('Price elasticity', reg.coef_)
print(reg.intercept_)
y_pred = reg.predict(np.log(X))
r2_score(np.log(y), y_pred)

# Groupby and [aggregate with namedAgg][1]:
df_demand_tbl_qtr = df_sample.groupby(["quarter", "Item_ID", "Category"]).agg(Units_Sold=("Units_Sold", "sum"),  
                                                      price_weighted_mean=("Average_Price", wm)).reset_index()

X = df_demand_tbl_qtr['Units_Sold']
X = np.array(X).reshape(-1, 1)
y = df_demand_tbl_qtr['price_weighted_mean']

#log-log
reg = LinearRegression().fit(np.log(X), np.log(y))
print('R2: ', reg.score(np.log(X), np.log(y)))
print('Price elasticity', reg.coef_)
print('Intercept', reg.intercept_)

###Loop for Quarterly, Item numbers
df_demand_tbl_qtr = df.groupby(["quarter", "Item_ID", "Category"]).agg(Units_Sold=("Units_Sold", "sum"),  
                                                      price_weighted_mean=("Average_Price", wm)).reset_index()
n_items = df_demand_tbl_qtr['Item_ID'].nunique()
items = df_demand_tbl_qtr['Item_ID'].unique()
# dropping ALL duplicate item category combinations 
df_cat_lkp = df_demand_tbl_qtr[["Item_ID", "Category"]].drop_duplicates() 
score = {}
elasticity = {}
for item in items:
    df_ =df_demand_tbl_qtr[df_demand_tbl_qtr['Item_ID']==item]
    X = df_['Units_Sold']
    X = np.array(X).reshape(-1, 1)
    y = df_['price_weighted_mean']

    reg = LinearRegression().fit(np.log(X), np.log(y))
    score[item] = reg.score(np.log(X), np.log(y))
    elasticity[item] = reg.coef_
pde = pd.DataFrame([score, elasticity]).T.reset_index()
pde.columns = ['Item_ID', 'r2_score', 'elasticity']
pde_qtr=pd.merge(pde,  
          df_cat_lkp,  
          on ='Item_ID',  
          how ='left') 

###Plot the Demand curve [Quarterly data]
df_demand_tbl_cat_qtr = df.groupby(["quarter", "Category"]).agg(Units_Sold=("Units_Sold", "sum"),  
                                                      price_weighted_mean=("Average_Price", wm)).reset_index()
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df_demand_tbl_cat_qtr[df_demand_tbl_cat_qtr['Category']=='Diapers']['Units_Sold'], df_demand_tbl_cat_qtr[df_demand_tbl_cat_qtr['Category']=='Diapers']['price_weighted_mean'])
ax.set_xlabel('Qty')
ax.set_ylabel('Price')
plt.show()

###Derive summary from PDE numbers
x = pde_qtr[pde_qtr['elasticity'] < 0].groupby(["Category"])['Item_ID'].count().reset_index()
x.columns = ['Category', 'Normal_goods']
y = pde_qtr.groupby(["Category"])['Item_ID'].count().reset_index()
y.columns = ['Category', 'Total_goods']

df_plot = pd.merge(x,
                  y,
                  on='Category',
                  how='left')

ax = df_plot.plot.barh(x='Category')

#Histogram of elasticity/coeff 
x=np.array(pde_qtr['elasticity'])
histedges = np.array([-0.10, -0.08, -0.06, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
plt.hist(x, histedges,  normed=True)  # `density=False` would make counts
plt.ylabel('Count of Items')
plt.xlabel('Data')

n, bins, patches = plt.hist(x, pde_qtr['elasticity'], normed=True)

plt.hist(pde_qtr['elasticity'],density=False, bins=10,  histtype='bar', ec='black') 
plt.axis([-0.10, -0.08, -0.6, -0.04, -0.02, 0, 0.02, 0.04, 0.06, 0.08, 0.10])
#axis([xmin,xmax,ymin,ymax])
plt.xlabel('# Items')
plt.ylabel('Elasticity')

#Plot revenue with price
df_demand_cat_tbl_qtr = df.groupby(["quarter", "Category"]).agg(Units_Sold=("Units_Sold", "sum"),  
                                                      price_weighted_mean=("Average_Price", wm)).reset_index()

df = df_demand_tbl_qtr[df_demand_tbl_qtr['Item_ID']=='C1144']
df['Revenue']= df['Units_Sold']*df['price_weighted_mean']
fig, ax = plt.subplots(figsize=(16,8))
ax.scatter(df['price_weighted_mean'], (df['Revenue']))
ax.set_xlabel('price')
ax.set_ylabel('revenue')
plt.show()
