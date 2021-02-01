# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Industrial materials economics analysis

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns; sns.set()
import csv
from random import randint
import random
import chart_studio
import plotly.express as px


# %%
username = 'adam_misik' # your username
api_key = 'vjJDc0jFwLzCOHuwiM4P' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)


# %%
data = pd.read_excel('DATA_Industry_Materials.xlsx')
data = data[6:]
data.head(10)

# %% [markdown]
# ### Analysis of material price and quantities

# %%
data
# %%
import plotly.graph_objects as go


fig = go.Figure(go.Scatter(x=data['Unnamed: 3'], y=data['Unnamed: 5'], name = '2018',mode='markers', hovertemplate='<b>Price: </b>: €%{y:.2f}'+
    '<br><b>Enterprises: </b>: %{x}<br>'+' <b>Material: <b>' + '<i>%{text}</i>',text=data['Unnamed: 1']))
fig.update_layout(xaxis_title='Number of enterprises', yaxis_title='Price in €')
fig.add_trace(go.Scatter(x=data['Unnamed: 9'], y=data['Unnamed: 11'], mode='markers',name='2017',hovertemplate='<b>Price: </b>: €%{y:.2f}'+
    '<br><b>Enterprises: </b>: %{x}<br>'+' <b>Material: <b>' + '<i>%{text}</i>', text=data['Unnamed: 1']))
fig.add_trace(go.Scatter( x=data['Unnamed: 15'], y=data['Unnamed: 17'], mode='markers',name='2016',hovertemplate='<b>Price: </b>: €%{y:.2f}'+
    '<br><b>Enterprises: </b>: %{x}<br>'+' <b>Material: <b>' + '<i>%{text}</i>', text=data['Unnamed: 1']))

# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'materials_industry_prices', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/4/') #change to your url

# %% [markdown]
# ### Temporal analysis of interesting materials/goods

# %%
print(data[data['Unnamed: 1']=='Motor vehicles and engines'].index.values-6)


# %%
motor_vehicles_data = data.iloc[[228]]
motor_vehicles_data.head()


# %%
years = np.linspace(2014,2018,4).astype(np.int32)
motor_vehicles_data_prices = [motor_vehicles_data['Unnamed: 3'], motor_vehicles_data['Unnamed: 7'],motor_vehicles_data['Unnamed: 11'],motor_vehicles_data['Unnamed: 15']]
motor_vehicles_data_enterprises = [motor_vehicles_data['Unnamed: 2'], motor_vehicles_data['Unnamed: 6'],motor_vehicles_data['Unnamed: 10'],motor_vehicles_data['Unnamed: 14']]


# %%
motor_vehicles_data_enterprises = np.array(motor_vehicles_data_enterprises).astype(np.int32)
motor_vehicles_data_prices  = np.array(motor_vehicles_data_prices).astype(np.int32)


# %%
motor_vehicles_df = pd.DataFrame(list(zip(years,motor_vehicles_data_prices,motor_vehicles_data_enterprises)))


# %%
motor_vehicles_df = motor_vehicles_df.rename(columns={0:'Years',1:'Prices',2:'Enterprises'})


# %%
motor_vehicles_df = motor_vehicles_df.astype(int)


# %%
motor_vehicles_df = motor_vehicles_df.astype(int)

motor_vehicles_df['Prices'] = motor_vehicles_df['Prices'].astype(str).astype(int)
motor_vehicles_df['Enterprises'] = motor_vehicles_df['Enterprises'].astype(str).astype(int)


motor_vehicles_df.dtypes


# %%
fig = px.scatter(motor_vehicles_df, x='Years', y='Prices', hover_name = 'Enterprises')
fig.update_layout(title='Motor vehicles and engines production for sale')
fig.update_layout(xaxis_title='Years', yaxis_title='Price in TSD €')


# %%



