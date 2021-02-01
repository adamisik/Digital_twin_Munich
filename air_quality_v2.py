# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Analysis of air quality in Munich
# %% [markdown]
# Based on: https://aqicn.org/data-platform/register/

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from random import randint
import random
import chart_studio


# %%
username = 'adam_misik' # your username
api_key = 'vjJDc0jFwLzCOHuwiM4P' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)


# %%
data = pd.read_csv('munich-air-quality.csv',sep=r'\s*,\s*',
                           header=0, encoding='ascii', engine='python')
data = data.fillna(0)
data.head(10) #temporal data!


# %%
data['date'] = pd.to_datetime(data['date'])


# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy
from sklearn.linear_model import LinearRegression
import time
import datetime as dt
import plotly.express as px

fig = make_subplots(rows=3, cols=1,  shared_xaxes=True)

fig.add_trace(
    go.Scatter(x=data['date'], y=data['pm10'], mode="markers",name="Particulates (PM10) levels"),
    row=1, col=1
)


fig.add_trace(
    go.Scatter(x=data['date'], y=data['o3'], mode="markers",name="Ozone levels"),
    row=2, col=1
)

fig.add_trace(
    go.Scatter(x=data['date'], y=data['no2'], mode="markers",name="NO2 levels"),
    row=3, col=1
)

fig.update_yaxes(title_text="Levels [µg/m3]")
fig.update_xaxes(title_text="Date")
fig.update_layout(height=600, width=800)

fig.show()


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'air_quality_metrics_overview', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/14/') #change to your url

# %% [markdown]
# ## Air quality metrics with Trends

# %%
import plotly.express as px

fig = px.scatter(data, x="date", y="pm10",trendline="lowess", trendline_color_override="red", title="Particulates (PM10) levels")
fig.update_yaxes(title_text="Levels [µg/m3]")
fig.update_xaxes(title_text="Date")
fig2 = px.scatter(data, x="date", y="no2",trendline="lowess", trendline_color_override="red",title="NO2 levels")
fig2.update_yaxes(title_text="Levels [µg/m3]")
fig2.update_xaxes(title_text="Date")
fig3 = px.scatter(data, x="date", y="o3",trendline="lowess", trendline_color_override="red",title="Ozone levels")
fig3.update_yaxes(title_text="Levels [µg/m3]")
fig3.update_xaxes(title_text="Date")

fig.show()
fig2.show()
fig3.show()

# %% [markdown]
# ## Correlation Scatter Matrix 

# %%
data['o3'] = (data['o3']-data['o3'].min())/(data['o3'].max()-data['o3'].min())
data['pm10'] = (data['pm10']-data['pm10'].min())/(data['pm10'].max()-data['pm10'].min())
data['no2'] = (data['no2']-data['no2'].min())/(data['no2'].max()-data['no2'].min())

fig = px.scatter_matrix(data, dimensions=["pm10", "o3", "no2"])
fig.show()


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'correlation_scatter_matrix', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/16/') #change to your url

# %% [markdown]
# ## Imission temporal regions for Munich
# %% [markdown]
# Based on: https://www.lfu.bayern.de/luft/immissionsmessungen/dokumentation/index.htm

# %%
data = pd.read_excel('munich-air-quality_v2.xlsx')
data.head(10)


# %%
data_pm = data.loc[:,['Label2','PM10','Latitude','Longitude','Month']]
data_pm = data_pm.iloc[1:]
data_pm.head(10)

data_no = data.loc[:,['Label2','NO2 ','Latitude','Longitude','Month']]
data_no = data_no.iloc[1:]
data_no.head(10)

# %%
data_pm = data_pm.fillna(0)
data_no = data_no.fillna(0)


# %%
data_pm = data_pm.convert_dtypes()
data_pm['PM10'] = data_pm['PM10'].apply(pd.to_numeric)

data_no = data_no.convert_dtypes()
data_no['NO2 '] = data_no['NO2 '].apply(pd.to_numeric)

# %%
size = 6*np.ones(36)

# %%
import plotly.express as px

fig = px.scatter_mapbox(data_pm,lat='Latitude',lon='Longitude', size=size, hover_name='Label2',animation_group='Label2',animation_frame='Month',mapbox_style="carto-positron",title='PM10 concentration [µg/m^3] for the year 2018 in Munich',color='PM10',color_continuous_scale="Jet",range_color=(0,100))
fig.show()


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'pm10_geomap', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/35/') #change to your url

#%% 

import plotly.express as px

fig = px.scatter_mapbox(data_no,lat='Latitude',lon='Longitude', size=size, hover_name='Label2',animation_group='Label2',animation_frame='Month',mapbox_style="carto-positron",title='NO2 concentration [µg/m^3] for the year 2018 in Munich',color='NO2 ',color_continuous_scale="Jet",range_color=(0,100))
fig.show()

# %%

import chart_studio.plotly as py
py.plot(fig, filename = 'no2_geomap', auto_open=True)
#%%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/43/') #change to your url
# %% [markdown]
# ## Vehicle Counts in Munich

# %%
vehicles = pd.read_excel('Vehicle_Count.xlsx')
vehicles.drop(0)
vehicles = vehicles.rename(columns={"Unnamed: 0":"Types","Unnamed: 1":"2017","Unnamed: 2":"2018","Unnamed: 3":"2019"})
vehicles = vehicles[1:]


# %%
vehicles.head(10)
# %%
vehicles= vehicles.T
# %%
import plotly
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

fig = make_subplots(rows=1, cols=2)

x=['2017', '2018', '2019']
colors = plotly.colors.DEFAULT_PLOTLY_COLORS
d = {'x':['2017', '2018', '2019'],'y':list(vehicles.T.iloc[5])[1:]}
df = pd.DataFrame(data=d)
help_fig = px.scatter(df, x='x', y='y', trendline="ols")
x_trend1 = help_fig["data"][1]['x']
y_trend2 = help_fig["data"][1]['y']

fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[1])[1:], name='Gasoline',marker_color=colors[0]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[2])[1:], name='Diesel',marker_color=colors[1]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[3])[1:], name='Liquid Gas',marker_color=colors[2]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[4])[1:], name='Natural Gas',marker_color=colors[3]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[5])[1:], name='Electro',marker_color=colors[4]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[6])[1:], name='Hybrid (no Plug-In)',marker_color=colors[5]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[7])[1:], name='Plug-In Hybrid',marker_color=colors[6]),row=1, col=1)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[8])[1:], name='Other',marker_color=colors[7]),row=1, col=1)


fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[5])[1:], name='Electro',marker_color=colors[4],showlegend=False),row=1, col=2)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[6])[1:], name='Hybrid (no Plug-In)',marker_color=colors[5],showlegend=False),row=1, col=2)
fig.add_trace(go.Bar(x=x, y=list(vehicles.T.iloc[7])[1:], name='Plug-In Hybrid',marker_color=colors[6],showlegend=False),row=1, col=2)
fig.add_trace(go.Line(x=x_trend, y=y_trend,name='Trend line for pure electric cars'),row=1,col=2)


fig.update_layout(barmode='stack')
#fig.update_xaxes(categoryorder='total ascending')
fig.show()


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'vehicle_counts', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/21/') #change to your url


