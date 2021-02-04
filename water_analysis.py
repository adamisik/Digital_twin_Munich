# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Analysis of household water supply in Munich

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
data = pd.read_excel('/Users/adammisik/Documents/01_TUM/02_Master/02_Wahlmodule/05_PropENS/02_Coding/Digital_twin_Munich/data/water_supply_households.xlsx')
data = data.fillna(0)

data  = data.iloc[1:-3]
data.head(10) 


# %%
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import scipy
import time
import plotly.express as px

# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(go.Scatter(x=data['Berichtsjahr'], y=data['Wasserabg. an Haush. u. Kleingew. je Einw. und Tag'],name='Water supply',mode='lines'))
fig.add_trace(
    go.Scatter(x=data['Berichtsjahr'] , y=data['Wassergewinnung'],name='Water extraction',mode='lines'),
    secondary_y=True
)
fig.update_yaxes(title_text="Water extraction in 1000cbm", secondary_y=True)
fig.update_yaxes(title_text="Daily water supply per occupant in l", secondary_y=False)
fig.update_xaxes(title_text="Year")

fig.show()




# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'water_supply_curve', auto_open=True)

# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/62') #change to your url


# %%
