# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Household Waste analysis

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
data = pd.read_excel('Households_Waste.xlsx')
data.head(10)


# %%
import matplotlib, random

hex_colors_dic = {}
rgb_colors_dic = {}
hex_colors_only = []
for name, hex in matplotlib.colors.cnames.items():
    hex_colors_only.append(hex)
    hex_colors_dic[name] = hex
    rgb_colors_dic[name] = matplotlib.colors.to_rgb(hex)

print(hex_colors_only)

# getting random color from list of hex colors

print(random.choice(hex_colors_only))

# %%
import plotly.graph_objects as go

x=['2017','2018']
fig = go.Figure(go.Bar(x=x, y=list(data.iloc[0][1:]),name='Synthetic material for material use',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[2][1:]),name='Others',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[3][1:]),name='Clothes for recycling',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[4][1:]),name='Glass for recycling',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[5][1:]),name='Wood for recycling',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[6][1:]),name='Paper for recycling',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[7][1:]),name='Building Rubble',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[8][1:]),name='Organic waste',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[9][1:]),name='Electronic waste',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[10][1:]),name='Residual waste',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[11][1:]),name='Garden waste',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[12][1:]),name='Light packaging',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[13][1:]),name='Metals',marker_color=random.choice(hex_colors_only)))
fig.add_trace(go.Bar(x=x, y=list(data.iloc[14][1:]),name='Problematic waste incl. asbestos cement and mineral wool ',marker_color=random.choice(hex_colors_only)))


fig.update_layout(barmode="stack")
fig.update_xaxes(categoryorder="total descending")
fig.update_yaxes(title_text='Quantity in t')

# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'household_waste', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/29/') #change to your url


# %%



