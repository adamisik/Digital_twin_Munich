#In[]
import plotly.graph_objects as go
import urllib, json
import pandas as pd
import chart_studio
import chart_studio.plotly as py

# %%
df_flows = pd.read_excel('data/energy_sankey.xlsx', engine = 'openpyxl',sheet_name = 'flows')
df_labels= pd.read_excel('data/energy_sankey.xlsx', engine = 'openpyxl',sheet_name = 'labels')

df_flows = df_flows.dropna()
df_labels = df_labels.dropna()
df_labels.head()
# %%
opacity = 0.1
fig = go.Figure(data=[go.Sankey(
    valueformat = ".0f",
    valuesuffix = "GJ",
    arrangement = 'snap',
    orientation = 'h',
    # Define nodes
    node = dict(
      pad = 20,
      thickness = 10,
      line = dict(color = "black", width = 1),
      label =  df_labels.node_label,
      color =  df_labels.node_color
    ),
    # Add links
    link = dict(
      source =  df_flows.Source,
      target =  df_flows.Target,
      value =  df_flows.Value,
      color =  df_flows.Color
))])
#fig.update_layout(title_text="Flow Color Coding: Blue-Households, Orange-Industry, Green-Transportation", font_size=8)

fig.show()
# %%
username = 'adam_misik' # your username
api_key = 'vjJDc0jFwLzCOHuwiM4P' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

# %%
py.plot(fig, filename = 'sankey_energy', auto_open=True)
# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/23/') #change to your url
# %%
df_flows.keys()

# %%
