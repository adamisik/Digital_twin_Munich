# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# # Building data on Munich

# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns; sns.set()
import csv
from random import randint
import random
import chart_studio
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt


# %%
username = 'adam_misik' # your username
api_key = 'vjJDc0jFwLzCOHuwiM4P' # your api key - go to profile > settings > regenerate key
chart_studio.tools.set_credentials_file(username=username, api_key=api_key)

# %% [markdown]
# ## Data Pre-Processing

# %%
#Read Excel-Sheet
data = pd.read_excel('EnergyPerBuilding_Munich-2015-0_0.xlsx')
data.head(10)


# %%
#Formatting
data.columns = data.iloc[0]
data = data.iloc[1:]
data.head(10)


# %%
#Slice for relevant attributes (here coordinates)
geo_coordinates = data.loc[:, ['lon','lat','total_demand','occupants']]
geo_coordinates = geo_coordinates.rename(columns={"lon":"Latitude","lat":"Longitude","total_demand":"Heat demand","occupants":"Occupants"})
geo_coordinates.head(10)


# %%
#Format to correct data type
cols = geo_coordinates.select_dtypes(exclude=['float']).columns
geo_coordinates[cols] = geo_coordinates[cols].apply(pd.to_numeric, downcast='float', errors='coerce')

# %% [markdown]
# ## K-Means Clustering

# %%
# Calculate the optimal amount of clusters with Elbow method
K_clusters = range(1,10)
kmeans = [KMeans(n_clusters=i) for i in K_clusters]
Y_axis = geo_coordinates[['lat']]
X_axis = geo_coordinates[['lon']]
score = [kmeans[i].fit(Y_axis).score(Y_axis) for i in range(len(kmeans))]
# Visualize
plt.plot(K_clusters, score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()


# %%
#K-Means algorithm implementation
kmeans = KMeans(n_clusters = 5, init ='k-means++')
kmeans.fit(geo_coordinates[geo_coordinates.columns[0:2]]) # Compute k-means clustering.
geo_coordinates['cluster_label'] = kmeans.fit_predict(geo_coordinates[geo_coordinates.columns[0:2]])
centers = kmeans.cluster_centers_ # Coordinates of cluster centers.
labels = kmeans.predict(geo_coordinates[geo_coordinates.columns[0:2]]) # Labels of each point
geo_coordinates.head(10)


# %%
n_clusters = 5
x = 0.05 # percentage
indices = list()

for i in range(n_clusters):

    # (1) indices of all the points from X that belong to cluster i
    C_i = np.where(geo_coordinates['cluster_label'] == i)[0].tolist() 
    n_i = len(C_i) # number of points in cluster i

    # (2) indices of the points from X to be sampled from cluster i
    sample_i = np.random.choice(C_i, int(x * n_i)) 
    indices.extend(sample_i)
    print(i, sample_i)
geo_coordinates_red = geo_coordinates.iloc[indices] 
labels_red = labels[indices]


# %%
# Visualize the clustering results (as the data is too big for a viz, slice it randomly with an index range)
import plotly.express as px

geo_coordinates_red['cluster_label'] = geo_coordinates_red['cluster_label'].astype('str')

fig = px.scatter_mapbox(geo_coordinates_red, lon = 'Longitude', lat = 'Latitude',color='cluster_label',size='Occupants',mapbox_style='carto-positron')
fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.02,
    xanchor="right",
    x=1
))
fig.update_layout(legend_title='Cluster Labels')

fig.show()

# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'clustering_households_densities', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/4/') #change to your url

# %% [markdown]
# ## Heat map of yearly heat demand in Munich

# %%
geo_coordinates_red['Normalized Heat demand'] = (geo_coordinates_red['Heat demand'] -geo_coordinates_red['Heat demand'].min())/(geo_coordinates_red['Heat demand'].max() - geo_coordinates_red['Heat demand'].min())


# %%
fig = px.density_mapbox(geo_coordinates_red, lat='Latitude', lon='Longitude', z='Normalized Heat demand', radius=10,
                        center=dict(lat= 48.137154,  lon=11.576124), zoom=10,
                        mapbox_style="stamen-terrain")
fig.show()                        


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'heat_map_Munich_v2', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/10/') #change to your url

# %% [markdown]
# ## Bubble Map of Buildings by Occupants

# %%
import plotly.express as px
fig = px.scatter_mapbox(geo_coordinates_red, lat="Latitude", lon="Longitude", color="Occupants", size="Heat demand",
                  color_continuous_scale=px.colors.cyclical.IceFire, size_max=15, zoom=10,
                  mapbox_style="carto-positron")
fig.show()


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'bubbles_by_occupants', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/6/') #change to your url

# %% [markdown]
# ## Correlation between heat demand and occupants

# %%
import plotly.graph_objects as go
from plotly.subplots import make_subplots
# Create figure with secondary y-axis
fig = make_subplots(specs=[[{"secondary_y": True}]])

geo_coordinates_red['Yearly heat demand in kW'] = geo_coordinates_red['Heat demand']/1000
fig.add_trace(go.Histogram(x=geo_coordinates_red['Occupants']))
fig.add_trace(
    go.Line(x=geo_coordinates_red['Occupants'] , y=geo_coordinates_red['Yearly heat demand in kW'],mode='markers'),
    secondary_y=True
)
fig.update_yaxes(title_text="Yearly heat demand in kW", secondary_y=True)
fig.update_yaxes(title_text="Count", secondary_y=False)
fig.update_xaxes(title_text="Occupants")
fig.update_layout(showlegend=False)

fig.show()


# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'relation_occupants_heat_demand', auto_open=True)


# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/50/') #change to your url


# %%
# ## Multivariate Regression
regression_data = data.loc[:,['footprint_area','free_walls','floors','storey_area','heated_area','occupants','solar_gains','Tset','total_demand']]
regression_data.head(10)


# %%
jet= plt.get_cmap('jet')
colors = iter(jet(np.linspace(0,1,10)))
 
def correlation(df,variables, n_rows, n_cols):
    fig = plt.figure(figsize=(8,6))
    #fig = plt.figure(figsize=(14,9))
    for i, var in enumerate(variables):
        ax = fig.add_subplot(n_rows,n_cols,i+1)
        asset = df.loc[:,var]
        ax.scatter(df["total_demand"], asset, c = next(colors))
        ax.set_xlabel("Heat demand")
        ax.set_ylabel("{}".format(var))
        #ax.set_title(var +" vs total heat demand")
    fig.tight_layout() 
    plt.savefig('/Users/adammisik/Documents/01_TUM/02_Master/02_Wahlmodule/05_PropENS/02_Coding/Digital_twin_Munich/correlation_plot.png')
    plt.show()
        
# %%
variables = regression_data.columns[:-1]  
correlation(regression_data,variables,4,4)

# %%
# ## The correlation plots show us strong correlations between 
# the building area/solar gains and heat demand, but low correlation between 
# floors/temperature and heat demand.
# %%
regression_data.corr(method='pearson')
# %%
#Check for 0's
regression_data.isnull().sum().loc[variables]
len(regression_data.index)
# %%
import xgboost as xg 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import mean_squared_error as MSE

#Train-Test Split
train_X, test_X, train_y, test_y = train_test_split(regression_data[["footprint_area","free_walls","storey_area","solar_gains","heated_area"]], regression_data['total_demand'], 
                      test_size = 0.3, random_state = 123)
train_X.head(10)    

# Train and test set are converted to DMatrix objects, 
# as it is required by learning API. 
train_dmatrix = xg.DMatrix(data = train_X.to_numpy(), label = train_y.to_numpy()) 
test_dmatrix = xg.DMatrix(data = test_X.to_numpy(), label = test_y.to_numpy()) 
  
# Parameter dictionary specifying base learner 
param = {"booster":"gblinear", "objective":"reg:linear"} 
  
xgb_r = xg.train(params = param, dtrain = train_dmatrix, num_boost_round = 10) 
pred = xgb_r.predict(test_dmatrix) 
  
# RMSE Computation 
rmse = np.sqrt(MSE(test_y, pred)) 
print("RMSE XGBoost : % f" %(rmse)) 
x = np.linspace(0,len(test_y.to_numpy()),len(test_y.to_numpy()))
plt.scatter(x[:50],test_y.to_numpy()[:50])
plt.plot(x[:50],pred[:50],'r--',label='XGBoost Fitted model')
plt.xlabel('Predict')
plt.ylabel('Heat demand in W')
plt.legend()
# %%
# %%
#Structure Linear Regression
lr = LinearRegression()
lr.fit(train_X,train_y) 
# %%
close_predictions = lr.predict(test_X)   
rmse2 = np.sqrt(MSE(test_y, close_predictions)) 
print("RMSE Linear Model: % f" %(rmse2))
# %%
ratio = rmse2/rmse
print("Ratio of RMSE Linear/RMSE XGBoost:",ratio)
# %%
x = np.linspace(0,len(test_y),len(test_y))
df = pd.DataFrame({'Prediction#':x.astype(int),'Actual': test_y, 'Predicted': close_predictions})
df['XGB'] = pred
df1 = df.tail(50)

df1.set_index('Prediction#',inplace=True)
 
error = abs(df1['Actual'] - df1['Predicted'])/df1['Actual']
 
# Plot the error term between the actual and predicted values for the last 10 predictions
 
error.plot(kind='bar',figsize=(8,6))
plt.grid(which='major', linestyle='-', linewidth='0.5', color='green')
plt.grid(which='minor', linestyle=':', linewidth='0.5', color='black')
plt.xticks(rotation=45)
plt.ylabel('Relative error between prediction and actual value')
plt.show()
# %%
plt.scatter(df1.index, df1['Actual'], color='blue',label='Actual value')
plt.plot(df1['Predicted'], 'r--',label='Linear Regression model')
#plt.plot(error*df1['Actual'],color='red',label='Error')
plt.xticks(rotation=45)
plt.xlabel('Prediction#')
plt.ylabel('Household heat demand in W')
plt.legend()

# %%
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=2)

x= np.linspace(0,50,50)
fig.add_trace(go.Scatter(x=x, y=df1['Actual'], marker_color='blue', mode='markers',name='Actual Data', opacity=0.65),row=1,col=1)
fig.add_trace(go.Scatter(x=x, y=df1['Actual'], marker_color='blue',  mode='markers',name='Actual Data', opacity=0.65,showlegend=False),row=1,col=2)

fig.add_trace(go.Scatter(x=x, y=df1['Predicted'],
line=dict(color="red",width=4,dash="dot"),name='Linear Regression Model Fit'),row=1,col=1)
fig.add_trace(go.Scatter(x=x, y=df1['XGB'],
line=dict(color="yellow",width=4,dash="dot"),name='XGBoost Model Fit'),row=1,col=2)
fig.update_yaxes(title_text="Heat demand in W", row=1, col=1)
fig.update_xaxes(title_text="Prediction#", row=1, col=1)
fig.update_xaxes(title_text="Prediction#", row=1, col=2)

fig.show()
# %%
import chart_studio.plotly as py
py.plot(fig, filename = 'household_regression', auto_open=True)
# %%
import chart_studio.tools as tls
tls.get_embed('https://plotly.com/~adam_misik/41/') #change to your url

# %%
