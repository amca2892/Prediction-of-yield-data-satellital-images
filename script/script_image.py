# -*- coding: utf-8 -*-
"""
Created on Mon May  2 15:54:48 2022

@author: amca2
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 10:31:09 2022

@author: amca2
"""


import pandas as pd
import os
import glob
import rasterio as rio
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

path_0 = "D:\Image_Analysis\scripts"

path_yield = os.path.join(path_0,"data")
path_imgs = os.path.join(path_0,"imgs")


os.chdir(path_yield)

yield_filtered = pd.read_csv("yield_filtered.csv")
del yield_filtered["Unnamed: 0"]

# if not os.path.exists(path_imgs) :
#     os.makedirs(path_imgs)

xs = yield_filtered.iloc[0].Longitude
ys = yield_filtered.iloc[0].Latitude

#%%

os.chdir(path_imgs)

src = rio.open("2021-03-08_LAI.TIFF")
LAI = src.read(1)

index_transform = src.transform

# to get geospatial info : src.transform

(row,col) = rio.transform.rowcol(src.transform,xs,ys)

print(row,col)


# yield, latitude, longitude, row, col

#%%

df_all_data = pd.DataFrame(columns = ["wheat_mass","col","row"])

df_all_data["wheat_mass"] = yield_filtered["wheat_mass"]
df_all_data["row"],df_all_data["col"] = rio.transform.rowcol(src.transform,
                                                             yield_filtered["Longitude"],
                                                             yield_filtered["Latitude"])

#%%

# and = &
# or = |
selected_pixel = df_all_data[(df_all_data["row"]==29) & (df_all_data["col"]==29)]

# apply the median to the wheat_mass column
selected_pixel.wheat_mass.median()

# use the groupy by function to get the median for each pixel


df_per_pixel= df_all_data.groupby(["row","col"])["wheat_mass"].median().reset_index()
# df_per_pixel= df_all_data.groupby(["row","col"],as_index=False)["wheat_mass"].median()

#%%

os.chdir(path_imgs)

list_files_LAI = glob.glob("*LAI*")

# 13  (number of dates) * 5 (number of indexes LAI, NDVI, BSI, OSAVI, ARVI) + 3
# 768 rows 

for file in list_files_LAI :
    
    # open my tiff file
    src = rio.open(file)
    LAI = src.read(1)

    column_name = "LAI_" + file[8:10] + "_" + file[5:7]
    print(column_name)
    # input()
    # add a column named LAI_23_03 and add pixel the value from LAI the matrix 
    df_per_pixel[column_name] = LAI[df_per_pixel["row"],df_per_pixel["col"]]


#%%

list_files_12band = glob.glob("*12band*")

for file in list_files_12band :
    
    # open my tiff file
    src = rio.open(file)
    B4 = src.read(4)
    B8 = src.read(8)
    B11 = src.read(11)
    B2 = src.read(2)
    
    
    NDVI = (B8-B4)/(B8+B4)
    BSI = -((B11+B4)-(B8+B2))/((B11+B4)+(B8+B2))
    ARVI = (B8-(2*B4-B2))/(B8+(2*B4-B2))
    OSAVI = (1+0.16)*(B8-B4)/(B8+B4+0.16)
    
    list_indexes_names = ["NDVI","BSI","OSAVI","ARVI"]
    list_indexes = [NDVI,BSI,OSAVI,ARVI]

    for k in range(len(list_indexes_names)) :
        
        column_name = list_indexes_names[k] +  "_" + file[8:10] + "_" + file[5:7]
        df_per_pixel[column_name] = list_indexes[k][df_per_pixel["row"],df_per_pixel["col"]]
    
    
#%% Helper functions

def idx_selector(df):

    list_indexes_names = ["LAI", "NDVI","BSI","OSAVI","ARVI"]
    index_dict = {}
    index_corrs = {}
    best_idx = []
    
    for name in list_indexes_names:
        
        x = [  col for col in df if name in col ] 
        index_dict[name] = df[ x ]
        index_dict[name]["wheat_mass"] = df["wheat_mass"]
        index_corrs[name + " corr"] = index_dict[name].corr()
    
        
    for name in index_corrs:
        
        wheat_corr = index_corrs[name].dropna(axis=0, how="any")
        f, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(wheat_corr, mask=np.zeros_like(wheat_corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
                    square=True, ax=ax)
        plt.show()
        wheat_corr = index_corrs[name].reset_index()
    
        wheat_corr = wheat_corr.iloc[-1]
        wheat_corr = wheat_corr.drop(wheat_corr.index[0])
    
        selected_index = wheat_corr.index[(wheat_corr >= 0.3) | (wheat_corr <= -0.3) ].tolist()
        for idx in selected_index:
            best_idx.append(idx)
    best_idx = list(set(best_idx))
    
    df_best_idx = df[best_idx]
    df_best_idx['row'] = df['row']
    df_best_idx['col'] = df['col']
    
    df_best_idx = df_best_idx.dropna(axis=0, how='any')
    
    return df_best_idx

def idx_grapher(df):

    list_indexes_names = ["LAI", "NDVI","BSI","OSAVI","ARVI"]
    index_dict = {}

    
    for name in list_indexes_names:
        
        x = [  col for col in df.columns if name in col ] 
        index_dict[name] = df[ x ]
        
    
    
    for name in index_dict.keys():
        
        for i in range(len(index_dict[name])) :
            x = index_dict[name].columns
            # df_index.iloc[i].plot()
            plt.plot(x,index_dict[name].iloc[i])
        
        plt.title(f"{name} evolution 2021")
        plt.ylabel(name)
        plt.xlabel("weeks")
        plt.xticks(rotation=45)
        plt.show()
    return index_dict

#%% grapher

idx_grapher(df_per_pixel)
best_idx = idx_selector(df_per_pixel)

#%% Make yield clusters

df_kmeans = df_per_pixel.dropna(axis=0, how='any')
df_kmeans_y = df_kmeans[["wheat_mass"]]
df_kmeans_i = df_kmeans.drop(["row", "col", "wheat_mass"], axis=1)
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

scaler_i = MinMaxScaler().fit(df_kmeans_i)
scaler_y = MinMaxScaler().fit(df_kmeans_y)
X_means_i = scaler_i.transform(df_kmeans_i)
X_means_y = scaler_y.transform(df_kmeans_y)

X_means_i = df_kmeans_i.to_numpy()
X_means_y = scaler)_y.transform(df_kmeans_y

sse_i = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X_means_i)
    sse_i.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse_i)
plt.show()
sse_y = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(X_means_y)
    sse_y.append(km.inertia_)

plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse_y)
clusters_i = KMeans(n_clusters=2).fit_predict(X_means_i)
clusters_y = KMeans(n_clusters=2, random_state=1).fit_predict(X_means_y)
plt.show()


df_clustered = df_per_pixel.dropna(axis=0, how="any")
df_clustered["Cluster_index"] = clusters_i
df_clustered["Cluster_yield"] = clusters_y

sns.scatterplot(data=df_clustered, # indication on which dataframe to use
                x = "row",
                y = "col",
                hue = "Cluster_yield")
plt.show()
sns.scatterplot(data=df_clustered, # indication on which dataframe to use
                x = "row",
                y = "col",
                hue = "Cluster_index")
plt.show()
#%%

cluster_1 = df_clustered[df_clustered["Cluster_yield"]==0]
cluster_1_best_idx = idx_selector(cluster_1)
cluster_2 = df_clustered[df_clustered["Cluster_yield"]==1]
cluster_2_best_idx = idx_selector(cluster_2)

cluster_1_i = df_clustered[df_clustered["Cluster_index"]==0]
cluster_1_i_best_idx = idx_selector(cluster_1_i)

cluster_2_i = df_clustered[df_clustered["Cluster_index"]==1]
cluster_2_i_best_idx = idx_selector(cluster_2_i)

#%%

idxes = idx_grapher(df_per_pixel)
for key in idxes.keys():
    idxes[key]["wheat_mass"] = df_per_pixel['wheat_mass']
    idxes[key]["row"] = df_per_pixel['row']
    idxes[key]["col"] = df_per_pixel['col']
    idxes[key] = idxes[key].dropna(axis=0, how='any')

#%% Dataframe Selection

import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.kernel_ridge import KernelRidge
from matplotlib.colors import Normalize
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn import linear_model
from sklearn.model_selection import train_test_split

df_list = [cluster_1_best_idx, cluster_2_best_idx, cluster_1_i_best_idx, cluster_2_i_best_idx, idx_selector(df_per_pixel),idxes["LAI"],
          idxes["NDVI"], idxes["BSI"], idxes["OSAVI"], idxes["ARVI"]]

name_list = ["Cluster_1_yield", "Cluster_2_yield", "Cluster_1_idx", "Cluster_2_idx", "All clusters", "LAI", "NDVI", "BSI", "OSAVI", "ARVI"]

for df, name in zip(df_list, name_list) :
    
    X = df.drop(["wheat_mass", "row", "col"], axis=1)
    y = df['wheat_mass'].to_numpy()




    scaler = StandardScaler().fit(X)
    X = scaler.fit_transform(X)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1)








    kf = KFold(shuffle=True, n_splits=5)
    rf_reg = RandomForestRegressor(max_depth=3, random_state=0)
    svr_rbf_reg = SVR(kernel="rbf", gamma=0.01)
    lm = linear_model.LinearRegression()
    k_ridge = KernelRidge(kernel="rbf")
    nnet = MLPRegressor(random_state=1, max_iter=10000, solver="adam")
    model_list = [rf_reg, svr_rbf_reg, lm, k_ridge]
    model_names = ["random forest", "SVR", "lm", "kernel ridge"]
    
    for model, name_model in zip(model_list, model_names):
        score = cross_val_score(model, X_train, y_train, cv=kf)
        print(f"The score for the dataframe {name} and model {name_model} is {score.mean()}")
#%% Cluster 1 best idx model selection
from sklearn.metrics import r2_score

lm = linear_model.LinearRegression()
rf_reg = RandomForestRegressor(max_depth=2, random_state=0)
svr_regressor = SVR(kernel='rbf', gamma='auto')
lm_ridge = linear_model.Ridge(alpha=.5)
lm_by = linear_model.BayesianRidge()

X = cluster_1_best_idx.drop(["wheat_mass", "row", "col"], axis=1)
y = cluster_1_best_idx['wheat_mass'].to_numpy()

SelectKBest(f_regression, k = 37).fit_transform(X, y)


scaler = StandardScaler().fit(X)
X = scaler.fit_transform(X)
    
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state =1)

score = cross_val_score(lm_by, X_train, y_train, cv=kf, scoring="neg_root_mean_squared_error")

model = lm.fit(X_train, y_train)

print(score.mean())

test_r2 = r2_score(y_test, model.predict(X_test) )

ys = {"true_y": y_test, "pred_y":model.predict(X_test)}

ys_df = pd.DataFrame(ys)

#%% Cluster 1 best idx model selection continuation
