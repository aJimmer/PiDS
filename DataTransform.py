
import subprocess, time, random, os, datetime, itertools
import pandas as pd
import numpy as np

from LogToDataFrame import LogToDataFrame
from DataFrameToMatrix import DataFrameToMatrix

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

def get_sec(time_str):
    try:
        if time_str != 'nan':
            h, m, s = time_str[-18:].split(':')
            return float(h) * 3600 + float(m) * 60 + float(s)
    except ValueError:
        return 0.0
  
# Helper method for scatter/beeswarm plot
def jitter(arr):
    stdev = .02*(max(arr)-min(arr))
    return arr + np.random.randn(len(arr)) * stdev

# Encode a numeric column as zscores
def encode_numeric_zscore(df, name, mean=None, sd=None):
    if mean is None:
        mean = df[name].mean()

    if sd is None:
        sd = df[name].std()

    df[name] = (df[name] - mean) / sd

log_to_df = LogToDataFrame()
conn_log_df = log_to_df.create_dataframe('/home/mendel/conn.log')

conn_log_df['key'] = list(zip(conn_log_df["id.orig_h"], conn_log_df["id.orig_p"], conn_log_df["id.resp_h"], conn_log_df["id.resp_p"]))
conn_log_df['orig_bytes'] = conn_log_df['orig_bytes'].fillna(0)
conn_log_df['resp_bytes'] = conn_log_df['resp_bytes'].fillna(0)
conn_log_df['resp_pkts'] = conn_log_df['resp_pkts'].fillna(0)
conn_log_df['orig_ip_bytes'] = conn_log_df['orig_ip_bytes'].fillna(0)
conn_log_df['resp_ip_bytes'] = conn_log_df['resp_ip_bytes'].fillna(0)
conn_log_df['no_of_flows'] = conn_log_df.groupby('key')['key'].transform('count')
conn_log_df['duration'] = conn_log_df['duration'].astype(str).apply(get_sec)
conn_log_df['orig_bytes_sum'] = conn_log_df.groupby('key')['orig_bytes'].transform('sum')
conn_log_df['resp_bytes_sum'] = conn_log_df.groupby('key')['resp_bytes'].transform('sum')
conn_log_df['ratio_of_size'] = conn_log_df['resp_bytes_sum'] / conn_log_df['resp_bytes_sum'] + conn_log_df['orig_bytes_sum']
conn_log_df['ratio_of_size'] = conn_log_df['ratio_of_size'].fillna(0)

conn_log_df = conn_log_df[conn_log_df['proto'] != 'icmp']

numeric_names = ["id.orig_p","id.resp_p","duration","orig_bytes","resp_bytes","missed_bytes","orig_pkts","orig_ip_bytes","resp_pkts","resp_ip_bytes","no_of_flows","orig_bytes_sum","resp_bytes_sum","ratio_of_size"] 
final_df = conn_log_df.copy()

toMatrix = DataFrameToMatrix()
odd_matrix = toMatrix.fit_transform(final_df)
kmeans = KMeans(n_clusters=4).fit_predict(odd_matrix)
pca = PCA(n_components=3).fit_transform(odd_matrix)

# Now we can put our ML results back onto our dataframe!
final_df['x'] = pca[:,0] # PCA X Column
final_df['y'] = pca[:, 1] # PCA Y Column
final_df['cluster'] = kmeans

# Jitter so we can see instances that are projected coincident in 2D
final_df['jx'] = jitter(final_df['x'])
final_df['jy'] = jitter(final_df['y'])

for i in numeric_names:
  encode_numeric_zscore(final_df, i)

final_df['label'] = 1
Y_normal = final_df['label']
X_normal = final_df.drop(["x", "y","proto", "service", "local_orig", "local_resp", "history", "id.orig_h", "id.resp_h","uid", "key", "tunnel_parents", "conn_state", "label"], axis = 1)


