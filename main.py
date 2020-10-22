import os, time, re, sys, getopt
import pandas as pd
from LogToDataFrame import LogToDataFrame
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

import subprocess, time, random, os, datetime, itertools
import numpy as np

from DataFrameToMatrix import DataFrameToMatrix

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics

counter = 0
directory = '/mnt/thesis/captures/nms/'
log = 'conn.log'
output = '/mnt/thesis/captures/nms/data'

class Util:

    def __init__(self, directory):
        self.observer = Observer()
        self.watchDirectory = directory

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watchDirectory, recursive = True)
        self.observer.start()
        print("Now watching for file changes in : " + self.watchDirectory)
        
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()

class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):

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

        if event.is_directory:
            return None

        elif event.event_type == 'created' and ('conn.log' in event.src_path):
            print("Run ZAT: " + event.src_path)
            log_to_df = LogToDataFrame()
            conn_log_df = log_to_df.create_dataframe(event.src_path)
            
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

            X_normal.to_csv('/home/pi/PiDS/data.csv')
            os.system('sudo mv /home/pi/PiDS/data.csv /mnt/thesis/captures/nms/data/')

        elif event.event_type == 'created' and ('.log' not in event.src_path):
            print('created: ' + event.src_path)
            global counter
            
            if counter == 0:
                counter += 1

            elif counter > 0:
                digit = re.findall(r"\d\d", event.src_path)
                fileToProcess = '/mnt/thesis/captures/nms/' + 'capture-' + str((int(digit[0]) - 1) % 60).zfill(2) + '.pcap'
                print('Processing file: ', fileToProcess)
                os.system('zeek -r ' + fileToProcess)
                os.system('sudo rm ' + fileToProcess)
                os.system('sudo mv /home/pi/PiDS/*.log /mnt/thesis/captures/nms/logs/')

        elif event.event_type == 'deleted':
            print('deleted: ' + event.src_path)

        elif event.event_type == 'moved':
            print('moved: ' + event.src_path)

def usage():
    print("Pi Intrusion Detection System\n")
    print("Usage: PiDS.py -d directory -l log")
    print("-d --directory           - Provided path to PCAPs to monitor and generate zeek log")
    print("-l --log                 - Zeek log to generate and monitor i.e. conn.log, dhcp.log, dns.log, files.log, http.log, ntp.log, packet_filter.log, ssl.log, weird.log, x509.log")
    print("-o --output             - Provide an output path for stored machine learning ready dataset")
    print("\nExamples:")
    print("PiDS.py -d /mnt/thesis/captures/nms/logs -l conn.log")
    print("PiDS.py -d /mnt/thesis/captures/nms/logs -l http.log -o /Users/Jimenez/Downloads/")
    sys.exit(0)
    
def main():
    global counter
    global directory
    global log
    global output

    print('Welcome to PiDS')
    
    try:
        opts, args = getopt.getopt(sys.argv[1:],"hd:l:o:", ["help","directory","log","output"]) 
    
    except getopt.GetoptError as err:
        print(str(err))
        usage()
    
    for o, a in opts:
        if o in ("-h","--help"):
            usage()
        elif o in ("-d","--directory"):
            directory = a
        elif o in ("-l","--log"):
            log = a
        elif o in ("-o","--output"):
            output = a
        else:
            assert False,"Unhandled Option"

    watch = Util(directory)
    watch.run()

if __name__ == '__main__':
    main()
