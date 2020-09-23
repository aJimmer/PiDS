import os, time, re, sys, getopt
import pandas as pd
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

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
        if event.is_directory:
            return None

        elif event.event_type == 'created' and ('.log' in event.src_path):
            print("Run ZAT")
            #log_to_df = LogToDataFrame()
            #conn_df = log_to_df.create_dataframe(path + conn)
            #print conn_df
            
        elif event.event_type == 'created' and ('.log' not in event.src_path):
            print('created: ' + event.src_path)
            global counter
            
            if counter == 0:
                counter += 1

            elif counter > 0:
                digit = re.findall(r"\d\d", event.src_path)
                fileToProcess = '/mnt/thesis/captures/nms/' + 'capture-' + str((int(digit[0]) - 1) % 60).zfill(2) + '.pcap'
                os.system('zeek -r ' + fileToProcess)
                os.system('sudo mv /home/pi/PiDS/*.log /mnt/thesis/captures/nms/logs/')

        elif event.event_type == 'deleted':
            print('deleted: ' + event.src_path)
        #elif event.event_type == 'modified':
            #print('modified: ' + event.src_path)
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
