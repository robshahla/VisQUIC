# VisQUIC
By: Robert Shahla and Barak Gahtan

This repository contains the code used to create the VisQUIC dataset. The dataset is used to train a machine learning model to classify QUIC traffic. The dataset includes traces of QUIC traffic captured from different webpages in PCAP format, as well as images that represent the traffic in a visual way. The dataset is created in three major steps:
1) Create the traces by issuing requests to webpages and capturing the traffic.
2) Convert the pcap files to csv files that contain only the QUIC traffic.
3) Prepare the images dataset, i.e. create the `png` files from the csv files.


## Requirements
The following was tested on Ubuntu 22.04.1 LTS.

1) Around 100 - 150 GB of free space.
1) Python 3.7 or higher.
2) wget:
```bash
apt update && apt upgrade
apt install wget
```

3) Google Chrome. The version used to create this dataset is 114.0.5735.198. To download the current version: # TODO: check this
```bash
wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install ./google-chrome-stable_current_amd64.deb

# Make sure that Chrome is installed:
google-chrome --version
```

3) tshark 4.0.6 or higher: Note that if you are a non-super user, then you will need to either allow non-super users to capture packets when downloading the package, or add the user to the set of users that can capture packets.
```bash
apt install tshark
```

4) Python packages:
```
pip3 install bs4
```

## Usage
Clone the repo:
```
git clone git@github.com:robshahla/quic-classification.git  TODO: change this.
```

### Creating the dataset

1) Set the following alias for the `chrome` command:
```
alias chrome="google-chrome"
```

2) Set the `SSLKEYLOGFILE` environment variable to the path of the file where the SSL keys will be saved. This is needed to decrypt the QUIC traffic. Run the following: # TODO: check this, i think it is not needed.
```
export SSLKEYLOGFILE=<ssl_key_file_path>.log
```

3) Create the traces by issuing requests to webpages. The requests are issued
one at a time:
```
python3 create_trace.py --links_folder=<folder_path_containing_links> --output_folder=<path_to_output_folder> --requests_per_webpage=<number_of_requests> --starting_index=<index>
```
Where `--links_folder` is the path to the folder containing the links to the webpages to be used in the experiment. The folder is expected to contain subfolders of servers, and in each subfolder a file named `links.txt` which contains the links for different webpages, each link in a row.
`--output_folder` is the path to the folder where the traces will be saved. The output folder will contain subfolders for each server, and in each subfolder there will be a folder for each webpage, and in each webpage folder there will be a folder for each request. Each request folder will contain the pcap file, the HTML file, and the NetLog file and log. Note that currently only a directory under the current (`.`) directory can be used as the output folder. #TODO: Check or change this.
`--requests_per_webpage` is the number of requests to be sent to each webpage, sequentially.
`--starting_index` is a number to add as a prefix to the trace number. This is useful when running the script multiple times to create multiple datasets. The default value is 0.  
Usage example:
```
python3 create_trace.py --links_folder=links-for-request --output_folder=data2 --requests_per_webpage=100 --starting_index=0002
```

#TODO: maybe add info about the links-for-request folder.

### Preparing the dataset
1) Convert the pcap files to csv files that contain only the QUIC traffic, so that they can be used in the `prepare.py` script. Usage:
```
python3 pcap_to_csv.py <input_pcap_file(s)>
```
The files will be saved in the same directory as the input file(s) with the same name but with the extension `.csv`. You can also run the script `pcap_to_csv.sh` which will convert all the pcap files in directory specified inside the script. Note that the script will run single process at a time.  
To run the script:
```
./pcap_to_csv.sh
```

### Creating the images dataset
1) Create the images dataset, i.e. create the `png` files from the csv files generated in step 3. Usage:
```
python3 prepare.py --save_path <path_to_folder> --files <input_csv_file(s)>
``` 
Where `--save_path` is the path to the folder where the `png` files will be saved. The folder will contain subfolders for each trace, and in each trace folder there will be a folder for each window size, and in each window size folder there will be a folder for each overlap, and in each overlap folder there will be a folder for each label, and in each label folder there will the images with that label. Note that currently only a directory under the current (`.`) directory can be used as the save path. `--files` is the path to the csv file(s) generated in step 3. Usage example:
```
python3 prepare.py --save_path ./windows_data --files file1.csv file2.csv
```
You can also run the script `create_miniflowpics.sh` which will create the mini-flowpics dataset. Note that the script will run multiple processes in parallel, and it process all the csv files in the directory specified inside the script.  
To run the script:
```
./create_miniflowpics.sh
```

### Running the model

### Other Useful Scripts



## Contribution
This repository is part of a research project. It is managed Robert Shahla and Barak Gahtan. #TODO: Write more here on how to contribute.