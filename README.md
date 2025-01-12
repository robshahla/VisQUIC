# VisQUIC
By: Robert Shahla and Barak Gahtan.

This repository contains the code used to create the VisQUIC dataset. The dataset is used to train a machine learning model to classify QUIC traffic. The dataset includes traces of QUIC traffic captured from different webpages in `PCAP` format, as well as images that represent the traffic in a visual way. The dataset is created in three major steps:
1) Create the traces by issuing requests to webpages and capturing the traffic.
2) Convert the pcap files to csv files that contain only the QUIC traffic.
3) Prepare the images dataset, i.e. create the `png` files from the csv files.

## Dataset Description
The full dataset is available [here](https://www.dropbox.com/scl/fo/8qg9rnw8r9h5wyv0kihhk/AFC0c0jwM5zZHqc6tym3spA?rlkey=ggqq7w71dk8h8zp4mgbviirb7&st=bho6tmv0&dl=0) [and here](https://www.dropbox.com/scl/fo/8qg9rnw8r9h5wyv0kihhk/AFC0c0jwM5zZHqc6tym3spA?rlkey=q9i0rs2equxdgpchpr5del2ji&st=fq5l0ke8&dl=0).
The dataset contains images and PCAP traces of QUIC traffic, all of which in `zip` files.
The images have the prefix `rgb_images` and the traces have the prefix `VisQUIC`. 

Each `zip` file of the `rgb_images` contains a folder with the following hierarchy:
```
<zip_file_name>
│
└───<server_name>
    │
    └───<website_name>
        │
        └───<trace_number>
            │
            └───<window_size>
                │
                └───<overlap>
                    │
                    └───<label>
                        │
                        └───<image1>
                        │
                        └───<image2>
                        │
                        └───...
```

Each `zip` file of the `VisQUIC` contains a folder with the following hierarchy:
```
<zip_file_name>
│
└───<server_name>
    │
    └───<website_name>
        │
        └───<pcap_file1>
        │
        └───<key_file1>
        │
        └───<pcap_file2>
        │
        └───<key_file2>
        │
        └───...
```

For each `PCAP` file there is a corresponding `key` file that is used to decrypt the traffic. The key file is named the same as the `PCAP` file but with the extension `.key`. The key file is used to decrypt the traffic using the `tshark` command, or with `Wirshark` by specifying the key file in the `TLS` settings.
We note that not all traces can be decrypted, as some of the traces are encrypted with a different key that is not 
provided in the dataset.

The reason for storing the connection keys is as follows:
During a QUIC connection, the client and server use multiple keys for encrypting their packets, such as Handshake keys and Application Data keys [RFC 9001](https://datatracker.ietf.org/doc/html/rfc9001). The keys are used to encrypt a packet’s payload and a large portion of the packet’s header. An observer that observers the traffic, will not be able to read the encrypted parts of the packets, and since we capture the traffic using Tshark from an observer point of view, we need to store the connection keys (SSL keys) from the client side in order to later use them to decrypt the captured traffic by the observer.

## Sample Data

The links to the webpages used to create the dataset are available in the `links-for-request` folder.

A sample of the dataset is provided in this repository under the `dataset-samples` folder. The sample contains a small subset of the full dataset, and is used to demonstrate the structure of the dataset. The sample is taken from the `rgb_images1.zip` and `VisQUIC1.zip` files.
- `VisQUIC1` is the folder containing the images. It contains two subfolders for two web servers, each web server folder contains a subfolder for two webpages, and each webpage folder contains the traces (`PCAP` files) and their keys (`key` files) which are used to decrypt the traffic.
- `v5_windows_32_new` is the folder containing the corresponding images for the traces in `VisQUIC1`. It contains two subfolders for two web servers, each web server folder contains a subfolder for two webpages, and each webpage folder contains a subfolder for each trace. The hierarchy continues as described above.

The webpage links that were used to create the dataset are available in the file `links-for-request/all_websites.txt`. The file contains 9,075 links to different webpages.

## Requirements
The following was tested on Ubuntu 22.04.1 LTS.

1) Python 3.7 or higher.
2) wget:
```bash
apt update && apt upgrade
apt install wget
```

3) Google Chrome. The version used to create this dataset is 119.0.6046.159. To download the current version as an example:
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
git clone git@github.com:robshahla/VisQUIC.git
```

### Creating the dataset

1) Set the following alias for the `chrome` command:
```
alias chrome="google-chrome"
```

2) Create the traces by issuing requests to webpages. The requests are issued
one at a time:
```
python3 create_trace.py --links_folder=<folder_path_containing_links> --output_folder=<path_to_output_folder> --requests_per_webpage=<number_of_requests> --starting_index=<index>
```
The parameters are as follows:
- `--links_folder` is the path to the folder containing the links to the webpages to be used in the experiment. The folder is expected to contain subfolders of servers, and in each subfolder a file named `links.txt` which contains the links for different webpages, each link in a row.
- `--output_folder` is the path to the folder where the traces will be saved. The output folder will contain subfolders for each server, and in each subfolder there will be a folder for each webpage, and in each webpage folder there will be a folder for each request. Each request folder will contain the pcap file, the HTML file, the NetLog file, log, and the key file. The key file is used to decrypt the traffic. 
- `--requests_per_webpage` is the number of requests to be sent to each webpage, sequentially.
- `--starting_index` is a number to add as a prefix to the trace number. This is useful when running the script multiple times to create multiple datasets. The default value is 0.  

Usage example:
```
python3 create_trace.py --links_folder=links-for-request --output_folder=data2 --requests_per_webpage=100 --starting_index=0002
```

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
You can also run the script `create_miniflowpics.sh` which will create the mini-flowpics dataset. Note that the script will run `100` processes in parallel, and it process all the csv files in the directory specified inside the script.
To run the script:
```
./create_miniflowpics.sh
```

## Contribution
This repository is part of a research project. It is managed Robert Shahla and Barak Gahtan. #TODO: Write more here on how to contribute.
