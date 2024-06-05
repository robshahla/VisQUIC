#!/bin/bash
# python3 pcap_to_csv.py `find high_class_data/ -type f -name "*.pcap" -printf "%p "`

# for FOLDER in `find v3_data/ -mindepth 1 -maxdepth 1 -type d`
# do
#     echo $FOLDER
#     for SUBFOLDER in `find $FOLDER -mindepth 1 -maxdepth 1 -type d`
#     do
#         echo $SUBFOLDER
#         python3 pcap_to_csv.py `find $SUBFOLDER -type f -name "*.pcap" -printf "%p "`
#     done
# done
# 59
for FOLDER in `find /Volumes/ELEMENTS/quic-classification/v4_data65/ -mindepth 1 -maxdepth 1 -type d`
do
    echo $FOLDER
    find $FOLDER -type f -name "*.pcap" | paste -sd " " - | xargs -n5000 -P1 python3 pcap_to_csv.py
    # python3 pcap_to_csv.py `find $FOLDER -type f -name "*.pcap" -printf "%p "`
done