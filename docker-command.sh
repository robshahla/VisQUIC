#!/bin/bash
sudo docker build -t trace-collector .
sudo docker run --cap-add=NET_RAW --cap-add=NET_ADMIN -v /mnt/sdc/trace-collector/VisQUIC:/mnt/sdc/trace-collector/VisQUIC -it trace-collector --name collector
