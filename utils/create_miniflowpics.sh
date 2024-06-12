#!/bin/bash
export SAVE_PATH=v5_windows_32_new

find VisQUIC01/ -type f -name "*[[:digit:]].csv" -printf "%p " | xargs -n1 -P100 python3 ../trace-handler/prepare.py --save_path=$SAVE_PATH --files