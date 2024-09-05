# Use a base image
FROM ubuntu:20.04

# Set the working directory
WORKDIR /mnt/sdc/trace-collector/VisQUIC

# mount the current directory to the container as a volume



RUN apt update
RUN apt install -y python3
RUN apt install -yq tshark
RUN apt install -y wget
RUN wget https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
RUN dpkg -i google-chrome-stable_current_amd64.deb; exit 0
RUN apt -fy install
RUN alias chrome="google-chrome --no-sandbox"

# RUN cd trace-handler

# Install any dependencies
CMD cd trace-handler && ./create_trace.sh
