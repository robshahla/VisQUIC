import json
import math
import os
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import zipfile
import tempfile
import pickle

from PIL import Image
from matplotlib import cm
from net_log_event_type_list_114_0_5735_133 import event_type
from urllib.parse import urlparse



MAX_PACKET_LENGTH = 1500


def get_hosts_from_domain(json_file):
    """
    return the hosts names from the json_file name.
    """
    domain = json_file.split(os.path.sep)[-1]
    
    # replace _ with / in the domain name
    domain = domain.replace("_", "/")
    domain = urlparse(domain).netloc
    hosts = domain.split(".")
    hosts = hosts[1:-1] if len(hosts) > 2 else hosts[:-1]
    return hosts


def get_quic_connection_ids(json_file):
    """
    go over all the lines in the json file and return a list of the source.id
    that correspond to events of type QUIC_SESSION
    """
    quic_connection_ids = []
    events_ids = []  # the event_ids of quic sessions that will later be used to find the quic connection ids

    hosts = get_hosts_from_domain(json_file)
    with open(json_file, 'r') as f:
        data = json.load(f)
        for event in data['events']:
            if 'type' in event:
                if event_type[event['type']] == 'QUIC_SESSION' or event_type[event['type']] == 'QUIC_STREAM_FACTORY_JOB_STALE_HOST_NOT_USED_ON_CONNECTION':
                    if 'params' in event:
                        if 'host' in event['params']:
                            for host in hosts:
                                if host in event['params']['host']:
                                    quic_connection_ids.append(event['params']['connection_id'])
                                    print(event['params']['connection_id'])
                                    break

                elif event_type[event['type']] == 'QUIC_SESSION_CERTIFICATE_VERIFIED':
                    if 'params' in event:
                        if 'subjects' in event['params']:
                            for host in hosts:
                                for domain in event['params']['subjects']:
                                    if host in domain:
                                        events_ids.append(event['source']['id'])
                                        break
        if len(quic_connection_ids) == 0:
            if len(events_ids):
                for event in data['events']:
                    if event_type[event['type']] == 'QUIC_SESSION':
                        if 'source' in event:
                            if event['source']['id'] in events_ids:
                                if 'params' in event:
                                    if 'connection_id' in event['params']:
                                        quic_connection_ids.append(event['params']['connection_id'])
                                        print(event['params']['connection_id'])

            
    return list(set(quic_connection_ids))


def clean_pcap_csv(csv_path, json_path, n_streams, client_ip="127.0.0.1", server_ip="127.0.0.2", save=False,
                   save_path=None):
    """
    Assumes server_stream_timestamps is a list of dictionaries.
     Each dict represent a Timestamp object of a specific server stream.
     The Dict structure is:
        {
        "Stream_id":<int>,
        "Accept_time":<float>,
        "Close_time":<float>
        }
    """
    data = pd.read_csv(csv_path)
    # change the column name from _ws.col.info to _ws.col.Info if _ws.col.info exists
    if '_ws.col.info' in data.columns:
        data.rename(columns={'_ws.col.info': '_ws.col.Info'}, inplace=True)
        
    if data.size == 0:
        return None
    
    quic_connection_ids = get_quic_connection_ids(json_path)
    if len(quic_connection_ids) == 0:
        return None
    
    # check if ip.src is an empty column
    ip_column = "ip"
    if data["ip.src"].isnull().all():
        ip_column = "ipv6"

    # client_ip = data["ip.src"][0]
    client_ip = list(data[data["_ws.col.Info"].str.contains(f"DCID={quic_connection_ids[0]}")][f"{ip_column}.src"])[0]
    # server_ips = data.loc[data["ip.src"] != client_ip]
    server_ips_quic = []
    for id in quic_connection_ids:
        server_ips_quic += list(set(data[data["_ws.col.Info"].str.contains(f"DCID={id}")][f"{ip_column}.dst"]))
    
    # remove entries from server_ips_quic that are equal to client_ip
    server_ips_quic = [ip for ip in server_ips_quic if ip != client_ip]

    if len(server_ips_quic) == 0:
        return None
    print(server_ips_quic)
    server_packets = data[data[f'{ip_column}.src'].isin(server_ips_quic)]
    server_header_packets = server_packets[server_packets["_ws.col.Info"].str.contains("HEADERS")]
    server_ip = server_header_packets[f"{ip_column}.src"].value_counts().idxmax() if server_header_packets.size > 0 else None
    if server_ip is None:
        return None

    valid_client = (data[f'{ip_column}.src'] == client_ip) & (data[f'{ip_column}.dst'] == server_ip)
    valid_server = (data[f'{ip_column}.dst'] == client_ip) & (data[f'{ip_column}.src'] == server_ip)
    valid_data = data[valid_client | valid_server]
    server_stream_timestamps = data['frame.time_relative']

    first = valid_data['frame.time_relative'].min()

    def label(row):
        """a header sent means that there's data that is going to be sent, and in our
        case it means that and object is going to be sent."""
        # return the number of of times the word "HEADERS" appeared in row["_ws.col.Info"]
        return row["_ws.col.Info"].count("HEADERS")
            
    def label_packet_direction(row):
        """
        We set the direction for each packet. This function is used to label the packets.
        0 - client to server
        1 - server to client
        """
        if row[f'{ip_column}.src'] == client_ip:
            return 0
        else:
            return 1

    valid_data['object_started'] = valid_data.apply(func=lambda row: label(row=row), axis=1)
    valid_data['Server_src'] = valid_data.apply(func=lambda row: label_packet_direction(row=row), axis=1)
    clean_data = valid_data[['frame.time_relative', 'frame.len', f'{ip_column}.src', f'{ip_column}.dst', 'Server_src', 'object_started']]

    clean_data.rename(columns={'frame.time_relative': 'Time', 'frame.len': 'Length', f'{ip_column}.src': 'Source',
                               f'{ip_column}.dst': 'Destination'}, inplace=True)
    if save:
        clean_data.to_csv(save_path)

    return clean_data


def load_timestamps(timestamps_paths):
    res = []
    for path_t in timestamps_paths:
        with open(path_t, 'r') as f:
            res.append(json.loads(f.read()))
    return res


def clean_dataset(sniff_paths, sniff_n_streams, sniff_timestamps_paths, save_paths, save_data, connections_ips):
    """
    full sniff paths, number of stream foreach full sniff, list of lists (server timestamps for each full sniff),
     save_paths for each clean full sniff, save_data flag, list of tuples (each tuple is <client_ip>, <server_ip>)
    """
    for path_csv, n_streams, timestamps_paths, save_path, ips in zip(sniff_paths, sniff_n_streams,
                                                                     sniff_timestamps_paths, save_paths,
                                                                     connections_ips):
        server_stream_timestamps = load_timestamps(timestamps_paths=timestamps_paths)
        client_ip, server_ip = ips[0], ips[1]
        clean_pcap_csv(path=path_csv,
                       n_streams=n_streams,
                       server_stream_timestamps=server_stream_timestamps,
                       client_ip=client_ip,
                       server_ip=server_ip,
                       save=save_data,
                       save_path=save_path)


def prepare_folders(path):
    path = fr"{os.path.sep}".join(path.split(fr"{os.path.sep}")[:-1])
    if not os.path.exists(path):
        os.makedirs(path)

def save_image(image, path):
    img = Image.fromarray(np.uint8(cm.gist_earth(image) * 255)).transpose(Image.FLIP_TOP_BOTTOM)

    # normalized = (image - image.min()) * 255 / (image.max() - image.min())
    # img = Image.fromarray(np.uint8(normalized), mode='L').transpose(Image.FLIP_TOP_BOTTOM)

    img.save(path)


def window_data_to_multi_hist(data, time_bins, length_bins, window_size, trace = None, current_time = None):
    """
    Given the data of a window, convert it to a histogram.
    The histogram is in RGB scale.
    R - packets from the server to the client (generally the objects data)
    G - packets from the client to the server (generally ACK messages)
    B - the aggregated number of packet from both directions, over the whole trace.
    :param data: the data of the current window
    :param time_bins: number of time bins in the histogram (width of the image)
    :param length_bins: number of length bins in the histogram (height of the image)
    :param window_size: size of the window in seconds
    :param trace: the whole trace, used for the aggregated number of packets
    :param current_time: the current time of the window, used for the aggregated number of packets
    """
    hist = np.zeros((length_bins, time_bins, 3))

    dt_step = window_size / time_bins
    dl_step = MAX_PACKET_LENGTH / length_bins
    for x, dt in enumerate(np.arange(start=0, stop=window_size, step=dt_step)):
        for y, dl in enumerate(np.arange(start=0, stop=MAX_PACKET_LENGTH, step=dl_step)):
            relevant_time = (data['Time'] >= dt) & (data['Time'] < dt + dt_step)
            server_relevant_length = (data['Length'] > dl) & (data['Length'] <= dl + dl_step) & (
                    data['Server_src'] == 1)
            client_relevant_length = (data['Length'] > dl) & (data['Length'] <= dl + dl_step) & (
                    data['Server_src'] == 0)
            
            aggregated_relevant_time = (trace['Time'] < current_time + dt + dt_step)
            aggregated_relevant_length = (trace['Length'] > dl) & (trace['Length'] <= dl + dl_step)

            hist[y][x][0] = np.sum(
                relevant_time & server_relevant_length)  # red for packets from the server to the client (generally the video data)
            hist[y][x][1] = np.sum(
                relevant_time & client_relevant_length)  # green for packets from the client to the server (generally ACK messages)

    return hist


def multi_hist_to_rgb_image(image, path):
    """
    convert the histogram to an RGB image.
    """
    if image[:, :, 0].max() != image[:, :, 0].min():
        image[:, :, 0] = (image[:, :, 0] - image[:, :, 0].min()) * 255 / (image[:, :, 0].max() - image[:, :, 0].min())
    if image[:, :, 1].max() != image[:, :, 1].min():
        image[:, :, 1] = (image[:, :, 1] - image[:, :, 1].min()) * 255 / (image[:, :, 1].max() - image[:, :, 1].min())
    if image[:, :, 2].max() != image[:, :, 2].min():
        image[:, :, 2] = (image[:, :, 2] - image[:, :, 2].min()) * 255 / (image[:, :, 2].max() - image[:, :, 2].min())
    img = Image.fromarray(np.uint8(image), mode="RGB").transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)


def multi_hist_to_rg_image(image, path):
    """
    convert the histogram to an RG image, without the G channel.
    """
    if image[:, :, 0].max() != image[:, :, 0].min():
        image[:, :, 0] = (image[:, :, 0] - image[:, :, 0].min()) * 255 / (image[:, :, 0].max() - image[:, :, 0].min())
    if image[:, :, 1].max() != image[:, :, 1].min():
        image[:, :, 1] = (image[:, :, 1] - image[:, :, 1].min()) * 255 / (image[:, :, 1].max() - image[:, :, 1].min())
    
    image[:, :, 2] = 0
    img = Image.fromarray(np.uint8(image), mode="RGB").transpose(Image.FLIP_TOP_BOTTOM)
    img.save(path)


def window_data_to_hist(data, time_bins, length_bins, window_size, trace = None, current_time = None):
    """
    Given the data of a window, convert it to a histogram.
    The histogram is in gray scale,
    where the value of each pixel is the number of packets in the corresponding bin
    (the relevant time and packet length bin).
    """
    hist = np.zeros((length_bins, time_bins))

    dt_step = window_size / time_bins
    dl_step = MAX_PACKET_LENGTH / length_bins
    for x, dt in enumerate(np.arange(start=0, stop=window_size, step=dt_step)):
        for y, dl in enumerate(np.arange(start=0, stop=MAX_PACKET_LENGTH, step=dl_step)):
            relevant_time = (data['Time'] >= dt) & (data['Time'] < dt + dt_step)
            relevant_length = (data['Length'] > dl) & (data['Length'] <= dl + dl_step)
            count = np.sum(relevant_time & relevant_length)
            hist[y][x] = count

    return hist

def print_hist(hist):
    for x in range(hist.shape[0]):
        for y in range(hist.shape[1]):
            print(hist[x][y][2], end=" ")
        print()


def best_bandwidth_for_window(base_save_path, window_size, overlap, windows_indexes, section, time_bins,
                              length_bins, to_hist=window_data_to_hist, save=save_image, bandwidths=np.linspace(0.01, 0.3, 50), cv=5):
    """
    create windows from section and save them as images in base_save_path
    :param base_save_path: base path to save the windows (flowpics)
    :param window_size: size of the window in seconds
    :param overlap: overlap between windows in percentange
    :param windows_indexes: dict of indexes for each label, assigns an index to each window, used for naming the files
    :param section: section to create windows from. This is a pandas DataFrame containing packet data with timestamps and other relevant information
    :param label: label of the sectionm the label will be used for all windows created from this section
    :param time_bins: number of time bins in the histogram (width of the image)
    :param length_bins: number of length bins in the histogram (height of the image)
    :param to_hist: function that converts the window data to a histogram
    :param save: function that saves the image
    """
    info = section['Time'].agg(['min', 'max'])
    step_size = (1 - overlap) * window_size
    start = info['min']
    stop = info['max']

    bws = []
    if stop - start >= window_size:
        for dt in np.arange(start=start, stop=stop, step=step_size):
            relevant_indexes = (section['Time'] >= dt) & (section['Time'] < dt + window_size)
            window_data = pd.DataFrame(section[relevant_indexes])
            window_data['Time'] -= dt
            data = list(zip(window_data['Time'], window_data['Length']))

            for bw in [0.01]:
                print("best bandwidth:", bw)
                save_path = f'./temp/kde-{bw}-{window_size}-{windows_indexes[1]}.png'
                window_to_kde(data, bw, save_path)
            windows_indexes[1] += 1
            
    return bws


def window_to_kde(data, bw, save_path):

    # Fit the KDE
    kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)

    # Define a fine 2D grid over time and size
    x_grid = np.linspace(0, 1, 500)
    y_grid = np.linspace(0, 1500, 500)
    Xgrid, Ygrid = np.meshgrid(x_grid, y_grid)
    XY = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
    Z = np.exp(kde.score_samples(XY)).reshape(Xgrid.shape)

    # Define the timing window bins. For example, bins of size 0.1.
    bins = np.arange(0, 1.1, 0.0001)
    bin_labels = 0.5 * (bins[1:] + bins[:-1])

    # Aggregate the KDE within each timing window
    aggregated = [np.sum(Z[(Xgrid >= bins[i]) & (Xgrid < bins[i + 1])]) for i in range(len(bins) - 1)]
    # Plot
    plt.figure(figsize=(10, 10))
    plt.bar(bin_labels, aggregated, width=0.01)  # 0.09 width for some gap between bars
    plt.tight_layout()
    plt.savefig(save_path, dpi = 150)

    # # Fit the KDE
    # kde = KernelDensity(kernel='gaussian', bandwidth=bw).fit(data)

    # # Define a fine 2D grid over time and size
    # x_grid = np.linspace(0, 1, 500)
    # y_grid = np.linspace(0, 1500, 500)
    # Xgrid, Ygrid = np.meshgrid(x_grid, y_grid)
    # XY = np.vstack([Xgrid.ravel(), Ygrid.ravel()]).T
    # Z = np.exp(kde.score_samples(XY)).reshape(Xgrid.shape)

    # # Define the timing window bins. For example, bins of size 0.1.
    # bins = np.arange(0, 1.1, 0.0001)
    # bin_labels = 0.5 * (bins[1:] + bins[:-1])

    # # Aggregate the KDE within each timing window
    # aggregated = [np.sum(Z[(Xgrid >= bins[i]) & (Xgrid < bins[i + 1])]) for i in range(len(bins) - 1)]

    # # Plot
    # plt.figure(figsize=(10, 10))  # Set the figure size to 1x1 inch
    # plt.bar(bin_labels, aggregated, width=0.09)  
    # plt.xlabel("Time")
    # plt.ylabel("Aggregated KDE")
    # plt.gca().set_aspect('equal', adjustable='box')  # Maintain aspect ratio
    # plt.tight_layout()
    # plt.savefig(save_path, dpi=150)  # Save at 32 dpi to get an image of size 32x32 pixels


def window_data_to_list(data, time_bins, length_bins, window_size, trace = None, current_time = None):
    """
    Given the data of a window, convert it to a list of lists of the shape:
    [time_bins,length_bins]
    We need two lists, one for the client sent packets and one for the server sent packets.
    :param data: the data of the current window
    :param time_bins: number of time bins in the histogram (width of the image)
    :param length_bins: number of length bins in the histogram (height of the image)
    :param window_size: size of the window in seconds
    :param trace: the whole trace, used for the aggregated number of packets
    :param current_time: the current time of the window, used for the aggregated number of packets
    """
    hist = np.zeros((length_bins, time_bins, 3))
    client_to_server = np.zeros((time_bins, length_bins))
    server_to_client = np.zeros((time_bins, length_bins))

    dt_step = window_size / time_bins
    dl_step = MAX_PACKET_LENGTH / length_bins
    for x, dt in enumerate(np.arange(start=0, stop=window_size, step=dt_step)):
        relevant_time = (data['Time'] >= dt) & (data['Time'] < dt + dt_step)

        # for each packet in the relevant time, add its length to the relevant length bin
        for packet in data[relevant_time].itertuples():
            y = int(packet.Length // dl_step)
            if packet.Server_src == 1:
                server_to_client[x][y] += 1
            else:
                client_to_server[x][y] += 1

    return client_to_server, server_to_client


def save_pickle(pickle_data, path):
    with open(path, 'wb') as f:
        pickle.dump(pickle_data, f)


def section_to_windows_pickles(base_save_path, window_size, overlap, windows_indexes, section, time_bins,
                                length_bins, to_hist=window_data_to_list, save=save_pickle):
    """
    create pickles from section and save them as pkl in base_save_path
    :param base_save_path: base path to save the windows (flowpics)
    :param window_size: size of the window in seconds
    :param overlap: overlap between windows in percentange
    :param windows_indexes: dict of indexes for each label, assigns an index to each window, used for naming the files
    :param section: section to create windows from. This is a pandas DataFrame containing packet data with timestamps and other relevant information
    :param label: label of the sectionm the label will be used for all windows created from this section
    :param time_bins: number of time bins in the histogram (width of the image)
    :param length_bins: number of length bins in the histogram (height of the image)
    :param to_hist: function that converts the window data to a histogram
    :param save: function that saves the image
    """
    info = section['Time'].agg(['min', 'max'])
    step_size = (1 - overlap) * window_size
    start = info['min']
    stop = info['max']

    data_points = []
    if stop - start >= window_size:
        for dt in np.arange(start=start, stop=stop, step=step_size):
            relevant_indexes = (section['Time'] >= dt) & (section['Time'] < dt + window_size)
            window_data = pd.DataFrame(section[relevant_indexes])
            window_data['Time'] -= dt
            client_to_server, server_to_client = to_hist(data=window_data,
                            time_bins=time_bins,
                            length_bins=length_bins,
                            window_size=window_size,
                            trace=section,
                            current_time=dt+window_size)
            
            # the label is the number of headers in the window, sent from the server to the client
            # which is the sum of the values in the row object_started, where Server_src is 1
            label = np.sum(window_data[window_data['Server_src'] == 1]['object_started'])
            client_to_server_data = (client_to_server, label)
            server_to_client_data = (server_to_client, label)
            
            data_points.append((client_to_server_data, server_to_client_data, label))

            print("progress:", dt, "/", stop)
        # normalize the data by the min and max of the data
        normalize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x)) if np.max(x) != np.min(x) else x

        for client_to_server_data, server_to_client_data, label in data_points:
            print("saving files")
            client_to_server_data = (normalize(client_to_server_data[0]), label)
            server_to_client_data = (normalize(server_to_client_data[0]), label)
            save_path = fr"{base_save_path}{os.path.sep}{window_size}{os.path.sep}{overlap}{os.path.sep}{label}{os.path.sep}{windows_indexes[1]}.pkl"
            print(save_path)
            windows_indexes[1] += 1
            prepare_folders(path=save_path)
            save(pickle_data=[client_to_server_data, server_to_client_data], path=save_path)
            print("save window", save_path)


def section_to_windows_images(base_save_path, window_size, overlap, windows_indexes, section, time_bins,
                              length_bins, to_hist=window_data_to_hist, save=save_image):
    """
    create windows from section and save them as images in base_save_path
    :param base_save_path: base path to save the windows (flowpics)
    :param window_size: size of the window in seconds
    :param overlap: overlap between windows in percentange
    :param windows_indexes: dict of indexes for each label, assigns an index to each window, used for naming the files
    :param section: section to create windows from. This is a pandas DataFrame containing packet data with timestamps and other relevant information
    :param label: label of the sectionm the label will be used for all windows created from this section
    :param time_bins: number of time bins in the histogram (width of the image)
    :param length_bins: number of length bins in the histogram (height of the image)
    :param to_hist: function that converts the window data to a histogram
    :param save: function that saves the image
    """
    info = section['Time'].agg(['min', 'max'])
    step_size = (1 - overlap) * window_size
    start = info['min']
    stop = info['max']

    if stop - start >= window_size:
        for dt in np.arange(start=start, stop=stop, step=step_size):
            relevant_indexes = (section['Time'] >= dt) & (section['Time'] < dt + window_size)
            window_data = pd.DataFrame(section[relevant_indexes])
            window_data['Time'] -= dt
            image = to_hist(data=window_data,
                            time_bins=time_bins,
                            length_bins=length_bins,
                            window_size=window_size,
                            trace=section,
                            current_time=dt+window_size)
            
            # the label is the number of headers in the window, sent from the server to the client
            # which is the sum of the values in the row object_started, where Server_src is 1
            label = np.sum(window_data[window_data['Server_src'] == 1]['object_started'])
            
            # label = np.sum((window_data['object_started'] == 1) & (window_data['Server_src'] == 1))
            # print_hist(image)

            save_path = fr"{base_save_path}{os.path.sep}{window_size}{os.path.sep}{overlap}{os.path.sep}{label}{os.path.sep}{windows_indexes[1]}.png"
            print(save_path)
            windows_indexes[1] += 1
            prepare_folders(path=save_path)
            save(image=image, path=save_path)
            print("save window", save_path)

def change_root_dir(path, new_root):
    return os.path.sep.join([new_root] + path.split(os.path.sep)[1:])

def check_header_type(csv_path, json_path, n_streams, client_ip="127.0.0.1", server_ip="127.0.0.2", save=False,
                   save_path=None):
    """
    Assumes server_stream_timestamps is a list of dictionaries.
     Each dict represent a Timestamp object of a specific server stream.
     The Dict structure is:
        {
        "Stream_id":<int>,
        "Accept_time":<float>,
        "Close_time":<float>
        }
    """
    data = pd.read_csv(csv_path)
    # change the column name from _ws.col.info to _ws.col.Info if _ws.col.info exists
    if '_ws.col.info' in data.columns:
        data.rename(columns={'_ws.col.info': '_ws.col.Info'}, inplace=True)
        
    if data.size == 0:
        return None
    
    try:
        quic_connection_ids = get_quic_connection_ids(json_path)
    except:
        return None

    if len(quic_connection_ids) == 0:
        return None
    # client_ip = data["ip.src"][0]
    client_ip = list(data[data["_ws.col.Info"].str.contains(f"DCID={quic_connection_ids[0]}")]["ip.src"])[0]
    # server_ips = data.loc[data["ip.src"] != client_ip]
    server_ips_quic = []
    for id in quic_connection_ids:
        server_ips_quic += list(set(data[data["_ws.col.Info"].str.contains(f"DCID={id}")]["ip.dst"]))
    
    # remove entries from server_ips_quic that are equal to client_ip
    server_ips_quic = [ip for ip in server_ips_quic if ip != client_ip]

    if len(server_ips_quic) == 0:
        return None
    # print(server_ips_quic)
    server_packets = data[data['ip.src'].isin(server_ips_quic)]
    server_header_packets = server_packets[server_packets["_ws.col.Info"].str.contains("HEADERS")]
    server_ip = server_header_packets["ip.src"].value_counts().idxmax() if server_header_packets.size > 0 else None
    if server_ip is None:
        return None

    valid_client = (data['ip.src'] == client_ip) & (data['ip.dst'] == server_ip)
    valid_server = (data['ip.dst'] == client_ip) & (data['ip.src'] == server_ip)
    valid_data = data[valid_client | valid_server]
    server_stream_timestamps = data['frame.time_relative']

    first = valid_data['frame.time_relative'].min()

    def label(row):
        """a header sent means that there's data that is going to be sent, and in our
        case it means that and object is going to be sent."""
        # return the number of of times the word "HEADERS" appeared in row["_ws.col.Info"]
        return row["_ws.col.Info"].count("HEADERS")
        
        # return 1 if "HEADERS" in row["_ws.col.Info"] else 0
        # return sum([timestamp['Accept_time'] <= row['Time'] - first <= timestamp['Close_time'] for timestamp in
        #             server_stream_timestamps])
    
    def label_packet_direction(row):
        """
        We set the direction for each packet. This function is used to label the packets.
        0 - client to server
        1 - server to client
        """
        if row['ip.src'] == client_ip:
            return 0
        else:
            return 1

    # check if there is a row in valid_server that contains the substring HEADERS in row["_ws.col.Info"], 
    # though it is not equal to "HEADERS: 200 OK". There might be multiple headers in the same row

    # make sure that the number of HEADERS and 200 OK are equal in each row
    valid_data = data[valid_server]
    num_headers = valid_data["_ws.col.Info"].str.count("HEADERS")
    num_200_ok = valid_data["_ws.col.Info"].str.count("200 OK")
    if num_headers.sum() != num_200_ok.sum():
        print("number of headers and 200 OK are not equal")
        # print csv_path
        print(csv_path)
        return False
    
    print("number of headers and 200 OK are equal")
    return True


def check_header_type_for_all_files(args):
    zip_folder = args.zip_folder
    zip_folder = "./v5_data1"
    # for each zip file in the folder, find all the csv files in the subtree in the zip
    not_equal_files = []
    for zip_file in os.listdir(zip_folder):
        if zip_file != "mercedes-benz.com.zip":
            continue
        print("working on zip file:", zip_file)
        if zip_file.endswith(".zip"):
            zip_ref = zipfile.ZipFile(fr"{zip_folder}{os.path.sep}{zip_file}", 'r')
            with tempfile.TemporaryDirectory() as temp_dir:
                zip_ref.extractall(temp_dir)
                for root, dirs, files in os.walk(temp_dir):
                    for file in files:
                        if file.endswith(".csv") and "clean.csv" not in file:
                            print("working on file:", file)
                            csv_file = fr"{root}{os.path.sep}{file}"
                            json_file = csv_file.replace(".csv", ".json")
                            base_save_path = fr"{csv_file[:-4]}_colored_windows"
                            # base_save_path = change_root_dir(base_save_path, args.save_path)
                            clean_csv_save_path = f"{csv_file[:-4]}_clean.csv"

                            print("working on file:", csv_file)
                            is_equal = check_header_type(csv_path=csv_file,
                                              json_path=json_file,
                                              n_streams=1,
                                              save=True,
                                              save_path=clean_csv_save_path)
                            if is_equal is not None and not is_equal:
                                print("number of headers and 200 OK are not equal")
                                print(csv_file)
                                not_equal_files.append(csv_file)
                            else:
                                print("number of headers and 200 OK are equal")
    
    print("number of not equal files:", len(not_equal_files))
    print("not equal files:")
    for file in not_equal_files:
        print(file)


def create_pickles(args):
    files = args.files
    for csv_file in files:
        json_file = csv_file.replace(".csv", ".json")
        base_save_path = fr"{csv_file[:-4]}_colored_windows2"
        base_save_path = change_root_dir(base_save_path, args.save_path)
        clean_csv_save_path = f"{csv_file[:-4]}_clean.csv"

        print("working on file:", csv_file)
        data = clean_pcap_csv(csv_path=csv_file,
                              json_path=json_file,
                              n_streams=1,
                              save=True,
                              save_path=clean_csv_save_path)
        
        if data is None:
            print("file is empty")
            continue

        for window_size in [0.3]:
            for overlap in [0.9]:
                section_to_windows_pickles(base_save_path=base_save_path,
                                            window_size=window_size,
                                            overlap=overlap,
                                            windows_indexes={1: 0},
                                            section=data,
                                            time_bins=32,
                                            length_bins=1500,
                                            to_hist=window_data_to_list,
                                            save=save_pickle)


def create_histogram(clean_csv_file):
    """
    Given a clean csv file, plot a histogram which contains the number of bytes trasmitted in each time_bin.
    Each time_bin is 1 millisecond.
    """

    with open(clean_csv_file) as f:
        # read the csv file 
        data = pd.read_csv(clean_csv_file)
        
        data['Length'] = data['Length'].astype(int)
        data['Time'] = data['Time'].astype(float)
        data['Time'] = data['Time'] * 1000
        data['Time'] = data['Time'].astype(int)
        data['Time'] = data['Time'] - data['Time'].min()

        # take only packets sent from the server, with Server_src = 1
        data_server = data[data['Server_src'] == 1]
        data_client = data[data['Server_src'] == 0]
        

        # create a histogram which for each Time value, contains the number of packets transmitted
        # in that Time
        hist, bins = np.histogram(data_server['Time'], bins=list(range(data_server['Time'].max())))
        hist_client, bins_client = np.histogram(data_client['Time'], bins=list(range(data_client['Time'].max())))

        # desplay the histogram
        plt.bar(bins[:-1], hist, width=1)
        plt.bar(bins_client[:-1], hist_client, width=1)
        # set the ylim
        plt.xlim(0, 400)
        plt.show()


def check_nan_inf_values(pkl_file):
    """
    Check if the given pkl file contains values that are nan or inf
    """
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
        for j in range(len(data)):
            for i in range(len(data[j])):
                if np.isnan(data[j][i]).any() or np.isinf(data[j][i]).any():
                    print("shit")
        
        print("all_good")

    return


def main(args):
    files = args.files
    for csv_file in files:
        json_file = csv_file.replace(".csv", ".json")
        base_save_path = fr"{csv_file[:-4]}_colored_windows"
        base_save_path = change_root_dir(base_save_path, args.save_path)
        clean_csv_save_path = f"{csv_file[:-4]}_clean.csv"
        
        # check if the clean csv file already exists
        if os.path.exists(clean_csv_save_path):
            print(f"clean csv file {clean_csv_save_path} already exists")
            continue

        print("working on file:", csv_file)
        data = clean_pcap_csv(csv_path=csv_file,
                              json_path=json_file,
                              n_streams=1,
                              save=True,
                              save_path=clean_csv_save_path)
        
        if data is None:
            print("file is empty")
            continue

        window_sizes = [0.1, 0.3]
        overlaps = [0.9, 0]

        for window_size in window_sizes:
            for overlap in overlaps:
                section_to_windows_images(base_save_path=base_save_path,
                                            window_size=window_size,
                                            overlap=overlap,
                                            windows_indexes={1: 0},
                                            section=data,
                                            time_bins=32,
                                            length_bins=32,
                                            to_hist=window_data_to_multi_hist,
                                            save=multi_hist_to_rg_image)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocessing parameters")
    parser.add_argument("--save_path", help="A path to save the flowpics."
                                        " The images will save in '{res_path}{os.path.sep}{n_streams}'")
    parser.add_argument('--files', nargs='+', default=[])
    parser.add_argument('--zip_folder', default="", help="The folder that contains the zips, each zip contains the traces of a webserver")
    args = parser.parse_args()

    main(args)
