import os
import sys
import subprocess

def main():
    # We assume that all the files are pcaps, and have the extension .pcap
    files = sys.argv[1:]
    parallel_processes = 5
    while len(files) > 0:
        num_of_files = min(parallel_processes, len(files))
        # requests = [f"""tshark -r {file} -R quic -2 -T fields -e frame.number -e frame.time_relative -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -e _ws.col.Info -E header=y -E separator=, -E quote=d -E occurrence=f -o tls.keylog_file:$SSLKEYLOGFILE > {file[:-5]}.csv""" for file in files[:num_of_files]]
        requests = [f"""tshark -r {file} -R quic -2 -T fields -e frame.number -e frame.time_relative -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e ipv6.src - e ipv6.dst -e ip.proto -e _ws.col.Info -E header=y -E separator=, -E quote=d -E occurrence=f -o tls.keylog_file:{file.replace('.pcap', '.key')} > {file[:-5]}.csv""" for file in files[:num_of_files]]
        print(len(files))
        processes = [subprocess.Popen(request, shell=True, executable='/bin/bash') for request in requests]
        for process in processes:
            process.wait()
        files = files[num_of_files:]
        # if file.endswith(".pcap"):
        #     print(file)
        #     request = f"""tshark -r {file} -R quic -2 -T fields -e frame.number -e frame.time_relative -e frame.len -e eth.src -e eth.dst -e ip.src -e ip.dst -e ip.proto -e _ws.col.Info -E header=y -E separator=, -E quote=d -E occurrence=f -o tls.keylog_file:$SSLKEYLOGFILE > {file[:-5]}.csv"""
        #     subprocess.run(request, shell=True, executable='/bin/bash')


if __name__ == '__main__':
    main()