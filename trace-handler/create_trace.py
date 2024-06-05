import codecs
import json
import logging
import subprocess
import os
import sys
import signal
import time
import argparse

from bs4 import BeautifulSoup

logging.basicConfig(stream=sys.stdout, level=0)
logger = logging.getLogger(__name__)


def read_html(filename):
    with codecs.open(filename, 'r') as f:
        html = f.read()

        js_object_names = []

        parse = BeautifulSoup(html, 'lxml')
        script_tags = parse.find_all('script')

        # Extract JavaScript object names
        for script_tag in script_tags:
            # Get the JavaScript code within the <script> tag
            js_code = script_tag.get_text()

            # Split the code by semicolons to separate statements
            statements = js_code.split(';')

            # Extract object names from each statement
            for statement in statements:
                # Remove leading/trailing whitespace
                statement = statement.strip()

                # Check if the statement defines an object
                if statement.startswith('var') or statement.startswith('let') or statement.startswith('const'):
                    # Extract the object name from the statement
                    parts = statement.split('=')
                    if len(parts) >= 2:
                        object_name = parts[0].split()[-1]
                        js_object_names.append(object_name)

        print(js_object_names)

        # for t in parse.find_all('script'):
        #     print(t)

        # print(parse.head)


def read_json(filename):
    fields = [
        'method',
        'url',
        'priority',
        'time',
        'quic',
        'stream'

    ]
    with open(filename, 'r') as f:
        data = json.load(f)
        count = 0
        quic_objects = 0
        for event in data['events']:
            if 'params' in event:
                if 'method' in event['params'] or 'priority' in event['params']:
                    stuff_to_print = [str(key) + ":" + str(event['params'][key]) for field in fields for key, value in event['params'].items() if field in key]
                    if len(stuff_to_print) > 0:
                        print(stuff_to_print)
                    if 'using_quic' in event['params']:
                        if str(event['params']['using_quic']) == "True":
                            quic_objects += 1

                if 'method' in event['params']:
                    count += 1


                # if 'method' in event['params']:
                #     print(event['params']['method'], event['params']['url'], event['time'])
                # if 'priority' in event['params']:
                #     if 'url' in event['params']:
                #         print(event['params']['priority'], event['params']['url'], event['time'])
                #     else:
                #         print(event)
        print("all: ", count)
        print("quic: ", quic_objects)


def create_folder(parent_path, url: str) -> str:
    """
    Create a folder for the request.
    :param request_id: The ID of the request
    :param url: The URL of the request
    :return: The path of the folder
    """
    path = parent_path + f'{os.path.sep}{url}'
    # path = parent_path + f'{url}'
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def run_tshark(interface, output_file):
    """
    Start tshark to capture packets on the given interface.
    """
    command = f'tshark -i {interface} -w {output_file}'
    process = subprocess.Popen(command, stdout=subprocess.PIPE,
                           shell=True, preexec_fn=os.setsid)

    return process


def kill_tshark(process):
    """
    Kill the tshark process.
    :param process: The process to kill
    """
    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
    time.sleep(3)
    subprocess.run('pkill -15 -f tshark', shell=True, executable='/bin/zsh')
    # subprocess.run('pkill -15 -f tshark', shell=True, executable='/bin/bash')


def issue_request(websites, website_path, request_id: int, url: str, starting_index):
    """
    Issue a request to the given url. The json, html, and logs are all
    saved in a folder that is created.
    :param request_id: The ID of the request
    :param url: The URL of the request
    """
    logger.info('Issuing request {} to {}'.format(request_id, url))
    url_folder_name = url.replace('/', '_')
    path = create_folder(website_path, url_folder_name)
    filename = path + f'{os.path.sep}{url_folder_name}-{starting_index}{request_id}'
    json_file = f'{filename}.json'
    html_file = f'{filename}.html'
    log_file = f'{filename}.log'
    pcap_file = f'{filename}.pcap'


    # request = "chrome " \
    # request = "chromium --no-sandbox " \
            #   "--headless " \
    request = f"SSLKEYLOGFILE={filename}.key /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome " \
              "--autoplay-policy=no-user-gesture-required " \
              "--dump-dom " \
              "--disable-gpu " \
              "--enable-logging " \
              "--enable-quic " \
              "--disable-application-cache " \
              "--incognito " \
              "--new-window " \
              "--v=3 " \
              f"--log-net-log={json_file} " \
              f"{url} " \
              f"> /dev/null " \
              f"2> /dev/null"

    # create sslkeylogfile at the ~ directory, and the name of the file is filename.key with 777 permissions
    subprocess.run(f'touch {filename}.key', shell=True, executable='/bin/zsh', )

    # Start tshark
    logger.info("Starting tshark")
    tshark_process = run_tshark('en0', f'{pcap_file}')

    logger.info("sleeping for 1 second")
    time.sleep(5)

    logger.info('Running request: {}'.format(request))
    try:
        print("path of json file: ", json_file)
        p = subprocess.run(request, shell=True, executable='/bin/zsh', timeout=1200)

    except subprocess.TimeoutExpired:
        logger.info("Timeout expired")
    logger.info("finished chrome request")
    logger.info("sleeping for 1 second")
    time.sleep(5)

    logger.info("Killing tshark")
    kill_tshark(tshark_process)


def get_websites(links_folder: str, websites_to_use: list):
    """
    Receives a path to a folder that contains subfolders for servers,
    and each subfolder contains a links.txt file with different links to download.
    :param links_folder: The path to the folder.
    :return: A dictionary that maps server names to a list of links.
    """
    websites = {}
    for root, dirs, files in os.walk(links_folder):
        for dir in dirs:
            if len(websites_to_use) == 0 or dir in websites_to_use:
                links_file = links_folder + f'/{dir}/links.txt'
                with open(links_file, 'r') as f:
                    links = f.readlines()
                    websites[dir] = [link.strip() for link in links]
    return websites


def cut_suffix(url):
    """
    take the prefix of the website url until the `?` character.
    :param url: The url to cut
    """
    if '?' in url:
        return url[:url.index('?')]
    return url

def main(args):
    root_path = ""
    root_data_directory = args.output_folder
    root_path = root_data_directory

    websites = get_websites(args.links_folder, args.websites)
    websites = {k: v for k, v in websites.items() if len(v) > 0}
    print(websites)
    requests_per_webpage = int(args.requests_per_webpage)
    for i in range(requests_per_webpage):
        for website in websites:
            website = cut_suffix(website)
            website_path = create_folder(root_path, website)
            for url in websites[website]:
                issue_request(websites, website_path, i, url, args.starting_index)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="preprocessing parameters")    
    parser.add_argument("--links_folder", help="path to a folder that contains subfolders for servers, \
                        and each subfolder contains a links.txt file with different links to download")
    
    parser.add_argument("--websites", help="name of the websites to use", nargs="+", default=[])
    
    parser.add_argument("--output_folder", help="path to a folder that contains subfolders for servers. \
                        Each subfolder contains a folder for each link, and each link folder contains the traces \
                        for each request.")
    
    parser.add_argument("--requests_per_webpage", help="number of requests to issue per webpage")

    parser.add_argument("--starting_index", help="starting index for the trace names", type=str, default="0")
    
    args = parser.parse_args()
    main(args)


