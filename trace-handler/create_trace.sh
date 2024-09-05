#!/bin/bash
# for WEBSITE in adobe.com bleacherreport.com cdnetworks.com cloudflare.com cnn.com discord.com facebook.com google.com independent.co.uk instagram.com jetbrains.com logitech.com mercedes-benz.com nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com youtube.com
# for WEBSITE in cnn.com discord.com facebook.com google.com independent.co.uk instagram.com jetbrains.com logitech.com mercedes-benz.com nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com youtube.com
# for WEBSITE in youtube.com
# for WEBSITE in logitech.com mercedes-benz.com nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com youtube.com facebook.com google.com independent.co.uk instagram.com jetbrains.com 
# for WEBSITE in nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com youtube.com facebook.com google.com independent.co.uk instagram.com jetbrains.com 
# for WEBSITE in bleacherreport.com cdnetworks.com cloudflare.com cnn.com discord.com facebook.com google.com independent.co.uk instagram.com jetbrains.com logitech.com mercedes-benz.com nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com youtube.com
# for WEBSITE in adobe.com bleacherreport.com cdnetworks.com cloudflare.com cnn.com discord.com facebook.com google.com independent.co.uk instagram.com jetbrains.com logitech.com mercedes-benz.com nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com youtube.com
# for WEBSITE in pinterest.com jetbrains.com mercedes-benz.com facebook.com instagram.com logitech.com discord.com
# for WEBSITE in google.com youtube.com cloudflare.com pcmag.com pinterest.com instagram.com cnn.com jetbrains.com wiggle.com nicelocal.com discord.com cdnetworks.com mercedes-benz.com independent.co.uk semrush.com logitech.com facebook.com
# for WEBSITE in mercedes-benz.com facebook.com bleacherreport.com pinterest.com instagram.com cnn.com


# for WEBSITE in mercedes-benz.com pinterest.com instagram.com
# do
#     # python3 create_trace.py --links_folder=output_links --websites $WEBSITE --output_folder=./v4_data --requests_per_webpage=1 --starting_index=0004
    
#     python3 create_trace.py --links_folder=output_links_new --websites $WEBSITE --output_folder=/Volumes/ELEMENTS/quic-classification/v4_data65 --requests_per_webpage=40 --starting_index=0089

#     # ./pcap_to_csv.sh
#     # sshpass -p '' scp -r v3_data/* barakg@132.68.39.111:/home/barakg/deepquic/quic-classification/v3_data
#     # rm -rf v3_data/*
# done

for WEBSITE in youtube.com adobe.com bleacherreport.com cdnetworks.com cloudflare.com cnn.com discord.com facebook.com google.com independent.co.uk instagram.com jetbrains.com logitech.com mercedes-benz.com nicelocal.com pcmag.com pinterest.com semrush.com wiggle.com
do
    # python3 create_trace.py --links_folder=output_links --websites $WEBSITE --output_folder=./v4_data --requests_per_webpage=1 --starting_index=0004
    
    # python3 create_trace.py --links_folder=output_links_new --websites $WEBSITE --output_folder=/Volumes/ELEMENTS/quic-classification/temp --requests_per_webpage=1 --starting_index=0089
    python3 create_trace.py --links_folder=../links-for-request --websites $WEBSITE --output_folder=/mnt/sdc/trace-collector/VisQUIC/v5_data8 --requests_per_webpage=1 --starting_index=0090

    # ./pcap_to_csv.sh
    # sshpass -p '' scp -r v3_data/* barakg@132.68.39.111:/home/barakg/deepquic/quic-classification/v3_data  #TODO: remove this before the commit
    # rm -rf v3_data/*
done
