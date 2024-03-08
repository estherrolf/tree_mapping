#!/bin/bash

# download the Sentinel data
python acquire_data/download_sentinel.py

# merge the individual lidar tiffs
python process_data/merge_lidar_by_site.py

# extract the global map data corresponding to the sites
python process_data/match_reference_maps_to_sites.py

# extract the Sentinel data corresponding to the sites
python process_data/match_sentinel_to_sites.py