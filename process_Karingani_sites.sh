#!/bin/bash

# Change to the directory containing the Python script
cd /n/home10/erolf/tree_mapping/process_data

# merge the individual lidar tifs
python merge_lidar_by_site.py

# align the lidar tifs with the rest of the data
#python align_layers_by_site.py