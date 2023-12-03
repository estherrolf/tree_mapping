import rasterio
import subprocess
import geo_utils
import os
import matplotlib.pyplot as plt

data_dir = '../data'
# data_dir = '../../../tambe_lab/Users/luciagordon/tree_mapping_lucia_branch/data'   

# 1. process global map 2019 data
# 1a. crop the huge CHM prediction file to the extent of the merged S2 imagery so it's easier to work with later
lidar_dir = os.path.join(data_dir, 'raw/lidar_by_site_32736')
input_lidar_sites = [os.path.join(lidar_dir, x) for x in os.listdir(lidar_dir) if x.endswith('.tif')]
tmp_file = '/tmp/all_lidar_tif.tif'
geo_utils.merge_tifs(input_lidar_sites, tmp_file, output_type='Float32')

# 1b. align the 2019 predictions to the lidar 10m
fp_2019_in = os.path.join(data_dir, 'raw/global_tch_maps/Forest_height_2019_SAFR.tif')
fp_2019_cropped = os.path.join(data_dir, 'raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif')
geo_utils.crop_input_to_target_tif(fp_2019_in, 
                                   fp_2019_cropped, 
                                   target_fn = tmp_file, 
                                   output_type='Byte', # same as input
                                   pixel_buffer=40,
                                   output_nodata='255')
os.remove(tmp_file)

# 2. process ETH data
# 2a. combine the two ETH data files into one so it's easier to work with later
input_fns = [
    data_dir + '/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S24E030_Map.tif',
    data_dir + '/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S27E030_Map.tif',
]
tch_out_dir = os.path.join(data_dir,'int/global_tch_maps/')
if not os.path.exists(tch_out_dir): os.mkdir(tch_out_dir)
merged_eth_fn = os.path.join(tch_out_dir, 'ETH_GlobalCanopyHeight_10m_merged.tif')

geo_utils.merge_tifs(input_fns, merged_eth_fn, output_type='Byte') # same as input
merged_eth_fn = os.path.join(tch_out_dir, 'ETH_GlobalCanopyHeight_10m_merged.tif')

# 2b. make a per-site file of each
# make a per-site file of each to match the 10m resolution data
compare_map_dir = os.path.join(data_dir,'existing_reference_data')
if not os.path.exists(compare_map_dir): os.mkdir(compare_map_dir)
compare_eth_dir = os.path.join(compare_map_dir,'eth_maps_per_site_10m')
if not os.path.exists(compare_eth_dir): os.mkdir(compare_eth_dir)
compare_glad_dir = os.path.join(compare_map_dir,'glad_maps_per_site_10m')
if not os.path.exists(compare_glad_dir): os.mkdir(compare_glad_dir)
label_dir = os.path.join(data_dir, 'int/lidar/lidar_by_site_32736_10m')
site_names = os.listdir(label_dir)
print(len(site_names))
for site_name in site_names:
    # find the label file
    tif_fns = os.listdir(os.path.join(label_dir, site_name))
    assert len(tif_fns) == 1
    tif_fn = tif_fns[0]
    target_fn = os.path.join(label_dir, site_name, tif_fn)

    output_fn = os.path.join(compare_eth_dir, f'ETH_MAP_{site_name}_10m.tif')
    
    geo_utils.match_input_to_target_tif(input_fn=merged_eth_fn, 
                              output_fn=output_fn, 
                              target_fn=target_fn, 
                              pixel_buffer = 0, 
                              verbose=False, 
                              output_type='Float32',
                              resampling="average",
                              src_nodata='255',
                              output_nodata='-9999.')
    
    output_fn = os.path.join(compare_glad_dir, f'GLAD_MAP_{site_name}_10m.tif')
    
    geo_utils.match_input_to_target_tif(input_fn=fp_2019_cropped, 
                              output_fn=output_fn, 
                              target_fn=target_fn, 
                              pixel_buffer = 0, 
                              verbose=False, 
                              output_type='Float32',
                              resampling="average",
                              src_nodata='255',
                              output_nodata='-9999.')

with rasterio.open(output_fn) as f:
    data = f.read()
    
plt.hist(data[data != -9999.].ravel())
plt.savefig('histogram.png')
