# imports
import rasterio
import subprocess
import geo_utils
import os
import matplotlib.pyplot as plt

data_dir = '../data'
# data_dir = '../../../tambe_lab/Users/luciagordon/tree_mapping_lucia_branch/data'   

def merge_ETH_maps():
    '''Merge the two ETH data files into one'''
    input_fns = [f'{data_dir}/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S24E030_Map.tif',
                f'{data_dir}/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S27E030_Map.tif']
    tch_out_dir = f'{data_dir}/int/global_tch_maps'
    if not os.path.exists(tch_out_dir): os.mkdir(tch_out_dir)
    merged_eth_fn = f'{tch_out_dir}/ETH_GlobalCanopyHeight_10m_merged.tif'

    geo_utils.merge_tifs(input_fns, merged_eth_fn, output_type='Byte') # same as input

    return merged_eth_fn

def crop_GLAD_map():
    '''Crop the GLAD map to the extent of the merged LiDAR imagery'''
    lidar_dir = f'{data_dir}/raw/lidar_by_site_32736'
    input_lidar_sites = [f'{lidar_dir}/{x}' for x in os.listdir(lidar_dir) if x.endswith('.tif')]
    tmp_file = f'{data_dir}/all_lidar_tif.tif'
    geo_utils.merge_tifs(input_lidar_sites, tmp_file, output_type='Float32')

    fp_2019_in = f'{data_dir}/raw/global_tch_maps/Forest_height_2019_SAFR.tif'
    fp_2019_cropped = f'{data_dir}/raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif'
    geo_utils.crop_input_to_target_tif(fp_2019_in, 
                                       fp_2019_cropped, 
                                       target_fn = tmp_file, 
                                       output_type = 'Byte', # same as input
                                       pixel_buffer = 40,
                                       output_nodata = '255')
    os.remove(tmp_file)

    return fp_2019_cropped

def make_per_site_files(map, input_fn):
    '''Make per-site files for the ETH and GLAD maps to match the 10m resolution data'''
    compare_map_dir = f'{data_dir}/existing_reference_data'
    if not os.path.exists(compare_map_dir): os.mkdir(compare_map_dir)
    compare_dir = f'{compare_map_dir}/{map}_maps_per_site_10m'
    if not os.path.exists(compare_dir): os.mkdir(compare_dir)
    label_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_10m'
    site_names = os.listdir(label_dir)
    print(len(site_names))
    for site_name in site_names:
        # find the label file
        tif_fns = os.listdir(f'{label_dir}/{site_name}')
        assert len(tif_fns) == 1
        tif_fn = tif_fns[0]
        target_fn = f'{label_dir}/{site_name}/{tif_fn}'
        output_fn = f'{compare_dir}/{map.upper()}_MAP_{site_name}_10m.tif'
        
        geo_utils.match_input_to_target_tif(input_fn=input_fn, 
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

    plt.figure(dpi = 300)
    plt.hist(data[data != -9999.].ravel())
    plt.savefig(f'{map}-histogram.png')
        
if __name__ == '__main__':
    merged_eth_fn = merge_ETH_maps()
    fp_2019_cropped = crop_GLAD_map()

    make_per_site_files('eth', merged_eth_fn)
    make_per_site_files('glad', fp_2019_cropped)
