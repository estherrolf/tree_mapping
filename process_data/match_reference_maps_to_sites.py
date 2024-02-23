# imports
from osgeo import gdal
import geo_utils
import os
import matplotlib.pyplot as plt
import rasterio
import sys

sys.path.insert(0, '') # necessary since utils is outside the process_data folder
from utils import get_project_dir

data_dir = f'{get_project_dir()}/data'

def merge_ETH_maps():
    '''Merges the two ETH data files into one'''
    eth_tiles = [f'{data_dir}/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S24E030_Map.tif',
                 f'{data_dir}/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S27E030_Map.tif']
    global_map_dir = f'{data_dir}/int/global_tch_maps'
    os.makedirs(global_map_dir, exist_ok=True)
    merged_eth_tiff = f'{global_map_dir}/ETH_GlobalCanopyHeight_10m_merged.tif'
    gdal.Warp(destNameOrDestDS=merged_eth_tiff, srcDSOrSrcDSTab=eth_tiles, srcNodata=255, dstNodata=255, format='GTiff', outputType=gdal.gdalconst.GDT_Byte)
    print('ETH tiffs merged')

def set_NaN_GLAD_map():
    '''Sets pixel values corresponding to water, snow, ice, or NaNs to a new NaN value of 255'''
    glad_tiff = f'{data_dir}/raw/global_tch_maps/Forest_height_2019_SAFR.tif'

    with rasterio.open(glad_tiff) as file:
        metadata = file.meta
        tiff_array = file.read()
        tiff_array[tiff_array > 100] = 255 # sets values of 101 (water), 102 (snow/ice), and 103 (NaN) to a new NaN value of 255

        with rasterio.open(f'{data_dir}/raw/global_tch_maps/Forest_height_2019_SAFR_processed.tif', 'w', **metadata) as out_file:
            out_file.write(tiff_array)
    
    print('GLAD tiff NaN set')

def make_per_site_files(map, map_tiff, resolution):
    '''Makes per-site files for the ETH and GLAD maps to match the coarsened LiDAR data'''
    ref_data_dir = f'{data_dir}/existing_reference_data'
    global_map_by_site_dir = f'{ref_data_dir}/{map}_maps_per_site_{resolution}m'
    os.makedirs(global_map_by_site_dir, exist_ok=True)
    lidar_coarsened_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_{resolution}m'

    for site in os.listdir(lidar_coarsened_dir):
        global_map_site_tiff = f'{global_map_by_site_dir}/{map.upper()}_MAP_{site}_{resolution}m.tif'
        lidar_site_tiff = f'{lidar_coarsened_dir}/{site}/{site}_CHM_{resolution}m.tif'

        geo_utils.match_input_to_target_tif(input_fn=map_tiff,
                                            output_fn=global_map_site_tiff,
                                            target_fn=lidar_site_tiff,
                                            output_type='Float32',
                                            resampling='average',
                                            src_nodata='255',
                                            output_nodata='-9999.0')
    
    print('Generated per-site files for {map}')

if __name__ == '__main__':
    merge_ETH_maps()
    crop_GLAD_map()
    
    for resolution in [10, 30]:
        make_per_site_files(map='eth', map_tiff=f'{data_dir}/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged.tif', resolution=resolution)
        make_per_site_files(map='glad', map_tiff=f'{data_dir}/raw/global_tch_maps/Forest_height_2019_SAFR_processed.tif', resolution=resolution)
    
    os.remove(f'{data_dir}/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged.tif')
    os.remove(f'{data_dir}/raw/global_tch_maps/Forest_height_2019_SAFR_processed.tif')
