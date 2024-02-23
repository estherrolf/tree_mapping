# imports
from osgeo import gdal
import numpy as np
import os
import rasterio
import sys

sys.path.insert(0, '') # necessary since utils is outside the process_data folder
from utils import get_project_dir

data_dir = f'{get_project_dir()}/data'
nan = -9999.0

def merge_raw_lidar_tiffs():
    '''Merges the 1m-resolution lidar tiffs for each site into a single tiff'''
    raw_lidar_dir = f'{data_dir}/raw/raw_lidar' # folder of folders of input lidar tiffs
    raw_lidar_dir_items = os.listdir(raw_lidar_dir) # folders of input lidar tiffs
    site_dirs = [item for item in raw_lidar_dir_items if not item.startswith('.')] # folders of input lidar tiffs
    lidar_1m_dir = f'{data_dir}/raw/lidar_by_site_32736_merged' # folder of merged 1m lidar tiffs
    os.makedirs(lidar_1m_dir, exist_ok=True) # make 1m lidar directory

    for site_dir in site_dirs:
        site_dir_items = os.listdir(f'{raw_lidar_dir}/{site_dir}/CHM') # items in site folder
        site_tiffs = [f'{raw_lidar_dir}/{site_dir}/CHM/{tiff}' for tiff in site_dir_items if tiff.endswith('.tif')] # tiffs in site folder
        gdal.Warp(destNameOrDestDS=f'{lidar_1m_dir}/{site_dir.split("_")[1]}_CHM_1m_merged.tif', srcDSOrSrcDSTab=site_tiffs, format='GTiff', srcNodata=nan, dstNodata=nan)
        print(f'Site {site_dir.split("_")[1]} merged')

def preprocess_lidar_1m_tiffs():
    '''Sets values in the 1m tiffs outside of 0-30m to NaN (-9999.0)'''
    lidar_1m_merged_dir = f'{data_dir}/raw/lidar_by_site_32736_merged'
    lidar_1m_merged_dir_tiffs = os.listdir(lidar_1m_merged_dir) # merged 1m lidar tiffs
    lidar_1m_dir = f'{data_dir}/raw/lidar_by_site_32736' # folder of preprocessed 1m lidar tiffs
    os.makedirs(lidar_1m_dir, exist_ok=True) # make 1m lidar directory

    for tiff in lidar_1m_merged_dir_tiffs:
        with rasterio.open(f'{lidar_1m_merged_dir}/{tiff}') as file:
            metadata = file.meta
            tiff_array = file.read()
            tiff_array[(tiff_array < 0) | (tiff_array > 30)] = nan # sets values outside of the 0-30 range to the no-data value

            with rasterio.open(f'{lidar_1m_dir}/{tiff}', 'w', **metadata) as out_file: # replaces the original tiff with a new tiff with the same metadata, just different values
                out_file.write(tiff_array)

def coarsen_lidar_1m_tiffs(target_resolution):
    '''Coarsens the 1m-resolution lidar tiff for each site to the target resolution'''
    lidar_1m_dir = f'{data_dir}/raw/lidar_by_site_32736' # folder of preprocessed 1m lidar tiffs
    lidar_1m_dir_tiffs = os.listdir(lidar_1m_dir) # preprocessed 1m lidar tiffs
    lidar_coarsened_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_{target_resolution}m' # folder of coarsened lidar tiffs
    os.makedirs(lidar_coarsened_dir, exist_ok=True) # make coarsened lidar directory

    for tiff in lidar_1m_dir_tiffs:
        site = tiff.split('_')[0]
        os.makedirs(f'{lidar_coarsened_dir}/{site}', exist_ok=True)
        gdal.Warp(destNameOrDestDS=f'{lidar_coarsened_dir}/{site}/{site}_CHM_{target_resolution}m.tif', srcDSOrSrcDSTab=f'{lidar_1m_dir}/{tiff}', format='GTiff', outputType=gdal.gdalconst.GDT_Float32, resampleAlg='average', xRes=target_resolution, yRes=target_resolution, srcNodata=nan, dstNodata=nan)
        print(f'Site {site} coarsened to {target_resolution}m')

if __name__ == '__main__':
    merge_raw_lidar_tiffs()
    preprocess_lidar_1m_tiffs()
    coarsen_lidar_1m_tiffs(target_resolution=10)
    coarsen_lidar_1m_tiffs(target_resolution=30)
