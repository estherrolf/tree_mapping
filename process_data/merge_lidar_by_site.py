# imports
from osgeo import gdal
import os
import sys

sys.path.insert(0, '') # necessary since utils is outside the process_data folder
from utils import get_project_dir

data_dir = f'{get_project_dir()}/data'

def merge_raw_lidar_tiffs():
    '''Merges the 1m-resolution lidar tiffs for each site into a single tiff'''
    raw_lidar_dir = f'{data_dir}/raw/raw_lidar' # folder of folders of input lidar tiffs
    raw_lidar_dir_items = os.listdir(raw_lidar_dir) # folders of input lidar tiffs
    site_dirs = [item for item in raw_lidar_dir_items if not item.startswith('.')] # folders of input lidar tiffs
    lidar_1m_dir = f'{data_dir}/raw/lidar_by_site_32736' # folder of merged 1m lidar tiffs
    os.makedirs(lidar_1m_dir, exist_ok=True) # make 1m lidar directory

    for site_dir in site_dirs:
        site_dir_items = os.listdir(f'{raw_lidar_dir}/{site_dir}/CHM') # items in site folder
        site_tiffs = [f'{raw_lidar_dir}/{site_dir}/CHM/{tiff}' for tiff in site_dir_items if tiff.endswith('.tif')] # tiffs in site folder
        gdal.Warp(destNameOrDestDS=f'{lidar_1m_dir}/{site_dir.split("_")[1]}_CHM_1m_merged.tif', srcDSOrSrcDSTab=site_tiffs, format='GTiff', srcNodata='-9999.0', dstNodata='-9999.0')
        print(f'Site {site_dir.split("_")[1]} merged')

def coarsen_lidar_1m_tiffs(target_resolution):
    '''Coarsens the 1m-resolution lidar tiff for each site to the target resolution'''
    lidar_1m_dir_tiffs = os.listdir(f'{data_dir}/raw/lidar_by_site_32736') # merged 1m lidar tiffs
    lidar_coarsened_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_{target_resolution}m' # folder of coarsened lidar tiffs
    os.makedirs(lidar_coarsened_dir, exist_ok=True) # make coarsened lidar directory

    for tiff in lidar_1m_dir_tiffs:
        site = tiff.split('_')[0]
        os.makedirs(f'{lidar_coarsened_dir}/{site}', exist_ok=True)
        gdal.Warp(destNameOrDestDS=f'{lidar_coarsened_dir}/{site}/{site}_CHM_{target_resolution}m.tif', srcDSOrSrcDSTab=f'{data_dir}/raw/lidar_by_site_32736/{tiff}', format='GTiff', outputType=gdal.gdalconst.GDT_Float32, resampleAlg='average', xRes=target_resolution, yRes=target_resolution, srcNodata='-9999.0', dstNodata='-9999.0')
        print(f'Site {site} coarsened to {target_resolution}_m')

merge_raw_lidar_tiffs()
coarsen_lidar_1m_tiffs(target_resolution=10)
coarsen_lidar_1m_tiffs(target_resolution=30)
