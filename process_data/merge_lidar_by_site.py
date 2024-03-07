# imports
from osgeo import gdal
import math
import numpy as np
import os
import rasterio
import sys

sys.path.insert(0, '') # necessary since utils is outside the process_data folder
from utils import get_project_dir

data_dir = f'{get_project_dir()}/data'
NAN = -9999.0

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
        gdal.Warp(destNameOrDestDS=f'{lidar_1m_dir}/{site_dir.split("_")[1]}_CHM_1m_merged.tif', srcDSOrSrcDSTab=site_tiffs, format='GTiff', srcNodata=NAN, dstNodata=NAN)
        print(f'Site {site_dir.split("_")[1]} merged')

def preprocess_lidar_1m_tiffs():
    '''Sets values in the 1m tiffs above 30m to NaN (-9999.0) and non-NaN negative values to 0'''

    lidar_1m_merged_dir = f'{data_dir}/raw/lidar_by_site_32736_merged'
    lidar_1m_merged_dir_tiffs = os.listdir(lidar_1m_merged_dir) # merged 1m lidar tiffs
    lidar_1m_dir = f'{data_dir}/raw/lidar_by_site_32736' # folder of preprocessed 1m lidar tiffs
    os.makedirs(lidar_1m_dir, exist_ok=True) # make 1m lidar directory

    for tiff_path in lidar_1m_merged_dir_tiffs:
        with rasterio.open(f'{lidar_1m_merged_dir}/{tiff_path}') as tiff:
            metadata = tiff.meta
            tiff_array = tiff.read()
            tiff_array[(tiff_array < 0) & (tiff_array != NAN)] = 0 # sets non-NaN negative values to 0
            tiff_array[tiff_array > 30] = NAN # sets values > 30 to the no-data value

            with rasterio.open(f'{lidar_1m_dir}/{tiff_path}', 'w', **metadata) as out_file: # replaces the original tiff with a new tiff with the same metadata, just different values
                out_file.write(tiff_array)

def coarsen_tiff(output_path, input_path, target_resolution):
    '''Coarsens an input tiff to a target resolution and saves it'''

    with rasterio.open(input_path) as input_tiff:
        input_tiff_array = input_tiff.read(1) # extracts the first band of the tiff as an array
        num_input_tiff_rows, num_input_tiff_cols = input_tiff.shape
        input_resolution = input_tiff.res[0] # meters per pixel
        input_tiff_metadata = input_tiff.meta # dictionary of driver, dtype, nodata, width, height, count, crs, & transform
        factor = int(target_resolution / input_resolution) # downsampling factor
        coarsened_tiff_array = np.full((int(math.ceil(num_input_tiff_rows/factor)), int(math.ceil(num_input_tiff_cols/factor))), NAN) # initializes an array filled with NaNs of the proper size for the coarsened tiff
        print(f'Input tiff shape = {input_tiff_array.shape}')

        for top in range(0, num_input_tiff_rows, factor): # begins from the top of the padded input array
            for left in range(0, num_input_tiff_cols, factor): # begins from the left end of the padded input array
                block = input_tiff_array[top : top+factor, left : left+factor].copy()
                block_without_nans = block[block != NAN]
                coarsened_tiff_array[int(top / factor), int(left / factor)] = NAN if len(block_without_nans) == 0 else np.percentile(a=block_without_nans, q=90) # sets a pixel in the coarsened tiff array to be the 90th percentile of a sub-array in the input tiff
        
        print(f'Coarsened tiff shape = {coarsened_tiff_array.shape}')
        coarsened_tiff_metadata = input_tiff_metadata.copy()
        coarsened_tiff_metadata['height'] = coarsened_tiff_array.shape[0] # sets the height of the coarsened tiff to the number of rows in its array
        coarsened_tiff_metadata['width'] = coarsened_tiff_array.shape[1] # sets the width of the coarsened tiff to the number of columns in its array
        coarsened_tiff_metadata['transform'] = rasterio.transform.Affine(factor*input_tiff.transform[0], input_tiff.transform[1], input_tiff.transform[2], input_tiff.transform[3], factor*input_tiff.transform[4], input_tiff.transform[5]) # sets the x- and y-resolutions of the coarsened tiff

        with rasterio.open(output_path, 'w', **coarsened_tiff_metadata) as out_file:
            out_file.write(coarsened_tiff_array, indexes=1) # saves the coarsened tiff

def coarsen_lidar_1m_tiffs(target_resolution):
    '''Coarsens the 1m-resolution lidar tiff for each site to the target resolution'''

    lidar_1m_dir = f'{data_dir}/raw/lidar_by_site_32736' # folder of preprocessed 1m lidar tiffs
    lidar_1m_dir_tiffs = os.listdir(lidar_1m_dir) # preprocessed 1m lidar tiffs
    lidar_coarsened_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_{target_resolution}m' # folder of coarsened lidar tiffs
    os.makedirs(lidar_coarsened_dir, exist_ok=True) # make coarsened lidar directory

    for tiff in lidar_1m_dir_tiffs:
        site = tiff.split('_')[0]
        print(f'Coarsening {site} to {target_resolution}m')
        os.makedirs(f'{lidar_coarsened_dir}/{site}', exist_ok=True)
        coarsen_tiff(output_path=f'{lidar_coarsened_dir}/{site}/{site}_CHM_{target_resolution}m.tif', input_path=f'{lidar_1m_dir}/{tiff}', target_resolution=target_resolution)

if __name__ == '__main__':
    merge_raw_lidar_tiffs()
    preprocess_lidar_1m_tiffs()
    coarsen_lidar_1m_tiffs(target_resolution=10)
    coarsen_lidar_1m_tiffs(target_resolution=30)
