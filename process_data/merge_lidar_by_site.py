# imports
from osgeo import gdal
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
        tiff = rasterio.open(f'{lidar_1m_merged_dir}/{tiff_path}')
        metadata = tiff.meta
        tiff_array = tiff.read()
        tiff_array[(tiff_array < 0) & (tiff_array != NAN)] = 0 # sets non-NaN negative values to 0
        tiff_array[tiff_array > 30] = NAN # sets values > 30 to the no-data value

        with rasterio.open(f'{lidar_1m_dir}/{tiff_path}', 'w', **metadata) as out_file: # replaces the original tiff with a new tiff with the same metadata, just different values
            out_file.write(tiff_array)

def pad_array(array, pad_value, row_divisor, column_divisor):
    '''Pads an array by adding rows and/or columns filled with a constant value to the bottom and/or right of the array'''

    num_input_rows, num_input_cols = array.shape
    num_rows_to_append = int((row_divisor - num_input_rows%row_divisor) % row_divisor) # divisible by row divisor
    num_cols_to_append = int((column_divisor - num_input_cols%column_divisor) % column_divisor) # divisible by column divisor
    rows_to_append = np.full((num_rows_to_append, num_input_cols), pad_value)
    padded_array = np.vstack((array, rows_to_append)) # adds rows to the bottom
    cols_to_append = np.full((padded_array.shape[0], num_cols_to_append), pad_value)
    padded_array = np.hstack((padded_array, cols_to_append)) # adds columns to the right

    return padded_array

def calculate_percentile(array, percentile):
    '''Calculates the percentile of the non-NaN values of an array according to the linear interpolation method'''

    array = np.array(array) # converts array to a numpy array if it is not already
    array_without_nans = array[array != NAN] # 1D array of non-NaN values

    if len(array_without_nans) == 0: # if the original array contains all NaNs
        return NAN

    sorted_array = sorted(array_without_nans)
    index = percentile/100 * (len(sorted_array) - 1)

    if index.is_integer():
        return sorted_array[int(index)]

    # linear interpolation between the lower and upper indices is performed if the index is not an integer
    lower_index = int(index)
    upper_index = lower_index + 1
    interpolation = index - lower_index
    percentile = sorted_array[lower_index] + interpolation * (sorted_array[upper_index] - sorted_array[lower_index])

    return percentile

def coarsen_tiff(output_path, input_path, target_resolution):
    '''Coarsens an input tiff to a target resolution and saves it'''

    input_tiff = rasterio.open(input_path)
    input_tiff_array = input_tiff.read(1) # extracts the first band of the tiff as an array
    input_resolution = input_tiff.res[0] # meters per pixel
    input_tiff_metadata = input_tiff.meta # dictionary of driver, dtype, nodata, width, height, count, crs, & transform
    factor = int(target_resolution / input_resolution) # downsampling factor
    print(f'Input tiff shape = {input_tiff_array.shape}')
    padded_tiff_array = pad_array(array=input_tiff_array, pad_value=NAN, row_divisor=target_resolution, column_divisor=factor) # pads the input tiff so that the numbers of rows and columns are divisible by the downsampling factor
    num_padded_tiff_rows, num_padded_tiff_cols = padded_tiff_array.shape
    print(f'Padded tiff shape = {num_padded_tiff_rows, num_padded_tiff_cols}')
    coarsened_tiff_array = np.full((int(num_padded_tiff_rows / factor), int(num_padded_tiff_cols / factor)), NAN) # initializes an array filled with NaNs of the proper size for the coarsened tiff

    for top in range(0, num_padded_tiff_rows, factor): # begins from the top of the padded input array
        for left in range(0, num_padded_tiff_cols, factor): # begins from the left end of the padded input array
            coarsened_tiff_array[int(top / factor), int(left / factor)] = calculate_percentile(array=padded_tiff_array[top : top+factor, left : left+factor], percentile=90) # sets a pixel in the coarsened tiff array to be the 90th percentile of a factor x factor sub-array in the padded input tiff

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
