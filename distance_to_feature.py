# imports
import matplotlib.pyplot as plt
import numpy as np
import os
from osgeo import gdal
from utils import get_project_dir

gdal.UseExceptions()

# directories
project_dir = get_project_dir()
lidar_10m_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_10m'

# process feature raster
feature = 'river'
feature_raster = gdal.Open(f'{feature}_raster.tiff')
FEATURE_NUM_ROWS = feature_raster.RasterYSize
FEATURE_NUM_COLS = feature_raster.RasterXSize
FEATURE_PIXEL_HEIGHT = feature_raster.GetGeoTransform()[5] # m
FEATURE_PIXEL_WIDTH = feature_raster.GetGeoTransform()[1] # m
FEATURE_TOP = feature_raster.GetGeoTransform()[3] # m
FEATURE_LEFT = feature_raster.GetGeoTransform()[0] # m
FEATURE_BOTTOM = FEATURE_TOP + FEATURE_PIXEL_HEIGHT * FEATURE_NUM_ROWS # m
FEATURE_RIGHT = FEATURE_LEFT + FEATURE_PIXEL_WIDTH * FEATURE_NUM_COLS # m
feature_array = ((feature_raster.GetRasterBand(1)).ReadAsArray(0, 0, FEATURE_NUM_COLS, FEATURE_NUM_ROWS).astype(np.float32)) # 0 = not feature, 255 = feature
feature_boolean_array = np.argwhere(feature_array == 255)

print(f'feature: pixel height = {FEATURE_PIXEL_HEIGHT}, pixel width = {FEATURE_PIXEL_WIDTH}')
print(f'feature: top = {str(FEATURE_TOP)}, bottom = {str(FEATURE_BOTTOM)}, left = {str(FEATURE_LEFT)}, right = {str(FEATURE_RIGHT)}')
print(f'feature raster shape = {feature_array.shape}')

def get_site_bounds(site):
    site_raster = gdal.Open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif')
    site_num_rows = site_raster.RasterYSize
    site_num_cols = site_raster.RasterXSize
    site_pixel_height = site_raster.GetGeoTransform()[5] # m
    site_pixel_width = site_raster.GetGeoTransform()[1] # m
    site_top = site_raster.GetGeoTransform()[3] # m
    site_left = site_raster.GetGeoTransform()[0] # m
    site_bottom = site_top + site_pixel_height * site_num_rows # m
    site_right = site_left + site_pixel_width * site_num_cols # m
    site_array = ((site_raster.GetRasterBand(1)).ReadAsArray(0, 0, site_num_cols, site_num_rows).astype(np.float32))
    # print(f'site: pixel height = {site_pixel_height}, pixel width = {site_pixel_width}')
    # print('site: top = ' + str(site_top) + ', bottom = ' + str(site_bottom) + ', left = ' + str(site_left) + ', right = ' + str(site_right))
    print(f'site raster shape = {site_array.shape}')

    return site_top, site_left, site_bottom, site_right

def get_distance_to_feature(index):
    distances_to_features = np.sqrt((feature_boolean_array[:, 0] - index[0])**2 + (feature_boolean_array[:, 1] - index[1])**2)
    min_distance = FEATURE_PIXEL_WIDTH * np.min(distances_to_features) # m
    
    return min_distance

def get_site_distances_to_feature(site):
    site_top, site_left, site_bottom, site_right = get_site_bounds(site)
    # river_array_cropped = river_array[int((site_top-RIVER_TOP)/RIVER_PIXEL_HEIGHT) : int(river_array.shape[0] + (site_bottom-RIVER_BOTTOM)/RIVER_PIXEL_HEIGHT), int((site_left-RIVER_LEFT)/RIVER_PIXEL_WIDTH) : int(river_array.shape[1] + (site_right-RIVER_RIGHT)/RIVER_PIXEL_WIDTH)].astype('uint8') # crop the river raster to cover the same area as the site raster
    site_top_in_feature_array = int((site_top-FEATURE_TOP)/FEATURE_PIXEL_HEIGHT)
    site_bottom_in_feature_array = int(feature_array.shape[0] + (site_bottom-FEATURE_BOTTOM)/FEATURE_PIXEL_HEIGHT)
    site_left_in_feature_array = int((site_left-FEATURE_LEFT)/FEATURE_PIXEL_WIDTH)
    site_right_in_feature_array = int(feature_array.shape[1] + (site_right-FEATURE_RIGHT)/FEATURE_PIXEL_WIDTH)
    # print('river array shape after cropping to match site =', river_array_cropped.shape)

    # index = [300, 400]
    # print(get_distance_to_feature(river_boolean_array, index))
    # river_array_cropped[300, 400] = 255

    # plt.figure(dpi=300)
    # plt.imshow(river_array_cropped) # plot the array of pixel values as an image
    # plt.axis('off') # remove axes        
    # plt.savefig('river_array_cropped.png', bbox_inches='tight', pad_inches=0)
    # plt.close() # close the image to save memory

    site_distances_to_feature = np.array([[get_distance_to_feature([row, col]) for col in range(site_left_in_feature_array, site_right_in_feature_array)] for row in range(site_top_in_feature_array, site_bottom_in_feature_array)])
    # print(site_distances_to_feature)
    print(f'site distances to feature shape = {site_distances_to_feature.shape}')
    return site_distances_to_feature

sites = sorted(os.listdir(lidar_10m_dir))
# site = sites[0]

# import rasterio
# with rasterio.open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif') as file:
#     site_labels = file.read().ravel()
# print(f'len site labels = {len(site_labels)}')
# print(f'len distances to feature = {len(get_site_distances_to_feature(site))}')

os.makedirs(f'{project_dir}/distances-to-{feature}', exist_ok=True)

for site in sites:
    np.save(f'{project_dir}/distances-to-{feature}/{site}_distances_to_{feature}', get_site_distances_to_feature(site))
