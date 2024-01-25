from osgeo import gdal
from utils import get_project_dir
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio

gdal.UseExceptions()

project_dir = get_project_dir()

'''compare tiffs'''
# site = 1

# mine = f'../../../tambe_lab/Users/luciagordon/tree_mapping/data/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'
# # mine2 = f'../../../tambe_lab/Users/luciagordon/tree_mapping_test_2/data/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'
# hers = f'../../../tambe_lab/Everyone/Karingani_data/esther_chm_merging_jan22_2024/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'
# hers_old = f'../../../tambe_lab/Everyone/Karingani_data/esther_chm_merging_jan11_2024/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'
# # this = f'../../../tambe_lab/Users/luciagordon/tree-mapping-v1/data/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'

# # mine = f'../../../tambe_lab/Users/luciagordon/tree_mapping_test/data/int/lidar/lidar_by_site_32736_10m/KaringaniSite0{site}/KaringaniSite0{site}_CHM_10m.tif'
# # hers = f'../../../tambe_lab/Everyone/Karingani_data/esther_lidar_outputs_dec_19/lidar_by_site_32736_10m/KaringaniSite0{site}/KaringaniSite0{site}_CHM_10m.tif'
# # this = f'../../../tambe_lab/Users/luciagordon/tree-mapping-v1/data/int/lidar/lidar_by_site_32736_10m/KaringaniSite0{site}/KaringaniSite0{site}_CHM_10m.tif'
# print(rasterio.__version__) # 1.3.9
# def process_tiff(path):
#     tiff = gdal.Open(path)
#     num_rows = tiff.RasterYSize
#     num_cols = tiff.RasterXSize
#     array = ((tiff.GetRasterBand(1)).ReadAsArray(0, 0, num_cols, num_rows).astype(np.float32))
#     return array

# my_arr = process_tiff(mine)
# # my_arr2 = process_tiff(mine2)
# her_arr = process_tiff(hers)
# her_old_arr = process_tiff(hers_old)
# # this_arr = process_tiff(this)

# # print((my_arr == my_arr2).all())
# print((my_arr == her_arr).all())
# print((her_arr == her_old_arr).all())

# print((this_arr == her_arr).all())
# print((this_arr == my_arr).all())

'''check what % of the 10m data is outside the 0-30 m height range'''
lidar_10m_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_10m'
sites = os.listdir(lidar_10m_dir)
num_points_in_range = 0
total_points = 0

for site in sites:
    with rasterio.open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif') as file:
        site_labels = file.read().ravel()
        num_points_in_range += len(np.where((site_labels >= 0) & (site_labels <= 30))[0])
        total_points += len(np.where(site_labels != -9999)[0])
        # for label in site_labels:
        #     if label < 0:
        #         if np.isnan(label): 
        #             print(label)
        # total_points += np.count_nonzero(~np.isnan(site_labels))
        # total_points += len(site_labels)

print(f'Percentage of points in range = {100*num_points_in_range/total_points}')

'''plot distance to feature'''
# lidar_10m_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_10m'
# sites = os.listdir(lidar_10m_dir)
# site = sites[1]
# array = np.load(f'{project_dir}/distances-to-river/{site}_distances_to_river.npy')

# plt.figure(dpi=300)
# plt.imshow(array) # plot the array of pixel values as an image
# plt.axis('off') # remove axes        
# plt.savefig(f'site-{site}-distance-to-river.png', bbox_inches='tight', pad_inches=0)
# plt.close() # close the image to save memory

# site_raster = gdal.Open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif')
# site_num_rows = site_raster.RasterYSize
# site_num_cols = site_raster.RasterXSize
# site_pixel_height = site_raster.GetGeoTransform()[5] # m
# site_pixel_width = site_raster.GetGeoTransform()[1] # m
# site_top = site_raster.GetGeoTransform()[3] # m
# site_left = site_raster.GetGeoTransform()[0] # m
# site_bottom = site_top + site_pixel_height * site_num_rows # m
# site_right = site_left + site_pixel_width * site_num_cols # m

# feature_raster = gdal.Open(f'river_raster.tiff')
# FEATURE_NUM_ROWS = feature_raster.RasterYSize
# FEATURE_NUM_COLS = feature_raster.RasterXSize
# FEATURE_PIXEL_HEIGHT = feature_raster.GetGeoTransform()[5] # m
# FEATURE_PIXEL_WIDTH = feature_raster.GetGeoTransform()[1] # m
# FEATURE_TOP = feature_raster.GetGeoTransform()[3] # m
# FEATURE_LEFT = feature_raster.GetGeoTransform()[0] # m
# FEATURE_BOTTOM = FEATURE_TOP + FEATURE_PIXEL_HEIGHT * FEATURE_NUM_ROWS # m
# FEATURE_RIGHT = FEATURE_LEFT + FEATURE_PIXEL_WIDTH * FEATURE_NUM_COLS # m
# feature_array = ((feature_raster.GetRasterBand(1)).ReadAsArray(0, 0, FEATURE_NUM_COLS, FEATURE_NUM_ROWS).astype(np.float32)) # 0 = not feature, 255 = feature
# site_top_in_feature_array = int((site_top-FEATURE_TOP)/FEATURE_PIXEL_HEIGHT)
# site_bottom_in_feature_array = int(feature_array.shape[0] + (site_bottom-FEATURE_BOTTOM)/FEATURE_PIXEL_HEIGHT)
# site_left_in_feature_array = int((site_left-FEATURE_LEFT)/FEATURE_PIXEL_WIDTH)
# site_right_in_feature_array = int(feature_array.shape[1] + (site_right-FEATURE_RIGHT)/FEATURE_PIXEL_WIDTH)

# for i in range(site_top_in_feature_array, site_bottom_in_feature_array+1):
#     for j in range(site_left_in_feature_array, site_right_in_feature_array+1):
#         if (i == site_top_in_feature_array or i == site_bottom_in_feature_array) or (j == site_left_in_feature_array or j == site_right_in_feature_array):
#             feature_array[i][j] = 255

# plt.figure(dpi=300)
# plt.imshow(feature_array)
# plt.axis('off') # remove axes        
# plt.savefig(f'river-raster-{site}-boxed.png', bbox_inches='tight', pad_inches=0)
# plt.close() # close the image to save memory

'''get maximum distance to river'''
# distances_to_river_dir = f'{project_dir}/distances-to-river'
# max_distances = []

# for array in os.listdir(distances_to_river_dir):
#     max_distances += [np.max(np.load(f'{distances_to_river_dir}/{array}'))]

# print(np.max(max_distances)) # 2854 m
