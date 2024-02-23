# imports
from osgeo import gdal
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio
import yaml

gdal.UseExceptions()

def get_project_dir():
    with open('project_dir.yaml', 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)

    return cfg['project_dir']

def tiff_to_array(path):
    tiff = gdal.Open(path)
    num_rows = tiff.RasterYSize
    num_cols = tiff.RasterXSize
    array = ((tiff.GetRasterBand(1)).ReadAsArray(0, 0, num_cols, num_rows).astype(np.float32))
    return array

def compare_tiffs():
    sites = os.listdir('../../../tambe_lab/Users/luciagordon/tree_mapping_email/data/int/lidar/lidar_by_site_32736_10m')

    for site in sites:
        my_10m_feb19 = f'../../../tambe_lab/Users/luciagordon/tree_mapping_feb19/data/int/lidar/lidar_by_site_32736_10m/{site}/{site}_CHM_10m.tif'
        my_30m_feb19 = f'../../../tambe_lab/Users/luciagordon/tree_mapping_feb19/data/int/lidar/lidar_by_site_32736_30m/{site}/{site}_CHM_30m.tif'

        my_10m = f'../../../tambe_lab/Users/luciagordon/tree_mapping_warp/data/int/lidar/lidar_by_site_32736_10m/{site}/{site}_CHM_10m.tif'
        my_30m = f'../../../tambe_lab/Users/luciagordon/tree_mapping_warp/data/int/lidar/lidar_by_site_32736_30m/{site}/{site}_CHM_30m.tif'

        # her_10m = f'../../../tambe_lab/Everyone/Karingani_data/esther_10m_30m_using_lucias1m/int/lidar/lidar_by_site_32736_10m/{site}/{site}_CHM_10m.tif'
        # her_30m = f'../../../tambe_lab/Everyone/Karingani_data/esther_10m_30m_using_lucias1m/int/lidar/lidar_by_site_32736_30m/{site}/{site}_CHM_30m.tif'

        # mine = f'../../../tambe_lab/Users/luciagordon/tree_mapping_feb_16/data/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'
        # mine2 = f'../../../tambe_lab/Users/luciagordon/tree_mapping_feb_15/data/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'

        mine_feb19 = f'../../../tambe_lab/Users/luciagordon/tree_mapping_feb19/data/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'
        mine_mergetifs = f'../../../tambe_lab/Users/luciagordon/tree_mapping_merge_tifs/data/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'
        mine_warp = f'../../../tambe_lab/Users/luciagordon/tree_mapping_warp/data/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'
        hers_feb15 = f'../../../tambe_lab/Everyone/Karingani_data/esther_generated_feb15/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'
        hers_mergetifs = f'../../../tambe_lab/Everyone/Karingani_data/esther_feb16_mergetifs_envvars/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'
        hers_warp = f'../../../tambe_lab/Everyone/Karingani_data/esther_feb16_warp/raw/lidar_by_site_32736/{site}_CHM_1m_merged.tif'

        # hers = f'../../../tambe_lab/Everyone/Karingani_data/esther_chm_merging_jan22_2024/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'
        # hers_old = f'../../../tambe_lab/Everyone/Karingani_data/esther_chm_merging_jan11_2024/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'
        # this = f'../../../tambe_lab/Users/luciagordon/tree-mapping-v1/data/raw/lidar_by_site_32736/KaringaniSite0{site}_CHM_1m_merged.tif'

        # mine = f'../../../tambe_lab/Users/luciagordon/tree_mapping_test/data/int/lidar/lidar_by_site_32736_10m/KaringaniSite0{site}/KaringaniSite0{site}_CHM_10m.tif'
        # hers = f'../../../tambe_lab/Everyone/Karingani_data/esther_lidar_outputs_dec_19/lidar_by_site_32736_10m/KaringaniSite0{site}/KaringaniSite0{site}_CHM_10m.tif'
        # this = f'../../../tambe_lab/Users/luciagordon/tree-mapping-v1/data/int/lidar/lidar_by_site_32736_10m/KaringaniSite0{site}/KaringaniSite0{site}_CHM_10m.tif'
        # print(rasterio.__version__) # 1.3.9

        my_10m_feb19_arr = tiff_to_array(my_10m_feb19)
        my_30m_feb19_arr = tiff_to_array(my_30m_feb19)

        mine_feb19_arr = tiff_to_array(mine_feb19)
        mine_mergetifs_arr = tiff_to_array(mine_mergetifs)
        mine_warp_arr = tiff_to_array(mine_warp)
        hers_feb15_arr = tiff_to_array(hers_feb15)
        hers_mergetifs_arr = tiff_to_array(hers_mergetifs)
        hers_warp_arr = tiff_to_array(hers_warp)

        my_10m_arr = tiff_to_array(my_10m)
        my_30m_arr = tiff_to_array(my_30m)

        # her_10m_arr = tiff_to_array(her_10m)
        # her_30m_arr = tiff_to_array(her_30m)


        # print((my_arr == my_arr2).all())
        print(site)
        print('mine 10m', (my_10m_feb19_arr == my_10m_arr).all())
        print('mine 30m', (my_30m_feb19_arr == my_30m_arr).all())
        print('my feb19 my warp', (mine_feb19_arr == mine_warp_arr).all())
        # print('my mergetifs my warp', (mine_mergetifs_arr == mine_warp_arr).all())
        # print('mine merge tiffs hers merge tiffs', (mine_mergetifs_arr == hers_mergetifs_arr).all())
        # print('mine hers warp', (mine_warp_arr == hers_warp_arr).all())
        # print('hers feb 15 hers merge tiffs', (hers_feb15_arr == hers_mergetifs_arr).all())
        # print('hers feb 15 hers warp', (hers_feb15_arr == hers_warp_arr).all())
        # print('hers mergetifs hers warp', (hers_mergetifs_arr == hers_warp_arr).all())
        # print('10m', (my_10m_arr == her_10m_arr).all())
        # print('30m', (my_30m_arr == her_30m_arr).all())

    my_eth = '../../../tambe_lab/Users/luciagordon/tree_mapping/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged.tif'
    her_eth = '../../../tambe_lab/Everyone/Karingani_data/ETH_GlobalCanopyHeight_10m_merged.tif'
    my_SAFR = '../../../tambe_lab/Users/luciagordon/tree_mapping/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif'
    her_SAFR = '../../../tambe_lab/Everyone/Karingani_data/Forest_height_2019_SAFR_cropped_2.tif'

    # her_arr = tiff_to_array(hers)
    # her_old_arr = tiff_to_array(hers_old)
    # this_arr = tiff_to_array(this)

    # my_eth_arr = tiff_to_array(my_eth)
    # her_eth_arr = tiff_to_array(her_eth)
    # my_safr_arr = tiff_to_array(my_SAFR)
    # her_safr_arr = tiff_to_array(her_SAFR)

    # print((my_eth_arr == her_eth_arr).all())
    # print((my_safr_arr == her_safr_arr).all())

    # print((her_arr == hers_today_arr).all())
    # print((my_arr == her_arr).all())
    # print((her_arr == her_old_arr).all())

    # print((this_arr == her_arr).all())
    # print((this_arr == my_arr).all())

def check_percentage_data_in_range():
    '''check what % of the 10m data is outside the 0-30 m height range'''
    lidar_10m_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_10m'
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

def plot_distance_to_feature():
    lidar_10m_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_10m'
    sites = os.listdir(lidar_10m_dir)
    site = sites[1]
    array = np.load(f'{get_project_dir()}/distances-to-river/{site}_distances_to_river.npy')

    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes        
    plt.savefig(f'site-{site}-distance-to-river.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

    site_raster = gdal.Open(f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif')
    site_num_rows = site_raster.RasterYSize
    site_num_cols = site_raster.RasterXSize
    site_pixel_height = site_raster.GetGeoTransform()[5] # m
    site_pixel_width = site_raster.GetGeoTransform()[1] # m
    site_top = site_raster.GetGeoTransform()[3] # m
    site_left = site_raster.GetGeoTransform()[0] # m
    site_bottom = site_top + site_pixel_height * site_num_rows # m
    site_right = site_left + site_pixel_width * site_num_cols # m

    feature_raster = gdal.Open(f'river_raster.tiff')
    FEATURE_NUM_ROWS = feature_raster.RasterYSize
    FEATURE_NUM_COLS = feature_raster.RasterXSize
    FEATURE_PIXEL_HEIGHT = feature_raster.GetGeoTransform()[5] # m
    FEATURE_PIXEL_WIDTH = feature_raster.GetGeoTransform()[1] # m
    FEATURE_TOP = feature_raster.GetGeoTransform()[3] # m
    FEATURE_LEFT = feature_raster.GetGeoTransform()[0] # m
    FEATURE_BOTTOM = FEATURE_TOP + FEATURE_PIXEL_HEIGHT * FEATURE_NUM_ROWS # m
    FEATURE_RIGHT = FEATURE_LEFT + FEATURE_PIXEL_WIDTH * FEATURE_NUM_COLS # m
    feature_array = ((feature_raster.GetRasterBand(1)).ReadAsArray(0, 0, FEATURE_NUM_COLS, FEATURE_NUM_ROWS).astype(np.float32)) # 0 = not feature, 255 = feature
    site_top_in_feature_array = int((site_top-FEATURE_TOP)/FEATURE_PIXEL_HEIGHT)
    site_bottom_in_feature_array = int(feature_array.shape[0] + (site_bottom-FEATURE_BOTTOM)/FEATURE_PIXEL_HEIGHT)
    site_left_in_feature_array = int((site_left-FEATURE_LEFT)/FEATURE_PIXEL_WIDTH)
    site_right_in_feature_array = int(feature_array.shape[1] + (site_right-FEATURE_RIGHT)/FEATURE_PIXEL_WIDTH)

    for i in range(site_top_in_feature_array, site_bottom_in_feature_array+1):
        for j in range(site_left_in_feature_array, site_right_in_feature_array+1):
            if (i == site_top_in_feature_array or i == site_bottom_in_feature_array) or (j == site_left_in_feature_array or j == site_right_in_feature_array):
                feature_array[i][j] = 255

    plt.figure(dpi=300)
    plt.imshow(feature_array)
    plt.axis('off') # remove axes        
    plt.savefig(f'river-raster-{site}-boxed.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

def get_max_distance_to_river():
    distances_to_river_dir = f'{get_project_dir()}/distances-to-river'
    max_distances = []

    for array in os.listdir(distances_to_river_dir):
        max_distances += [np.max(np.load(f'{distances_to_river_dir}/{array}'))]

    print(np.max(max_distances)) # 2854 m

if __name__ == '__main__':
    compare_tiffs()