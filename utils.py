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

def compare_tiffs(tiff_1_path, tiff_2_path):
    tiff_1_array = tiff_to_array(tiff_1_path).ravel()
    tiff_2_array = tiff_to_array(tiff_2_path).ravel()
    # for i in range(len(tiff_1_array)):
    #     if tiff_1_array[i] != tiff_2_array[i]:
    #         print(i, tiff_1_array[i], tiff_2_array[i])
    return (tiff_1_array == tiff_2_array).all()

# if __name__ == '__main__':
    # print(compare_tiffs(f'{get_project_dir()}/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged_Float32.tif', f'{get_project_dir()}/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged.tif'))
    # for tiff in os.listdir(f'{get_project_dir()}/data/existing_reference_data/eth_maps_per_site_10m'):
    #     print(compare_tiffs(f'{get_project_dir()}/data/existing_reference_data/eth_maps_per_site_10m-byte/{tiff}', f'{get_project_dir()}/data/existing_reference_data/eth_maps_per_site_10m/{tiff}'))
    # for tiff in os.listdir(f'{get_project_dir()}/data/existing_reference_data/glad_maps_per_site_10m'):
    #     print(compare_tiffs(f'{get_project_dir()}/data/existing_reference_data/glad_maps_per_site_10m-byte/{tiff}', f'{get_project_dir()}/data/existing_reference_data/glad_maps_per_site_10m/{tiff}'))

    # print(compare_tiffs(f'{get_project_dir()}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped_warp.tif', f'{get_project_dir()}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif'))
    # print(compare_tiffs(f'{get_project_dir()}/lidar_sites_merged_warp.tif', f'{get_project_dir()}/lidar_sites_merged.tif'))
    # print(compare_tiffs(f'{get_project_dir()}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped_float32.tif', f'{get_project_dir()}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif'))
    # print(np.max(tiff_to_array(f'{get_project_dir()}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped_no-r.tif')))
    # for tiff in os.listdir(f'{get_project_dir()}/data/raw/lidar_by_site_32736_merged'):
    #     print(compare_tiffs(f'{get_project_dir()}/data/raw/lidar_by_site_32736_merged/{tiff}', f'{get_project_dir()}/data/raw/lidar_by_site_32736_merged_noquotes/{tiff}'))
    # print(compare_tiffs(f'{get_project_dir()}/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged-warp.tif', f'{get_project_dir()}/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged-og.tif'))

def compare():
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

        # my_10m_feb19_arr = tiff_to_array(my_10m_feb19)
        # my_30m_feb19_arr = tiff_to_array(my_30m_feb19)

        # mine_feb19_arr = tiff_to_array(mine_feb19)
        # mine_mergetifs_arr = tiff_to_array(mine_mergetifs)
        # mine_warp_arr = tiff_to_array(mine_warp)
        # hers_feb15_arr = tiff_to_array(hers_feb15)
        # hers_mergetifs_arr = tiff_to_array(hers_mergetifs)
        # hers_warp_arr = tiff_to_array(hers_warp)

        # my_10m_arr = tiff_to_array(my_10m)
        # my_30m_arr = tiff_to_array(my_30m)

        # her_10m_arr = tiff_to_array(her_10m)
        # her_30m_arr = tiff_to_array(her_30m)


        # print((my_arr == my_arr2).all())
        # print(site)
        # print('mine 10m', (my_10m_feb19_arr == my_10m_arr).all())
        # print('mine 30m', (my_30m_feb19_arr == my_30m_arr).all())
        # print('my feb19 my warp', (mine_feb19_arr == mine_warp_arr).all())
        # print('my mergetifs my warp', (mine_mergetifs_arr == mine_warp_arr).all())
        # print('mine merge tiffs hers merge tiffs', (mine_mergetifs_arr == hers_mergetifs_arr).all())
        # print('mine hers warp', (mine_warp_arr == hers_warp_arr).all())
        # print('hers feb 15 hers merge tiffs', (hers_feb15_arr == hers_mergetifs_arr).all())
        # print('hers feb 15 hers warp', (hers_feb15_arr == hers_warp_arr).all())
        # print('hers mergetifs hers warp', (hers_mergetifs_arr == hers_warp_arr).all())
        # print('10m', (my_10m_arr == her_10m_arr).all())
        # print('30m', (my_30m_arr == her_30m_arr).all())
    my_eth_warp = '../../../tambe_lab/Users/luciagordon/tree_mapping/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged-2.tif'
    my_eth = '../../../tambe_lab/Users/luciagordon/tree_mapping/data/int/global_tch_maps/ETH_GlobalCanopyHeight_10m_merged.tif'
    her_eth = '../../../tambe_lab/Everyone/Karingani_data/ETH_GlobalCanopyHeight_10m_merged.tif'
    my_SAFR = '../../../tambe_lab/Users/luciagordon/tree_mapping/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif'
    her_SAFR = '../../../tambe_lab/Everyone/Karingani_data/Forest_height_2019_SAFR_cropped_2.tif'

    # her_arr = tiff_to_array(hers)
    # her_old_arr = tiff_to_array(hers_old)
    # this_arr = tiff_to_array(this)

    my_eth_warp_arr = tiff_to_array(my_eth_warp)
    my_eth_arr = tiff_to_array(my_eth)
    # her_eth_arr = tiff_to_array(her_eth)
    # my_safr_arr = tiff_to_array(my_SAFR)
    # her_safr_arr = tiff_to_array(her_SAFR)

    print((my_eth_arr == my_eth_warp_arr).all())
    # print((my_eth_arr == her_eth_arr).all())
    # print((my_safr_arr == her_safr_arr).all())

    # print((her_arr == hers_today_arr).all())
    # print((my_arr == her_arr).all())
    # print((her_arr == her_old_arr).all())

    # print((this_arr == her_arr).all())
    # print((this_arr == my_arr).all())

def check_percentage_data_in_range(resolution):
    '''check what % of the 10m data is outside the 0-30 m height range'''
    lidar_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_10m'
    sites = os.listdir(lidar_dir)
    num_points_in_range = 0
    total_points = 0

    for site in sites:
        with rasterio.open(f'{lidar_dir}/{site}/{site}_CHM_10m.tif') as file:
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

def plot_distance_to_feature(feature, resolution):
    lidar_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    sites = os.listdir(lidar_dir)
    site = sites[2]
    array = np.load(f'{get_project_dir()}/data/features/{feature}/distances_to_{feature}_{resolution}m/{site}_distances_to_{feature}_{resolution}m.npy')

    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes        
    plt.savefig(f'{site}_distance_to_{feature}_{resolution}m.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

    feature_raster = rasterio.open(f'{get_project_dir()}/data/features/{feature}/{feature}_raster_{resolution}m.tif')
    feature_left, feature_bottom, feature_right, feature_top = feature_raster.bounds
    feature_array = feature_raster.read(1)

    site_left, site_bottom, site_right, site_top = rasterio.open(f'{lidar_dir}/{site}/{site}_CHM_{resolution}m.tif').bounds # site bounds
    site_top_in_feature_array = int((feature_top - site_top) / resolution)
    site_bottom_in_feature_array = int(feature_array.shape[0] + (feature_bottom - site_bottom) / resolution)
    site_left_in_feature_array = int((site_left - feature_left) / resolution)
    site_right_in_feature_array = int(feature_array.shape[1] + (site_right - feature_right) / resolution)

    for i in range(site_top_in_feature_array, site_bottom_in_feature_array+1):
        for j in range(site_left_in_feature_array, site_right_in_feature_array+1):
            if (i == site_top_in_feature_array or i == site_bottom_in_feature_array) or (j == site_left_in_feature_array or j == site_right_in_feature_array):
                feature_array[i][j] = 255

    plt.figure(dpi=300)
    plt.imshow(feature_array)
    plt.axis('off') # remove axes        
    plt.savefig(f'{feature}_raster_{resolution}m_{site}_boxed.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

def get_max_distance_to_river():
    distances_to_river_dir = f'{get_project_dir()}/distances-to-river'
    max_distances = []

    for array in os.listdir(distances_to_river_dir):
        max_distances += [np.max(np.load(f'{distances_to_river_dir}/{array}'))]

    print(np.max(max_distances)) # 2854 m

def get_CHM_values(resolution):
    if resolution == 1:
        lidar_dir = f'{get_project_dir()}/data/raw/lidar_by_site_32736'
    else:
        lidar_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    
    lidar_dir_items = os.listdir(lidar_dir)
    values = []

    for item in lidar_dir_items:
        if resolution == 1:
            values += rasterio.open(f'{lidar_dir}/{item}').read().ravel().tolist()
        else:
            values += rasterio.open(f'{lidar_dir}/{item}/{item}_CHM_{resolution}m.tif').read().ravel().tolist()
    
    np.save(f'{get_project_dir()}/data/CHM_values_{resolution}m', values)

    values = list(filter(lambda x: x != -9999.0, values))
    np.save(f'{get_project_dir()}/data/CHM_values_excluding_NaN_{resolution}m', values)

def histogram(resolution):
    values = np.load(f'{get_project_dir()}/data/CHM_values_excluding_NaN_{resolution}m.npy')

    plt.figure(dpi=300)
    counts, bins, _ = plt.hist(values, bins=range(0, 31, 1))
    print(bins)
    plt.title(f'{resolution}m Resolution')
    plt.xlabel('Tree Canopy Height (m)')
    plt.ylabel('Number of Pixels')
    plt.savefig(f'histogram_{resolution}m.png', bbox_inches='tight', pad_inches=0.1)
    plt.close() # close the image to save memory

if __name__ == '__main__':
    # plot_distance_to_feature(feature='river', resolution=10)
    # get_CHM_values(resolution=30)
    plot_distance_to_feature(feature='river', resolution=30)
    # histogram(resolution=30)