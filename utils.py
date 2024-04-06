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
    tiff_1_array = tiff_to_array(tiff_1_path)
    tiff_2_array = tiff_to_array(tiff_2_path)

    return (tiff_1_array == tiff_2_array).all()

def check_percentage_data_in_range():
    '''Checks what % of the 1m data is inside the 0-30 m height range'''

    lidar_dir = f'{get_project_dir()}/data/raw/lidar_by_site_32736_merged'
    tiffs = os.listdir(lidar_dir)
    num_points_in_range = 0
    num_non_NaN_points = 0

    for tiff in tiffs:
        with rasterio.open(f'{lidar_dir}/{tiff}') as file:
            site_labels = file.read().ravel()
            num_points_in_range += len(site_labels[(site_labels >= 0) & (site_labels <= 30)])
            num_non_NaN_points += len(site_labels[site_labels != -9999.0])

    print(f'Percentage of points in range = {100*num_points_in_range/num_non_NaN_points}%')

def global_data_lidar_NaN_comparison(map, resolution):
    '''Checks what % of the non-NaN LiDAR pixels are NaNs in the global maps'''

    lidar_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    global_map_dir = f'{get_project_dir()}/data/existing_reference_data/{map}_maps_per_site_{resolution}m'
    sites = os.listdir(lidar_dir)
    num_NaN_preds_for_non_NaN_labels = 0
    num_non_NaN_labels = 0

    for site in sites:
        with rasterio.open(f'{lidar_dir}/{site}/{site}_CHM_{resolution}m.tif') as lidar_file:
            site_labels = lidar_file.read().ravel()
        
        with rasterio.open(f'{global_map_dir}/{map.upper()}_MAP_{site}_{resolution}m.tif') as preds_file:
            site_predictions = preds_file.read().ravel()

        num_NaN_preds_for_non_NaN_labels += len([i for i in range(len(site_labels)) if site_labels[i] != -9999.0 and site_predictions[i] == -9999.0])
        num_non_NaN_labels += len(site_labels[site_labels != -9999.0])

    print(f'Percentage of non-NaN labels that have NaN predictions for {map} at {resolution}m-resolution = {100*num_NaN_preds_for_non_NaN_labels/num_non_NaN_labels}%')

def plot_distance_to_feature(feature, resolution):
    '''Plots distance to feature and puts box around a site'''
    lidar_dir = f'{get_project_dir()}/data/int/lidar/lidar_by_site_32736_{resolution}m'
    sites = os.listdir(lidar_dir)
    site = sites[2]
    array = np.load(f'{get_project_dir()}/data/features/{feature}/distances_to_{feature}_{resolution}m/{site}_distances_to_{feature}_{resolution}m.npy')

    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes        
    plt.savefig(f'figures/{feature}/{site}_distance_to_{feature}_{resolution}m.png', bbox_inches='tight', pad_inches=0)
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
    plt.savefig(f'figures/{feature}/{feature}_raster_{resolution}m_{site}_boxed.png', bbox_inches='tight', pad_inches=0)
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

def plot_tiff(path):
    array = tiff_to_array(path)

    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes        
    plt.savefig(f'{path.split("/")[-1].split(".")[0]}.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

def find_large_values():
    lidar_1m_merged_dir = f'{get_project_dir()}/data/raw/lidar_by_site_32736_merged'
    lidar_1m_merged_tiffs = os.listdir(lidar_1m_merged_dir)
    box_size = 100

    for tiff in lidar_1m_merged_tiffs:
        tiff_array = rasterio.open(f'{lidar_1m_merged_dir}/{tiff}').read(1)
        where_above_30 = np.array(np.argwhere(tiff_array > 30))
        print(f'{tiff}, number of pixels above 30 = {len(where_above_30)}')

        if len(where_above_30) > 0:
            for row, col in where_above_30:
                    print(tiff_array[row, col])
                    for i in range(row-box_size, row+box_size+1):
                        for j in range(col-box_size, col+box_size+1):
                            if (i == row-box_size or i == row+box_size) or (j == col-box_size or j == col+box_size):
                                tiff_array[i][j] = 0
        
            plt.figure(dpi=300)
            plt.imshow(tiff_array)
            plt.set_cmap('inferno')
            plt.axis('off') # remove axes        
            plt.savefig(f'{tiff.split(".")[0]}_large_vals_boxed.png', bbox_inches='tight', pad_inches=0)
            plt.close() # close the image to save memory

def get_max_height_near_river():
    distances_to_river_10m_dir = '../../../tambe_lab/Users/luciagordon/tree_mapping/data/features/river/distances_to_river_10m'
    sites = [array_name.split('_')[0] for array_name in os.listdir(distances_to_river_10m_dir)]
    all_values_near_river = []

    for site in sites:
        coords_near_river = np.argwhere(np.load(f'{distances_to_river_10m_dir}/{site}_distances_to_river_10m.npy') <= 500) # distance to river = n x 10
        values_near_river = tiff_to_array(f'../../../tambe_lab/Users/luciagordon/tree_mapping/data/int/lidar/lidar_by_site_32736_10m/{site}/{site}_CHM_10m.tif')[coords_near_river[:, 0], coords_near_river[:, 1]]
        all_values_near_river += list(values_near_river)
    
    print(max(all_values_near_river), len(np.argwhere(np.array(all_values_near_river) > 30)))

def str_to_bool(string):
    '''convert a string input to a Boolean variable'''
    if string.lower() in ('yes', 'true', 't', 'y', '1'):
        return True

    return False

if __name__ == '__main__':
    for resolution in [10,30]:
        global_data_lidar_NaN_comparison(map='eth', resolution=resolution)
        global_data_lidar_NaN_comparison(map='glad', resolution=resolution)
