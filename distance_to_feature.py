# imports
from osgeo import gdal
from utils import get_project_dir
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio

gdal.UseExceptions()

class DistanceToFeature:
    def __init__(self, feature, resolution):
        self.data_dir = f'{get_project_dir()}/data'
        self.feature = feature
        self.lidar_dir = f'{self.data_dir}/int/lidar/lidar_by_site_32736_{resolution}m'
        self.RESOLUTION = resolution

        # process feature raster
        feature_raster = rasterio.open(f'{self.data_dir}/features/{feature}/{feature}_raster_{resolution}m.tif')
        self.FEATURE_LEFT, self.FEATURE_BOTTOM, self.FEATURE_RIGHT, self.FEATURE_TOP = feature_raster.bounds
        self.feature_array = feature_raster.read(1)
        self.feature_boolean_array = np.argwhere(self.feature_array == 255)

        print(f'{feature} bounds: top = {self.FEATURE_TOP}, bottom = {self.FEATURE_BOTTOM}, left = {self.FEATURE_LEFT}, right = {self.FEATURE_RIGHT}')
        print(f'{feature} raster shape = {self.feature_array.shape}')

        os.makedirs(f'{self.data_dir}/features/{feature}/distances_to_{feature}_{resolution}m', exist_ok=True)
        sites = sorted([file.split('_')[0] for file in os.listdir(self.lidar_dir)])

        for site in sites:
            self.get_site_distances_to_feature(site)

    def get_distance_to_feature(self, index):
        distances_to_features = np.sqrt((self.feature_boolean_array[:, 0] - index[0])**2 + (self.feature_boolean_array[:, 1] - index[1])**2)
        min_distance = self.RESOLUTION * np.min(distances_to_features) # m
        
        return min_distance

    def get_site_distances_to_feature(self, site):
        site_left, site_bottom, site_right, site_top = rasterio.open(f'{self.lidar_dir}/{site}/{site}_CHM_{self.RESOLUTION}m.tif').bounds # site bounds
        
        site_top_in_feature_array = int((self.FEATURE_TOP - site_top) / self.RESOLUTION)
        site_bottom_in_feature_array = int(self.feature_array.shape[0] + (self.FEATURE_BOTTOM - site_bottom) / self.RESOLUTION)
        site_left_in_feature_array = int((site_left - self.FEATURE_LEFT) / self.RESOLUTION)
        site_right_in_feature_array = int(self.feature_array.shape[1] + (site_right - self.FEATURE_RIGHT) / self.RESOLUTION)

        site_distances_to_feature = np.array([[self.get_distance_to_feature([row, col]) for col in range(site_left_in_feature_array, site_right_in_feature_array)] for row in range(site_top_in_feature_array, site_bottom_in_feature_array)])
        print(f'{site}: site distances to feature shape = site shape: {site_distances_to_feature.shape == rasterio.open(f"{self.lidar_dir}/{site}/{site}_CHM_{self.RESOLUTION}m.tif").read(1).shape}')
        np.save(f'{self.data_dir}/features/{self.feature}/distances_to_{self.feature}_{self.RESOLUTION}m/{site}_distances_to_{self.feature}_{self.RESOLUTION}m', site_distances_to_feature)

if __name__ == '__main__':
    DistanceToFeature(feature='river', resolution=10)
    DistanceToFeature(feature='river', resolution=30)
