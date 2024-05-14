# imports
from osgeo import gdal
from utils import get_project_dir
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio

gdal.UseExceptions()
data_dir = f'{get_project_dir()}/data'

class FeatureBySite:
    def __init__(self, resolution, feature):
        self.resolution = resolution
        self.feature = feature
        self.lidar_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_{resolution}m'
        sites = sorted([file.split('_')[0] for file in os.listdir(self.lidar_dir)])

        # process feature raster
        with rasterio.open(f'{data_dir}/features/{feature}/{feature}_raster_{resolution}m.tif') as feature_raster:
            self.feature_transform = feature_raster.transform
            self.feature_array = feature_raster.read(1)

            print(f'{feature} raster shape = {self.feature_array.shape}')

        if feature == 'river':
            self.feature_boolean_array = np.argwhere(self.feature_array == 255)
            
            for site in sites:
                self.get_site_distances_to_feature(site)
        elif feature == 'geology':
            for site in sites:
                print(site)
                self.get_site_feature_values(site)

    def process_site_raster(self, site):
        with rasterio.open(f'{self.lidar_dir}/{site}/{site}_CHM_{self.resolution}m.tif') as site_raster:
            site_left, site_bottom, site_right, site_top = site_raster.bounds        
            site_top_in_feature_array, site_left_in_feature_array = rasterio.transform.rowcol(self.feature_transform, site_left, site_top)
            site_bottom_in_feature_array, site_right_in_feature_array = rasterio.transform.rowcol(self.feature_transform, site_right, site_bottom)

        return site_top_in_feature_array, site_bottom_in_feature_array, site_left_in_feature_array, site_right_in_feature_array

    def get_distance_to_feature(self, index):
        distances_to_features = np.sqrt((self.feature_boolean_array[:, 0] - index[0])**2 + (self.feature_boolean_array[:, 1] - index[1])**2) # distance from point to every feature pixel
        min_distance = self.resolution * np.min(distances_to_features) # meters
        
        return min_distance

    def get_site_distances_to_feature(self, site):
        site_top_in_feature_array, site_bottom_in_feature_array, site_left_in_feature_array, site_right_in_feature_array = self.process_site_raster(site) # gets bounds of site in terms of rows and columns in the feature array
        site_distances_to_feature = np.array([[self.get_distance_to_feature([row, col]) for col in range(site_left_in_feature_array, site_right_in_feature_array)] for row in range(site_top_in_feature_array, site_bottom_in_feature_array)]) # loops over all the points in the site
        assert len(site_distances_to_feature.ravel()) == (site_bottom_in_feature_array-site_top_in_feature_array) * (site_right_in_feature_array-site_left_in_feature_array) # makes sure we have a distance value for every point in the site
        os.makedirs(f'{data_dir}/features/{self.feature}/distances_to_{self.feature}_{self.resolution}m', exist_ok=True)
        np.save(f'{data_dir}/features/{self.feature}/distances_to_{self.feature}_{self.resolution}m/{site}_distances_to_{self.feature}_{self.resolution}m', site_distances_to_feature)

    def get_site_feature_values(self, site):
        site_top_in_feature_array, site_bottom_in_feature_array, site_left_in_feature_array, site_right_in_feature_array = self.process_site_raster(site) # gets bounds of site in terms of rows and columns in the feature array
        print(site_top_in_feature_array, site_bottom_in_feature_array, site_left_in_feature_array, site_right_in_feature_array)

        feature_array = self.feature_array.copy()

        if site_top_in_feature_array < 0:
            feature_array = np.vstack((np.zeros((-site_top_in_feature_array, feature_array.shape[1])), feature_array))
            print(feature_array.shape)
            site_bottom_in_feature_array -= site_top_in_feature_array
            site_top_in_feature_array -= site_top_in_feature_array

        print(site_top_in_feature_array, site_bottom_in_feature_array, site_left_in_feature_array, site_right_in_feature_array)
        site_feature_values = feature_array[site_top_in_feature_array : site_bottom_in_feature_array, site_left_in_feature_array : site_right_in_feature_array]
        print(site_feature_values.shape)
        assert len(site_feature_values.ravel()) == (site_bottom_in_feature_array-site_top_in_feature_array) * (site_right_in_feature_array-site_left_in_feature_array)
        os.makedirs(f'{data_dir}/features/{self.feature}/{self.feature}_{self.resolution}m', exist_ok=True)
        np.save(f'{data_dir}/features/{self.feature}/{self.feature}_{self.resolution}m/{site}_{self.feature}_{self.resolution}m', site_feature_values)

if __name__ == '__main__':
    FeatureBySite(resolution=10, feature='river')
    FeatureBySite(resolution=30, feature='river')
    FeatureBySite(resolution=10, feature='geology')
    FeatureBySite(resolution=30, feature='geology')
