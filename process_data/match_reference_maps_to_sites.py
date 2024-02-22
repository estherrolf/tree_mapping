# imports
from osgeo import gdal
import geo_utils
import os
import matplotlib.pyplot as plt
import rasterio
import sys

sys.path.insert(0, '') # necessary since utils is outside the process_data folder
from utils import get_project_dir

class CropReferenceMaps:
    def __init__(self, resolution):
        self.project_dir = get_project_dir()
        self.resolution = resolution

        self.make_per_site_files(map='eth', map_tiff=self.merge_ETH_maps())
        self.make_per_site_files(map='glad', map_tiff=self.crop_GLAD_map())

    def merge_ETH_maps(self):
        '''Merge the two ETH data files into one'''
        eth_tiles = [f'{self.project_dir}/data/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S24E030_Map.tif',
                     f'{self.project_dir}/data/raw/global_tch_maps/eth/ETH_GlobalCanopyHeight_10m_2020_S27E030_Map.tif']
        global_map_dir = f'{self.project_dir}/data/int/global_tch_maps'
        os.makedirs(global_map_dir, exist_ok=True)
        merged_eth_tiff = f'{global_map_dir}/ETH_GlobalCanopyHeight_10m_merged.tif'
        gdal.Warp(destNameOrDestDS=merged_eth_tiff, srcDSOrSrcDSTab=eth_tiles, srcNodata=255, dstNodata=255, format='GTiff', outputType=gdal.gdalconst.GDT_Byte)

        return merged_eth_tiff

    def crop_GLAD_map(self):
        '''Crop the GLAD map to the extent of the merged LiDAR imagery'''
        lidar_dir = f'{self.project_dir}/data/raw/lidar_by_site_32736'
        input_lidar_sites = [f'{lidar_dir}/{x}' for x in os.listdir(lidar_dir) if x.endswith('.tif')]
        lidar_sites_merged = f'{self.project_dir}/lidar_sites_merged.tif'
        gdal.Warp(destNameOrDestDS=lidar_sites_merged, srcDSOrSrcDSTab=input_lidar_sites, format='GTiff', srcNodata=255.0, dstNodata=255.0, outputType=gdal.gdalconst.GDT_Float32)

        glad_tiff = f'{self.project_dir}/data/raw/global_tch_maps/Forest_height_2019_SAFR.tif'
        cropped_glad_tiff = f'{self.project_dir}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped_no_data_103.tif'

        geo_utils.crop_input_to_target_tif(input_fn=glad_tiff,
                                           output_fn=cropped_glad_tiff,
                                           target_fn=lidar_sites_merged,
                                           output_type='Byte', # same as input
                                           pixel_buffer=40,
                                           output_nodata='255')
        os.remove(lidar_sites_merged)

        with rasterio.open(cropped_glad_tiff) as file:
            metadata = file.meta
            tiff_array = file.read()
            tiff_array[tiff_array > 100] = 255

            with rasterio.open(f'{self.project_dir}/data/raw/global_tch_maps/Forest_height_2019_SAFR_cropped.tif', 'w', **metadata) as out_file:
                out_file.write(tiff_array)

        return cropped_glad_tiff

    def make_per_site_files(self, map, map_tiff):
        '''Make per-site files for the ETH and GLAD maps to match the coarsened LiDAR data'''
        ref_data_dir = f'{self.project_dir}/data/existing_reference_data'
        global_map_by_site_dir = f'{ref_data_dir}/{map}_maps_per_site_{self.resolution}m'
        os.makedirs(global_map_by_site_dir, exist_ok=True)
        lidar_coarsened_dir = f'{self.project_dir}/data/int/lidar/lidar_by_site_32736_{self.resolution}m'

        for site in os.listdir(lidar_coarsened_dir):
            global_map_site_tiff = f'{global_map_by_site_dir}/{map.upper()}_MAP_{site}_{self.resolution}m.tif'
            lidar_site_tiff = f'{lidar_coarsened_dir}/{site}/{site}_CHM_{self.resolution}m.tif'

            geo_utils.match_input_to_target_tif(input_fn=map_tiff,
                                                output_fn=global_map_site_tiff,
                                                target_fn=lidar_site_tiff,
                                                pixel_buffer=0,
                                                verbose=False,
                                                output_type='Float32',
                                                resampling='average',
                                                src_nodata='255',
                                                output_nodata='255.0')

        # with rasterio.open(global_map_site_tiff) as file:
        #     data = file.read()

        # plt.figure(dpi=300)
        # plt.hist(data[data != -9999.].ravel())
        # plt.savefig(f'{self.project_dir}/{map}-histogram.png')

if __name__ == '__main__':
    CropReferenceMaps(resolution=10)
    CropReferenceMaps(resolution=30)
