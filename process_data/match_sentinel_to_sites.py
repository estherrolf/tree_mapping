# imports
import os
import sys
from geo_utils import match_input_to_target_tif

sys.path.insert(0, '') # necessary since utils is outside the process_data folder
from utils import get_project_dir

project_dir = get_project_dir()
sentinel_dirs = [f'{project_dir}/data/raw/sentinel_2021/S2B_MSIL2A_20210513T073609_R092_T36KVU_20210606T053436',
                 f'{project_dir}/data/raw/sentinel_2021/S2B_MSIL2A_20210513T073609_R092_T36KUU_20210514T122203',
                 f'{project_dir}/data/raw/sentinel_2021/S2B_MSIL2A_20210513T073609_R092_T36JUT_20210514T161910']

sentinel_tiles_per_site = {'KaringaniMassingirDevNode':1,
                           'KaringaniSite01':1,
                           'KaringaniSite02':1,
                           'KaringaniSite03':0,
                           'KaringaniSite04':1,
                           'KaringaniSite05':1,
                           'KaringaniSite06':1,
                           'KaringaniSite07DevNodeB':0,
                           'KaringaniSite08':1,
                           'KaringaniSite09':0,
                           'KaringaniSite10':0,
                           'KaringaniSite11DevNodeF':0,
                           'KaringaniSite12':0,
                           'KaringaniSite13':0,
                           'KaringaniSite14':1,
                           'KaringaniSite15':0,
                           'KaringaniSite16':1,
                           'KaringaniSite17':2,
                           'KaringaniSite18':2,
                           'KaringaniVultureSite01':0,
                           'KaringaniSungoloDevNode':1,
                           'KaringaniSouthSouthDevNode':0,
                           'KaringaniSouthNorthDevNode':0,
                           'Mbilu':0}

def crop_sentinel_to_karingani_data(resolution, buffer=40):
    '''Matches sentinel to CHM files'''

    lidar_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_{resolution}m' # where to read data from
    sentinel_by_site_dir = f'{project_dir}/data/int/sentinel/sentinel_by_site_32736_{resolution}m'
    sites = os.listdir(lidar_dir)

    for site in sites:
        print(site)
        os.makedirs(f'{sentinel_by_site_dir}/{site}', exist_ok=True)

        sentinel_tile_dir = sentinel_dirs[sentinel_tiles_per_site[site]]
        lidar_site_tiff = f'{lidar_dir}/{site}/{site}_CHM_{resolution}m.tif'

        for band_tiff in os.listdir(sentinel_tile_dir):
            band_tiff_path = f'{sentinel_tile_dir}/{band_tiff}'
            tiff_name_updated_resolution = '_'.join(band_tiff.split('_')[:-1]) + f'_{resolution}m.tif'
            cropped_band_tiff_path = f'{sentinel_by_site_dir}/{site}/{tiff_name_updated_resolution}'

            match_input_to_target_tif(input_fn=band_tiff_path,
                                      output_fn=cropped_band_tiff_path,
                                      target_fn=lidar_site_tiff,
                                      resampling='cubic',
                                      pixel_buffer=buffer,
                                      output_type='int16',
                                      verbose=False)

if __name__  == '__main__':
    crop_sentinel_to_karingani_data(resolution=10)
    crop_sentinel_to_karingani_data(resolution=30)
