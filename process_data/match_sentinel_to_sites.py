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

lidar_10m_dir = f'{project_dir}/data/int/lidar/lidar_by_site_32736_10m' # where to read data from
sentinel_by_site_dir = f'{project_dir}/data/int/sentinel/sentinel_by_site_32736_10m'

def crop_sentinel_to_karingani_data(buffer=40):
    # match sentinel to chm files
    sites = os.listdir(lidar_10m_dir)

    for site in sites:
        print(site)
        os.makedirs(f'{sentinel_by_site_dir}/{site}', exist_ok=True)

        sentinel_tile_dir = sentinel_dirs[sentinel_tiles_per_site[site]]
        lidar_10m_site_tiff = f'{lidar_10m_dir}/{site}/{site}_CHM_10m.tif'

        for band_tiff in os.listdir(sentinel_tile_dir):
            band_tiff_path = f'{sentinel_tile_dir}/{band_tiff}'
            cropped_band_tiff_path = f'{sentinel_by_site_dir}/{site}/{band_tiff}'

            match_input_to_target_tif(input_fn=band_tiff_path, 
                                      output_fn=cropped_band_tiff_path, 
                                      target_fn=lidar_10m_site_tiff, 
                                      resampling='near',
                                      pixel_buffer=buffer,
                                      output_type='int16',
                                      verbose=True)

# def process_alos_to_sentinel_site_data(sentinel_by_site_dir, 
#                                        chunked_alos_dir, 
#                                        merged_alos_fn,
#                                        buffer=0,
#                                        verbose=True):
    
#     # match sentinel to chm files
#     target_sites = os.listdir(sentinel_by_site_dir)
    
#     if not os.path.exists(f'{chunked_alos_dir}'):
#         os.mkdir(f'{chunked_alos_dir}')
            
#     for site in target_sites:
#         this_dir = os.path.join(sentinel_by_site_dir, site)
#         target_fn = [x for x in os.listdir(this_dir) if x.endswith('B02_10m.tif')][0]
  
#         if verbose: print(site)
#         if not os.path.exists(f'{chunked_alos_dir}/{site}'):
#             os.mkdir(f'{chunked_alos_dir}/{site}')
            
#         input_fp = merged_alos_fn
#         output_fp = f'{chunked_alos_dir}/{site}/{merged_alos_fn.split('/')[-1]}'
#         target_fp = os.path.join(this_dir,target_fn)

#         match_input_to_target_tif(input_fp, 
#                                       output_fp, 
#                                       target_fp, 
#                                       resampling='near',
#                                       pixel_buffer = buffer,
#                                       output_type='int16',
#                                       verbose=verbose)

if __name__  == '__main__':           
    crop_sentinel_to_karingani_data()

    # chunked_alos_dir = os.path.join(DATA_DIR, 'int/alos/alos_by_site_20_FNF')
    # if not os.path.exists(os.path.join(DATA_DIR, 'int/alos')): os.mkdir(os.path.join(DATA_DIR, 'int/alos'))
    # merged_alos_fn = os.path.join(DATA_DIR, 'raw/alos/Karingani_merged_20_FNF/Karingani_merged_20_C.tif')
    # process_alos_to_sentinel_site_data(sentinel_by_site_dir, chunked_alos_dir, merged_alos_fn)
