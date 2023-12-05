import os
import rasterio
import subprocess
from geo_utils import match_input_to_target_tif

DATA_DIR = '../data'     
# DATA_DIR = '../../../tambe_lab/Users/luciagordon/tree_mapping_lucia_branch/data'   
sentinel_dirs = ['none',
                 f'{DATA_DIR}/raw/sentinel_2021/S2B_MSIL2A_20210513T073609_R092_T36KVU_20210606T053436',
                 f'{DATA_DIR}/raw/sentinel_2021/S2B_MSIL2A_20210513T073609_R092_T36KUU_20210514T122203',
                 f'{DATA_DIR}/raw/sentinel_2021/S2B_MSIL2A_20210513T073609_R092_T36JUT_20210514T161910'
                ]

s2_tiles_per_site = {
    'KaringaniMassingirDevNode':2,
    'KaringaniSite01':2,
    'KaringaniSite02':2,
    'KaringaniSite03':1,
    'KaringaniSite04':2,
    'KaringaniSite05':2,
    'KaringaniSite06':2,
    'KaringaniSite07DevNodeB':1,
    'KaringaniSite08':2,
    'KaringaniSite09':1,
    'KaringaniSite10':1,
    'KaringaniSite11DevNodeF':1,
    'KaringaniSite12':1,
    'KaringaniSite13':1,
    'KaringaniSite14':2,
    'KaringaniSite15':1,
    'KaringaniSite16':2,
    'KaringaniSite17':3,
    'KaringaniSite18':3,
    'KaringaniVultureSite01':1,
    'KaringaniSungoloDevNode':2,
    'KaringaniSouthSouthDevNode':1,
    'KaringaniSouthNorthDevNode':1,
    'Mbilu': 1,  
}

def process_sentinel_to_Karingani_data(chm_dir, chunked_sentinel_dir, buffer=40):
    # match sentinel to chm files
    site_ids = os.listdir(chm_dir)
    
    if not os.path.exists(f"{chunked_sentinel_dir}"):
        os.mkdir(f"{chunked_sentinel_dir}")
            
    for site_id in site_ids:
        sentinel_dir = sentinel_dirs[s2_tiles_per_site[site_id]]
        chm_dir_this = os.path.join(chm_dir, site_id)
        target_fn = [x for x in os.listdir(chm_dir_this) if x.endswith('.tif')][0]
        target_fp = os.path.join(chm_dir_this, target_fn)

        print(site_id)
        if not os.path.exists(f"{chunked_sentinel_dir}/{site_id}"):
            os.mkdir(f"{chunked_sentinel_dir}/{site_id}")
        for s2_fn in list(os.listdir(sentinel_dir)):
            if not s2_fn.endswith('.tif'): continue
            input_fp = os.path.join(sentinel_dir, s2_fn)
            output_fp = f"{chunked_sentinel_dir}/{site_id}/{s2_fn}"
            match_input_to_target_tif(input_fp, 
                                      output_fp, 
                                      target_fp, 
                                      resampling="near",
                                      pixel_buffer = buffer,
                                      output_type="int16",
                                      verbose=True)
            
def process_alos_to_sentinel_site_data(chunked_sentinel_dir, 
                                       chunked_alos_dir, 
                                       merged_alos_fn,
                                       buffer=0,
                                       verbose=True):
    
    # match sentinel to chm files
    target_sites = os.listdir(chunked_sentinel_dir)
    
    if not os.path.exists(f"{chunked_alos_dir}"):
        os.mkdir(f"{chunked_alos_dir}")
            
    for site_id in target_sites:
        this_dir = os.path.join(chunked_sentinel_dir, site_id)
        target_fn = [x for x in os.listdir(this_dir) if x.endswith('B02_10m.tif')][0]
  
        if verbose: print(site_id)
        if not os.path.exists(f"{chunked_alos_dir}/{site_id}"):
            os.mkdir(f"{chunked_alos_dir}/{site_id}")
            
        input_fp = merged_alos_fn
        output_fp = f"{chunked_alos_dir}/{site_id}/{merged_alos_fn.split('/')[-1]}"
        target_fp = os.path.join(this_dir,target_fn)

        match_input_to_target_tif(input_fp, 
                                      output_fp, 
                                      target_fp, 
                                      resampling="near",
                                      pixel_buffer = buffer,
                                      output_type="int16",
                                      verbose=verbose)
            
            
if __name__  == "__main__":           
    data_dir = DATA_DIR
    # where to read data from
    chm_dir = os.path.join(data_dir, 'int/lidar/lidar_by_site_32736_10m')
    print(chm_dir)
    # where to put data
    if not os.path.exists(f'{data_dir}/int/sentinel'):
        os.mkdir(f'{data_dir}/int/sentinel')
    chunked_sentinel_dir = os.path.join(data_dir, 'int/sentinel/sentinel_by_site_32736_10m')
    chunked_alos_dir = os.path.join(data_dir, 'int/alos/alos_by_site_20_FNF')
    if not os.path.exists(os.path.join(data_dir, 'int/alos')): os.mkdir(os.path.join(data_dir, 'int/alos'))
    merged_alos_fn = os.path.join(data_dir, 'raw/alos/Karingani_merged_20_FNF/Karingani_merged_20_C.tif')
    
    process_sentinel_to_Karingani_data(chm_dir, chunked_sentinel_dir)
    
 #   process_alos_to_sentinel_site_data(chunked_sentinel_dir, chunked_alos_dir, merged_alos_fn)
