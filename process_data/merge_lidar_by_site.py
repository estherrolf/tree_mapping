import os
import rasterio
import subprocess
from geo_utils import assign_crs_to_tif, merge_tifs

DATA_DIR = "/n/home10/erolf/tree_mapping/data"

def merge_lidar_tifs(tifs_to_merge, out_tif_fp, verbose=True):
    if verbose: print(len(tifs_to_merge))
    
    # assign to a common crs
    for tif_fp in tifs_to_merge: 
        assign_crs_to_tif(tif_fp, crs_out="EPSG:32736")
    
    # merge to one tif
    merge_tifs(tifs_to_merge, out_tif_fp,  nodata_val="-9999.0")
    
    return

def coarsen_lidar(input_fn,
                  output_fn,
                  target_res_x=10,
                  target_res_y=10,
                  ):

    # coarsen lidar to 10m resolution
    command = [
        "gdalwarp",
        "-overwrite",
        "-ot", "Float32",
        "-r", "average",  # this is the important bit
        "-of", "GTiff",
        "-tr", str(target_res_x), str(target_res_y),
        "-srcnodata", "-9999.",
        "-dstnodata", "-9999.",
        input_fn,
        output_fn
    ]
    subprocess.call(command)

def coarsen_all_lidar(processed_tif_fps_1m):
    
    for input_fn in processed_tif_fps_1m:
        site_id = input_fn.split('/')[-1].split('_')[0]
        output_fn =  os.path.join(coarsened_chm_dir, f'{site_id}_CHM_10m.tif')

        coarsen_lidar(input_fn,
                      output_fn, 
                      target_res_x=10,
                      target_res_y=10)
    
    
if __name__ == "__main__":
   
    data_dir = DATA_DIR 
    
    # where the input lidar tifs are stored
    raw_lidar_dir = os.path.join(data_dir, 'raw/raw_lidar')
    
    # where the merged lidar tifs will be stored after this is done
    crs_lidar_dir = os.path.join(data_dir, 'raw/lidar_by_site_32736')
    if not os.path.exists(crs_lidar_dir): os.mkdir(crs_lidar_dir)
    
    # get all the site names
    Karingani_sites = os.listdir(raw_lidar_dir)
    if '.DS_Store' in Karingani_sites: Karingani_sites.remove('.DS_Store')
    
    # run through each site and aggregate - 1m
    processed_tif_fps_1m = []
    for site_name in Karingani_sites:
        print(site_name)
        # prepare file names
        site_id = site_name.split('_')[1]
        site_dir = f'{raw_lidar_dir}/{site_name}/CHM'
        tifs_to_merge = [f'{site_dir}/{x}'  for x in os.listdir(site_dir) if x.endswith('.tif')]
        out_tif_fp = os.path.join(crs_lidar_dir, f'{site_id}_CHM_1m_merged.tif' )
        # merge files and save 
        merge_lidar_tifs(tifs_to_merge, out_tif_fp, verbose=False)
        processed_tif_fps_1m.append(out_tif_fp)
        
    # coarsen each to 10m res and save as a different file    
    # this cell needs to be moved to a script
    coarsened_chm_dir = f'{data_dir}/int/lidar/lidar_by_site_32736_10m'

    for dir_this in [f'{data_dir}/int/', f'{data_dir}/int/lidar/', coarsened_chm_dir]:
        if not os.path.exists(dir_this):
            os.mkdir(dir_this)
            print('made dir ',dir_this)

    coarsen_all_lidar(processed_tif_fps_1m)
    