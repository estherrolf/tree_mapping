import requests
import os
from urllib.parse import unquote
from typing import Optional, Sequence
import numpy as np
import planetary_computer
from pystac import Item
from tqdm import tqdm

import sys
sys.path.append('../process_data')
import geo_utils
# Code is modified from https://gist.github.com/calebrob6/438c3c1ca3078476792e1f5f2195bac5 

DATA_DIR = data_dir = "../data"

class ContentDispositionHeaderError(Exception):
    pass

def download_file_from_url(
    url: str,
    filename: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> None:
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        if not filename:
            content_disposition = response.headers.get('content-disposition')
            if content_disposition:
                filename = [part.split('=')[1] for part in content_disposition.split(';') if "filename" in part]
                if len(filename) > 0:
                    filename = unquote(filename[0].strip('\"'))
                else:
                    raise ContentDispositionHeaderError("No filename found in Content-Disposition header")
            else:
                filename = os.path.basename(url.split("?")[0])
                
        if output_dir is not None:
            local_path = os.path.join(output_dir, filename)
        else:
            local_path = filename

        num_bytes = int(response.headers.get('content-length', 0))
        progress_bar = tqdm(total=num_bytes, unit='iB', unit_scale=True)
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                progress_bar.update(len(chunk))
                f.write(chunk)
        progress_bar.close()
    else:
        raise requests.HTTPError(f"Unable to download file from URL (status code: {response.status_code})")
        
def download_alos_canopy_to_directory(
    stac_item_url: str,
    output_dir: str,
    bands: Sequence[str] = [
       "C", # canopy
    ]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    item = Item.from_file(stac_item_url)
    item = planetary_computer.sign(item)
    
    print('donwloading the following bands: ', bands)
    for band in bands:
        download_file_from_url(item.assets[band].href, output_dir=output_dir)
        
def download_canopy_tile(alos_id):
    pc_collection_path = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/alos-fnf-mosaic"

    if not os.path.exists(f"{DATA_DIR}/raw/alos"): os.mkdir(f"{DATA_DIR}/raw/alos")

    alos_data_dir = f"{DATA_DIR}/raw/alos/{alos_id}"
    pc_download_fp = f"{pc_collection_path}/items/{alos_id}"

    download_alos_canopy_to_directory(
            pc_download_fp,
            alos_data_dir,
            bands = [
               "C"
            ] 
        )
        
if __name__ == "__main__":

    
    canopy_ids_Karingani = [
        "S24E032_20_FNF",
        "S24E031_20_FNF",
        "S23E032_20_FNF",
        "S23E031_20_FNF", 
    ]
    
#     for alos_id in canopy_ids_Karingani:
#         download_canopy_tile(alos_id)
        
        
    alos_data_dir = f"{DATA_DIR}/raw/alos"
    
    if not os.path.exists(os.path.join(alos_data_dir,'Karingani_merged_20_FNF')): os.mkdir(os.path.join(alos_data_dir,'Karingani_merged_20_FNF'))
    
    compiled_tif_fn = os.path.join(alos_data_dir,'Karingani_merged_20_FNF','Karingani_merged_20_C.tif')
    
    in_tif_fps = [os.path.join(alos_data_dir, f'{alos_id}/{alos_id[:10]}_C.tif') for alos_id in canopy_ids_Karingani]
    
    geo_utils.merge_tifs(in_tif_fps, compiled_tif_fn, output_type='int8')
    