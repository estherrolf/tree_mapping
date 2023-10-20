import requests
import os
from urllib.parse import unquote
from typing import Optional, Sequence
import numpy as np
import planetary_computer
from pystac import Item
from tqdm import tqdm

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
        
def download_sentinel_to_directory(
    stac_item_url: str,
    output_dir: str,
    bands: Sequence[str] = [
       "B02", # blue
       "B03", # green
       "B04", # red
       "B08", # NIR
        "visual"
    ]
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    
    item = Item.from_file(stac_item_url)
    item = planetary_computer.sign(item)
    
    print('donwloading the following bands: ', bands)
    for band in bands:
        download_file_from_url(item.assets[band].href, output_dir=output_dir)
        
def download_sentinel_tile(sentinel_id):
    pc_collection_path = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a"

    if not os.path.exists(f"{DATA_DIR}/raw/sentinel"): os.mkdir(f"{DATA_DIR}/sentinel")

    sentinel_data_dir = f"{DATA_DIR}/raw/sentinel/{sentinel_id}"
    pc_download_fp = f"{pc_collection_path}/items/{sentinel_id}"

    download_sentinel_to_directory(
            pc_download_fp,
            sentinel_data_dir,
            bands = [
               "B02", # blue
               "B03", # green
              "B04", # red
              "B08", # NIR
               "visual"
            ] 
        )
        
if __name__ == "__main__":

    
    sentinel_ids_Karingani = [
        "S2A_MSIL2A_20230418T073611_R092_T36KVU_20230419T022704",
        "S2B_MSIL2A_20230413T073619_R092_T36KUU_20230413T131907",
        "S2A_MSIL2A_20230418T073611_R092_T36JVT_20230419T032358",
        "S2A_MSIL2A_20230418T073611_R092_T36JUT_20230419T022652"
    ]
    
    for sentinel_id in sentinel_ids_Karingani:
        download_sentinel_tile(sentinel_id)
    

    