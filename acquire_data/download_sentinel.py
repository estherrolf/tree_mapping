import requests
import os
from urllib.parse import unquote
from typing import Optional, Sequence
import numpy as np
import planetary_computer
from pystac import Item
from tqdm import tqdm
import json
import rasterio

# Code is modified from https://gist.github.com/calebrob6/438c3c1ca3078476792e1f5f2195bac5 

DATA_DIR = data_dir = "../data"
# DATA_DIR = data_dir = '../../../tambe_lab/Users/luciagordon/tree_mapping_lucia_branch/data'

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
    requests.packages.urllib3.disable_warnings()
    response = requests.get(stac_item_url, verify = False, timeout = 10)
    stac_item_data = response.json()
    item = planetary_computer.sign(Item.from_dict(stac_item_data))
    
    print('donwloading the following bands: ', bands)
    for band in bands:
        download_file_from_url(item.assets[band].href, output_dir=output_dir)
        
def download_sentinel_tile(sentinel_id, bands):
    pc_collection_path = f"https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a"

    if not os.path.exists(f"{DATA_DIR}/raw/sentinel_2021"): os.mkdir(f"{DATA_DIR}/raw/sentinel_2021")

    sentinel_data_dir = f"{DATA_DIR}/raw/sentinel_2021/{sentinel_id}"
    pc_download_fp = f"{pc_collection_path}/items/{sentinel_id}"

    download_sentinel_to_directory(
            pc_download_fp,
            sentinel_data_dir,
            # bands = [
            #    "B02", # blue
            #    "B03", # green
            #    "B04", # red
            #    "B08", # NIR
            #     "visual"
            # ] 
        bands=bands
        )
    
def calculate_image_statistics(sentinel_dirs, channels):
    
    image_stats_by_tile = {}
    for channel in channels:
        image_stats_by_tile[channel] = []

    for sentinel_dir in sentinel_dirs:
        for fn in os.listdir(sentinel_dir):
            if not fn.endswith('.tif'): continue
            channel = fn.split('_')[2].split('.')[0]
            if channel.startswith('B'):
                with rasterio.open(os.path.join(sentinel_dir, fn)) as f:

                    data = f.read()

                    this_dict = {
                        'vals': data.ravel()
                    }

                    image_stats_by_tile[channel].append(this_dict)
                    
    # aggregate by chanel
    image_stats_by_channel = {}
    for channel in channels:
        if channel.startswith('B'):
            all_pix = []
            for x in image_stats_by_tile[channel]:
                all_pix.append(x['vals'])
            image_stats_by_channel[channel] = {'mean': np.mean(all_pix),
                                              'std': np.std(all_pix)
                                              }
        
    # save
    if not os.path.exists(f"{DATA_DIR}/int/image_stats/"):
        os.mkdir(f"{DATA_DIR}/int/image_stats/")
    with open(f"{DATA_DIR}/int/image_stats/sentinel_2021_image_stats.json", "w") as outfile: 
        json.dump(image_stats_by_channel, outfile)
    
if __name__ == "__main__":
    
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'visual']

#    2023
#     sentinel_ids_Karingani = [
#         "S2A_MSIL2A_20230418T073611_R092_T36KVU_20230419T022704",
#         "S2B_MSIL2A_20230413T073619_R092_T36KUU_20230413T131907",
#         "S2A_MSIL2A_20230418T073611_R092_T36JVT_20230419T032358",
#         "S2A_MSIL2A_20230418T073611_R092_T36JUT_20230419T022652"
#     ]

   # 2021
    sentinel_ids_Karingani = [
       "S2B_MSIL2A_20210513T073609_R092_T36KVU_20210606T053436",
       "S2B_MSIL2A_20210513T073609_R092_T36KUU_20210514T122203",
       "S2B_MSIL2A_20210513T073609_R092_T36JUT_20210514T161910" 
       #    "S2B_MSIL2A_20210513T073609_R092_T36JVT_20210514T075426",# not actually overlapping but completes the quadrants
    ]
    
    for sentinel_id in sentinel_ids_Karingani:
        download_sentinel_tile(sentinel_id, bands)
        
    sentinel_dirs = [f"{DATA_DIR}/raw/sentinel_2021/{x}" for x in sentinel_ids_Karingani]
    calculate_image_statistics(sentinel_dirs, bands)
