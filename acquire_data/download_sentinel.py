'''download_sentinel.py saves the sentinel data to the specified directory
based on https://gist.github.com/calebrob6/438c3c1ca3078476792e1f5f2195bac5'''

# imports
import json
import numpy as np
import os
import planetary_computer
import rasterio
import requests
import sys
import urllib3
from pystac import Item
from tqdm import tqdm

sys.path.insert(0, '') # necessary since utils is outside the acquire_data folder
from utils import get_project_dir

class DownloadSentinel:
    def __init__(self):
        self.project_dir = get_project_dir()
        self.bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'visual']
        self.sentinel_2021_dir = f'{self.project_dir}/data/raw/sentinel_2021'

        sentinel_tiles_karingani = ['S2B_MSIL2A_20210513T073609_R092_T36KVU_20210606T053436',
                                    'S2B_MSIL2A_20210513T073609_R092_T36KUU_20210514T122203',
                                    'S2B_MSIL2A_20210513T073609_R092_T36JUT_20210514T161910'] # 05/13/21

        for tile in sentinel_tiles_karingani:
            os.makedirs(f'{self.sentinel_2021_dir}/{tile}', exist_ok=True)
            self.download_sentinel_tile(tile=tile)

        self.calculate_image_statistics(sentinel_dirs=[f'{self.sentinel_2021_dir}/{tile}' for tile in sentinel_tiles_karingani])

    def download_file_from_url(self, url, tile):
        response = requests.get(url, stream=True, timeout=20) # HTTP get request to given URL

        if response.status_code == 200: # server responds with successful status
            band_tiff_filename = os.path.basename(url.split('?')[0])
            num_bytes = int(response.headers.get('content-length', 0))
            progress_bar = tqdm(total=num_bytes, unit='iB', unit_scale=True)

            with open(f'{self.sentinel_2021_dir}/{tile}/{band_tiff_filename}', 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    progress_bar.update(len(chunk))
                    file.write(chunk)

            progress_bar.close()
        else:
            raise requests.HTTPError(f'Unable to download file from URL (status code: {response.status_code})')

    def download_sentinel_tile(self, tile):
        planetary_computer_url = f'https://planetarycomputer.microsoft.com/api/stac/v1/collections/sentinel-2-l2a/items/{tile}'
        urllib3.disable_warnings()
        response = requests.get(planetary_computer_url, verify=False, timeout=20)
        stac_item_data = response.json()
        item = planetary_computer.sign(Item.from_dict(stac_item_data))

        print(f'Downloading the following bands: {self.bands}')

        for band in self.bands:
            print(band)
            self.download_file_from_url(url=item.assets[band].href, tile=tile)

    def calculate_image_statistics(self, sentinel_dirs):
        data_stats_by_tile = {}

        for channel in self.bands:
            data_stats_by_tile[channel] = []

        for sentinel_dir in sentinel_dirs:
            for tile in os.listdir(sentinel_dir):
                if not tile.endswith('.tif'):
                    continue

                channel = tile.split('_')[2].split('.')[0]

                if channel.startswith('B'):
                    with rasterio.open(f'{sentinel_dir}/{tile}') as file:
                        data = file.read()
                        this_dict = {'vals': data.ravel()}
                        data_stats_by_tile[channel].append(this_dict)

        # aggregate by channel
        data_stats_by_channel = {}

        for channel in self.bands:
            if channel.startswith('B'):
                all_pix = []

                for x in data_stats_by_tile[channel]:
                    all_pix.append(x['vals'])

                data_stats_by_channel[channel] = {'mean': np.mean(all_pix), 'std': np.std(all_pix)}

        # save
        os.makedirs(f'{self.project_dir}/data/int/data_stats/', exist_ok=True)

        with open(f'{self.project_dir}/data/int/data_stats/S2_stats_by_channel.json', 'w', encoding='utf-8') as outfile:
            json.dump(data_stats_by_channel, outfile)

if __name__ == '__main__':
    DownloadSentinel()
