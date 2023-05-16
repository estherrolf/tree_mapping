#from torchgeo.datasets import RasterDataset, IntersectionDataset, Sentinel2
from torchgeo.datasets import IntersectionDataset, RasterDataset, Sentinel2, stack_samples
import rasterio
import subprocess
data_dir = "/n/home10/erolf/tree_mapping/data"

sentinel_layer_codes = {"b": "B02",
                        "g": "B03",
                        "r": "B04",
                        "nir":"B08",
                        "vis":"TCI",
                       }

def make_site_dataset(site,  
                      transforms, 
                      layers=["b","g","r","vis","chm"],
                      sentinel_data_dir = "/sentinel/S2A_MSIL2A_20230418T073611_R092_T36KVU_20230419T022704",
                      data_dir=data_dir):
    
    sentinel_data_dir = data_dir
    
    non_img_layers = ['chm']
    img_layers = [l for l in layers if not l in non_img_layers]
    
    sentinel = Sentinel2(
        sentinel_data_dir,
        bands=[sentinel_layer_codes[l.lower()] for l in img_layers]
    )
    
    # messy todo deal with nicer for now forse chm to be laste layer
    
    if "chm" in layers: assert layers[-1] == "chm"
    print(site)
    chm_dataset = RasterDataset(root=data_dir + f'/lidar/Karingani_merged_crs_10/site_{site}')
    ds = IntersectionDataset(sentinel, chm_dataset, transforms=transforms)
    
    return ds

def match_input_to_target_tif(input_fn, output_fn, target_fn):
    
    with rasterio.open(target_fn, "r") as f:
        left, bottom, right, top = f.bounds
        crs = f.crs.to_string()
        height, width = f.height, f.width

    command = [
        "gdalwarp",
        "-overwrite",
        "-ot", "Float32",
        "-t_srs", crs,
        "-r", "near",
        "-of", "GTiff",
        "-te", str(left), str(bottom), str(right), str(top),
        "-ts", str(width), str(height),
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        "-dstnodata", "-9999",
        input_fn,
        output_fn
    ]
    subprocess.call(command)
    
    print(f'saved in {output_fn}')
    return