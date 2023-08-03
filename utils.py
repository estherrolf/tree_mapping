import rasterio
import subprocess
import os
data_dir = "/n/home10/erolf/tree_mapping/data"

sentinel_layer_codes = {"b": "B02",
                        "g": "B03",
                        "r": "B04",
                        "nir":"B08",
                        "vis":"TCI",
                       }



def match_input_to_target_tif(input_fn, output_fn, target_fn, verbose=False):
    # crops and reprojects input tn to match target fn
    with rasterio.open(target_fn, "r") as f:
        left, bottom, right, top = f.bounds
        crs = f.crs.to_string()
        height, width = f.height, f.width

    command = [
        "gdalwarp",
        "-overwrite",
        "-ot", "Float32",
        "-t_srs", crs,
        "-r", "average",
        "-of", "GTiff",
        "-te", str(left), str(bottom), str(right), str(top),
        "-ts", str(width), str(height),
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        "-dstnodata", "-9999",
        input_fn,
        output_fn
    ]
    
    if verbose: print(command)
    subprocess.call(command)
    
    if verbose: print(f'saved in {output_fn}')
    return

def crop_input_to_target_tif(input_fn, output_fn, target_fn, buffer = 0, verbose=False):
    # will not reproject!
    with rasterio.open(target_fn, "r") as f:
        left, bottom, right, top = f.bounds
        crs = f.crs.to_string()
        height, width = f.height, f.width

    left = left - buffer
    bottom = bottom - buffer
    right = right + buffer
    top = top + buffer
    
    command = [
        "gdalwarp",
        "-overwrite",
        "-ot", "Float32",
        "-t_srs", crs,
        "-r", "average",
        "-of", "GTiff",
        "-te", str(left), str(bottom), str(right), str(top),
   #     "-ts", str(width), str(height),
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        "-dstnodata", "-9999",
        input_fn,
        output_fn
    ]
    if verbose: print(command)
    subprocess.call(command)
    
    if verbose: print(f'saved in {output_fn}')
    return