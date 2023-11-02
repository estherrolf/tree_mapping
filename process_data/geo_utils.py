import rasterio
import subprocess

def assign_crs_to_tif(tif_fp, crs_out):
    command = ["gdal_edit.py",
               "-a_srs", crs_out, 
               tif_fp]
    subprocess.call(command)
    
    
def merge_tifs(in_tif_fps, out_tif_fp,  nodata_val="-9999.0", output_type=None):
    # if output_type not specified, match the dtype of the first tif
    if output_type is None:
        with rasterio.open(in_tif_fps[0], "r") as f:
            output_type = f.dtypes[0]
    
    command = ["gdal_merge.py",
           "-ot", output_type, 
           "-of", "GTiff", 
           "-o", out_tif_fp,
           "-n", nodata_val,
           "-a_nodata", nodata_val,
          ] + in_tif_fps
    subprocess.call(command)
    
    
def match_input_to_target_tif(input_fn, 
                              output_fn, 
                              target_fn, 
                              pixel_buffer = 0, 
                              verbose=False, 
                              output_type=None,
                              resampling="average",
                              output_nodata="-9999."):
    """ Crops and reprojects input tn to match target fn in crs and res, possibly with a buffer of pixel_buffer pixels per edge.
    """
    
    # if output_type not specified, match the dtype of the input tif
    if output_type is None:
        with rasterio.open(input_fn, "r") as f:
            output_type = f.dtypes[0]
            print(output_type)
    
    # if output_type not specified, match the dtype of the input tif
    if output_nodata is None:
        with rasterio.open(input_fn, "r") as f:
            output_nodata = f.nodata
    
    with rasterio.open(target_fn, "r") as f:
        left, bottom, right, top = f.bounds
        crs = f.crs.to_string()
      #  height, width = f.height, f.width
        res_x, res_y = f.res
            
    left = left - (pixel_buffer * res_x)
    bottom = bottom - (pixel_buffer  * res_x)
    right = right + (pixel_buffer * res_y)
    top = top + (pixel_buffer  * res_y)

    command = [
        "gdalwarp",
        "-overwrite",
        "-ot", output_type,
        "-t_srs", crs,
        "-r", resampling,
        "-of", "GTiff",
        "-te", str(left), str(bottom), str(right), str(top),
        "-tr", str(res_x), str(res_y),
      #  "-ts", str(width), str(height),
        # "-co", "COMPRESS=LZW",
        # "-co", "BIGTIFF=YES",
        "-dstnodata", output_nodata,
        input_fn,
        output_fn
    ]
    
    if verbose: print(command)
    subprocess.call(command)
    
    if verbose: print(f'saved in {output_fn}')
    return

def crop_input_to_target_tif(input_fn, 
                             output_fn, 
                             target_fn, 
                             pixel_buffer = 0, 
                             verbose=False, 
                             output_type=None,
                             match_res=False,
                             resampling="average",
                             output_nodata=None):
    """
    Crops input file to match target file in crs, possibly with a buffer of pixel_buffer pixels per edge. 
    
    Will not reproject. Note: pixel buffer is in terms of the input fn resolution.
    """
    
    # if output_type not specified, match the dtype of the input tif
    with rasterio.open(input_fn, "r") as f:        
        if output_type is None:
            output_type = f.dtypes[0]
        res_x, res_y = f.res
        if output_nodata is None:
            output_nodata = f.nodata
            if output_nodata is None:
                print('no nodata specified in file and none specied to replace it')
            
    # Uses bounds and crs of target file, will not reproject!
    with rasterio.open(target_fn, "r") as f:
        left, bottom, right, top = f.bounds
        crs = f.crs.to_string()

    left = left - (pixel_buffer * res_x)
    bottom = bottom - (pixel_buffer  * res_x)
    right = right + (pixel_buffer * res_y)
    top = top + (pixel_buffer  * res_y)
    
    command = [
        "gdalwarp",
        "-overwrite",
        "-ot", output_type,
        "-t_srs", crs,
        "-r", resampling,
        "-of", "GTiff",
        "-te", str(left), str(bottom), str(right), str(top),
        "-co", "COMPRESS=LZW",
        "-co", "BIGTIFF=YES",
        "-dstnodata", output_nodata,
        input_fn,
        output_fn
    ]
    if match_res:
        command.insert(10, "-tr")
        command.insert(11, str(res_x))
        command.insert(12, str(res_y))
    if verbose: print(command)
    subprocess.call(command)
    
    if verbose: print(f'saved in {output_fn}')
    return