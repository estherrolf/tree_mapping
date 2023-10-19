import rasterio
import subprocess

def assign_crs_to_tif(tif_fp, crs_out):
    command = ["gdal_edit.py",
               "-a_srs", crs_out, 
               tif_fp]
    subprocess.call(command)
    
    
def merge_tifs(in_tif_fps, out_tif_fp,  nodata_val="-9999.0"):
    command = ["gdal_merge.py",
           "-ot", "Float32", 
           "-of", "GTiff", 
           "-o", out_tif_fp,
           "-n", nodata_val,
           "-a_nodata", nodata_val,
          ] + in_tif_fps
    subprocess.call(command)