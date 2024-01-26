# imports
from osgeo import gdal
from rasterio import features
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import rasterio

gdal.UseExceptions()

def rasterize(vector_path, raster_name, resolution=10): # resolution = number of meters covered by each pixel in the TIFF
    # rasterize shapefile
    vector = gpd.read_file(vector_path)
    bounds = vector.total_bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = rasterio.transform.from_origin(bounds[0], bounds[3], resolution, resolution)

    out_tiff = f'{raster_name}.tiff'
    with rasterio.open(out_tiff,
                       'w',
                       driver='GTiff',
                       height=height,
                       width=width,
                       count=1,
                       dtype=rasterio.uint8,
                       crs=vector.crs,
                       transform=transform) as dst:
        burned = features.rasterize(((geometry, 255) for geometry in vector.geometry),
                                     out_shape=(height, width),
                                     transform=transform,
                                     fill=0,
                                     all_touched=True,
                                     dtype=rasterio.uint8) # burn the features into the raster
        
        dst.write_band(1, burned) # write the rasterized shapefile to the GeoTIFF

    # convert raster to array
    raster = gdal.Open(f'{raster_name}.tiff')
    num_rows = raster.RasterYSize
    num_cols = raster.RasterXSize
    array = ((raster.GetRasterBand(1)).ReadAsArray(0, 0, num_cols, num_rows).astype(np.float32)) # 0 = not river, 255 = river
    print(f'Array shape = {array.shape}')
    print(f'Min array = {np.amin(array)}, max array = {np.amax(array)}')

    for row in range(array.shape[0]):
        for col in range(array.shape[1]):
            if array[row][col] != 0 and array[row][col] != 255:
                print(array[row][col])

    # plot raster
    plt.figure(dpi=300)
    plt.imshow(array) # plot the array of pixel values as an image
    plt.axis('off') # remove axes        
    plt.savefig(f'{raster_name}.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

rasterize(vector_path='../KaGR_Riv_Updated_20230326_UTM36s/KaGR_Riv_Updated_20230326_UTM36s.shp',
          raster_name='river_raster')
