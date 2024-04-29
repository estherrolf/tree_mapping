# imports
from rasterio import features
from utils import get_project_dir
import geopandas as gpd
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import os
import rasterio

data_dir = f'{get_project_dir()}/data'

def rasterize(vector_path, resolution): # resolution = number of meters covered by each pixel in the TIFF
    '''Converts a vector to a raster at a specified resolution'''

    feature = vector_path.split('/')[-1].split('_')[0]
    raster_path = f'{data_dir}/features/{feature}/{feature}_raster_{resolution}m.tif'

    # rasterize shapefile
    vector = gpd.read_file(vector_path)
    bounds = vector.total_bounds
    width = int((bounds[2] - bounds[0]) / resolution)
    height = int((bounds[3] - bounds[1]) / resolution)
    transform = rasterio.transform.from_origin(bounds[0], bounds[3], resolution, resolution)
    categories_to_numbers = {category: i+1 for i, category in enumerate(vector[feature.upper()].unique())}

    with rasterio.open(raster_path,
                       'w',
                       driver='GTiff',
                       height=height,
                       width=width,
                       count=1,
                       dtype=rasterio.uint8,
                       crs=vector.crs,
                       transform=transform) as dst:

        burned = features.rasterize(((geometry, value) for geometry, value in zip(vector.geometry, vector[feature.upper()].map(categories_to_numbers))),
                                     out_shape=(height, width),
                                     transform=transform,
                                     fill=0,
                                     all_touched=True,
                                     dtype=rasterio.uint8) # burn the features into the raster

        dst.write_band(1, burned) # write the rasterized shapefile to the GeoTIFF

    # convert raster to array
    array = rasterio.open(raster_path).read(1)
    print(f'Array shape = {array.shape}')
    print(f'Min array = {np.amin(array)}, max array = {np.amax(array)}')

    # plot raster
    categories_to_numbers = {'background': 0, **categories_to_numbers}
    plt.figure(dpi=300)
    cmap = plt.get_cmap('viridis', len(categories_to_numbers))
    plt.imshow(array, cmap=cmap) # plot the array of pixel values as an image
    patches = [mpatches.Patch(color=cmap(i), label=cat) for i, cat in enumerate(categories_to_numbers)]
    plt.legend(handles=patches[1:], fontsize=6)
    plt.axis('off') # remove axes
    os.makedirs(f'figures/{feature}', exist_ok=True)
    plt.savefig(f'figures/{feature}/{feature}_raster_{resolution}m.png', bbox_inches='tight', pad_inches=0)
    plt.close() # close the image to save memory

    print(f'Rasterized at {resolution}m resolution')

if __name__ == '__main__':
    rasterize(vector_path=f'{data_dir}/features/geology/geology_soil_vegetation_shapefile/geology_soil_vegetation_shapefile.shp', resolution=10)
    rasterize(vector_path=f'{data_dir}/features/geology/geology_soil_vegetation_shapefile/geology_soil_vegetation_shapefile.shp', resolution=30)
