# tree_mapping

## Set up the environment.
Create and activate a conda environment using 
```
conda env create -f tree_mapping.yml
conda activate tree_mapping
```


## prepare data.
First, from the `tree_mapping` directory, to download the relevant sentinel tiles, run the scripts `acquire_data.sh`. You will also need to get download:
 - the CHM files (from Esther or Jenia right now).

For comparison to existing data products, you will need to download:
 - existing tree cover predicted maps from [Lang et al. 2023](https://www.nature.com/articles/s41559-023-02206-6), by downloading the relevant tiles from their [tile browser](https://langnico.github.io/globalcanopyheight/assets/tile_index.html).
 - [TODO - Lucia] tree cover maps from Glad [project page](https://glad.umd.edu/dataset/gedi).

Second, from the same directory, run `process_Karingani_sites.sh` which will produce the data needed for training and evaluating our models: 
1. aggregate lidar-derived CHM data data by site (at native 1m resolution and coarsened 10m resolution).
2. aggregate all data products (sentinel2) and project to match the coarsened CHM data for each site. 

Third, run through the steps in `process_data/prepare_tch_reference_data.ipynb` which will produce the data needed to evaluate existing tree canopy height predicted maps. [TODOS: (now) process for 2019 map and (eventually) make into a script]

[TODO: (eventually) any other layers]

The notebook `tree_mapping/process_data/visualize_data_preparation` will show you what each of these steps is doing.

## evaluating existing data products
First run `prepare_tch_reference_data.ipynb` as explained in the previous section.

Then run `eval_existing_tch.ipynb`.
