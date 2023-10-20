# tree_mapping

# prepare data.
First, from the `tree_mapping` directory, to download the data run the scripts `acquire_data.sh` (this will donwload the relevant sentinel tiles). You will also need to get the CHM files (from Esther or Jenia right now).

Second, from the same directory, run `process_Karingani_sites.sh` which will produce the data needed for training and evaluating our models: 
1. aggregate lidar-derived CHM data data by site (at native 1m resolution and coarsened 10m resolution).
2. aggregate all data products (sentinel2) and project to match the coarsened CHM data for each site. 

[TODO: Third, run `process_external_data.sh` which will produce the data needed to evaluate existing tree canopy height predicted maps.]

[TODO: DEM Layers]

The notebook `tree_mapping/process_data/visualize_data_preparation` will show you what each of these steps is doing.

# evaluating existing data products
First run `prepare_tch_reference.ipynb`.

Then run `eval_existing_tch.ipynb`.
