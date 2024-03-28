# imports
import numpy as np
import os
import yaml
from utils import get_project_dir

def read_config_file(config_yaml):
    with open(config_yaml, 'r') as cfg_file:
        cfg = yaml.safe_load(cfg_file)
        
    return cfg

def make_splits(num_sites, seed=0):
    '''Splits site numbers into four groups, where each group appears in exactly one test set, one val set, and two train sets'''

    random_state = np.random.RandomState(seed) # creates a RandomState instance with a specific seed
    random_order = random_state.choice(num_sites, num_sites, replace = False) # list of site numbers in random order
    l1 = num_sites // 4
    l2 = num_sites // 2
    l3 = (num_sites * 3) // 4
    
    sets = [random_order[:l1], random_order[l1:l2], random_order[l2:l3], random_order[l3:]] 
    split_orders = [{'train': np.hstack((sets[1], sets[2])), 'val': np.array(sets[3]), 'test': np.array(sets[0])},
                    {'train': np.hstack((sets[2], sets[3])), 'val': np.array(sets[0]), 'test': np.array(sets[1])},
                    {'train': np.hstack((sets[3], sets[0])), 'val': np.array(sets[1]), 'test': np.array(sets[2])},
                    {'train': np.hstack((sets[0], sets[1])), 'val': np.array(sets[2]), 'test': np.array(sets[3])}]
    
    return split_orders

def get_site_splits(random_seed, data_dir=f'data'):
    '''Gets sites for train, validation, and test set for four splits'''

    non_hidden_dirs = [x for x in os.listdir(os.path.join(data_dir,'int/lidar/lidar_by_site_32736_10m')) if not x.startswith('.')]
    all_sites = np.sort(non_hidden_dirs)
    split_orders = make_splits(num_sites=len(all_sites), seed=random_seed)
    splits = {}

    for split_number, split_order in enumerate(split_orders):
        train_sites = [all_sites[x] for x in split_order['train']]
        val_sites = [all_sites[x] for x in split_order['val']]
        test_sites = [all_sites[x] for x in split_order['test']]
        
        split = {'train_sites': train_sites,
                 'val_sites': val_sites,
                 'test_sites': test_sites}
    
        splits[split_number] = split

    return splits