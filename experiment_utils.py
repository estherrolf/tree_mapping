import numpy as np
import os

def make_splits(num_sites,seed=0):
    # splits site numbers into four groups, where each group appears in exactly one test set,
    # one val set, and two train set
    rs = np.random.RandomState(seed)
    
    random_order = rs.choice(num_sites, num_sites, replace=False)
    l1 = num_sites // 4
    l2 = num_sites // 2
    l3 = (num_sites*3) // 4
    
    sets = [random_order[:l1], random_order[l1:l2], random_order[l2:l3], random_order[l3:]] 
    splits = [
        {'train': np.hstack((sets[1], sets[2])), 'val': np.array(sets[3]), 'test': np.array(sets[0])},
        {'train': np.hstack((sets[2], sets[3])), 'val': np.array(sets[0]), 'test': np.array(sets[1])},
        {'train': np.hstack((sets[3], sets[0])), 'val': np.array(sets[1]), 'test': np.array(sets[2])},
        {'train': np.hstack((sets[0], sets[1])), 'val': np.array(sets[2]), 'test': np.array(sets[3])},
    ]
    
    return splits

def get_site_splits(random_seed, 
                    data_dir='data'):
    # get sites for train, validation and test set for four splits
    
    all_sites = np.sort(os.listdir(os.path.join(data_dir,'int/lidar/lidar_by_site_32736_10m')))

    split_orders = make_splits(len(all_sites), seed=random_seed)
    splits = {}
    for split_number, split_order_this in enumerate(split_orders):
        
        train_sites = [all_sites[x] for x in split_order_this['train']]
        val_sites = [all_sites[x] for x in split_order_this['val']]
        test_sites = [all_sites[x] for x in split_order_this['test']]
        
        this_split = {
            "train_sites": train_sites,
            "val_sites": val_sites,
            "test_sites": test_sites
        }
    
        splits[split_number] = this_split

    return splits