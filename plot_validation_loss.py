'''
plot_validation_loss.py plots the loss as a function of epochs so we can determine if the loss is flattening out or if more epochs are needed
'''

# imports
from utils import get_project_dir
import matplotlib.pyplot as plt
import numpy as np
import os

project_dir = get_project_dir()

def visualize_loss(exp_dir, epochs):
    results = {'split_0': {}, 'split_1': {}, 'split_2': {}, 'split_3': {}}

    for split in results:
        for lr in [0.00001, 0.0001, 0.001, 0.01]:
            for wd in [0.00001, 0.0001, 0.001, 0.01]:
                results[split][f'lr_{lr}_wd_{wd}'] = np.load(f'{exp_dir}/{split}/logs/lr_{lr}_wd_{wd}/val_loss.npy')
    
    for split in results.keys():
        if len(results[split]) != 16:
            print('wrong length')
        for combo in results[split]:
            if len(results[split][combo]) != epochs:
                print('wrong shape')

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan', 'rosybrown', 'lightcoral', 'bisque', 'darkorange', 'forestgreen', 'limegreen', 'slategrey', 'lightsteelblue']

    for split in results.keys():
        fig, ax = plt.subplots(dpi=300)
        split_results = []
        title = f'{exp_dir.split("experiment_results/")[1]}_split_{split.split("_")[1]}'.replace("/", "_")
        print(title)

        for combo in results[split]:
            combo_results = results[split][combo]
            split_results += [combo_results]
            ax.plot(np.arange(1, len(combo_results)+1), combo_results, label=combo)

        ax.set(ylim=(0,10))
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Validation Loss (m$^2$)')
        os.makedirs(f'figures/{exp_dir.split("experiment_results/")[1]}', exist_ok=True)
        plt.savefig(f'figures/{exp_dir.split("experiment_results/")[1]}/{split}.png', bbox_inches='tight', pad_inches=0.1)
        plt.close() # close the image to save memory

        split_results = np.array(split_results)
        where_min = np.argwhere(split_results == np.min(split_results))[0]
        print(f'Split {split.split("_")[1]} achieves the minimum loss of {round(np.min(split_results),5)} for {list(results[split].keys())[where_min[0]]} after epoch {where_min[1]}')

if __name__ == '__main__':
    exp_dir = f'{project_dir}/experiment_results/local_only_models/128_filters/3_channels'
    epochs = 300

    visualize_loss(exp_dir, epochs)
