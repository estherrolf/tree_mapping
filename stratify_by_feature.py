# imports
from utils import get_project_dir
import json
import matplotlib as mpl
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

project_dir = get_project_dir()
resolution = 10
width = 0.2
small_gap = 0.5 * width
big_gap = 2 * width
fontsize = 18
plt.rcParams['xtick.labelsize'] = fontsize
plt.rcParams['ytick.labelsize'] = fontsize
plt.rcParams['xtick.major.size'] = 10
plt.rcParams['ytick.major.size'] = 10

def stratify_by_feature(feature, categories, x_axis_label):
    num_categories = len(categories)
    cmap = mpl.colormaps['hsv']

    with open('results.json', 'r') as results_json:
        results = json.load(results_json)
        settings = [setting for setting in results]
        colors = cmap(np.linspace(0, 1, len(settings))) # take colors at regular intervals spanning the colormap
        fig, ax = plt.subplots(figsize=(num_categories*(len(settings)*(width+small_gap)+big_gap), 0.5*(num_categories*(len(settings)*(width+small_gap)+big_gap))),
                               dpi=300)
        i = 0

        for setting in settings:
            setting_results = results[setting][feature]
            j = 0

            # for stats in [setting_results['0-3'], setting_results['3-6'], setting_results['6-10'], setting_results['10-30']]:
            for stats in [setting_results[interval] for interval in setting_results]:
                left = j*(len(settings)*(width+small_gap)+big_gap)+i*(width+small_gap)
                # draw the box
                box = patches.Rectangle(xy=(left, stats['q1']), width=width, height=stats['q3']-stats['q1'],
                                        edgecolor='black', facecolor=colors[i], fill=True)
                ax.add_patch(box)

                # draw median line
                median_line = patches.Rectangle(xy=(left, stats['median']), width=width, height=0,
                                                edgecolor='black', fill=False, linewidth=2)
                ax.add_patch(median_line)

                # draw whiskers
                ax.plot(2*[left+width/2], [stats['perc_10'], stats['q1']], color='black')  # lower whisker
                ax.plot(2*[left+width/2], [stats['perc_90'], stats['q3']], color='black')  # upper whisker

                # draw whisker caps
                ax.plot([left, left+width], [stats['perc_10'], stats['perc_10']], color='black')  # lower cap
                ax.plot([left, left+width], [stats['perc_90'], stats['perc_90']], color='black')  # upper cap

                j += 1
            i += 1

        tick_positions = [j*(len(settings)*(width+small_gap)+big_gap) - big_gap/2 for j in range(num_categories+1)]
        tick_positions += [np.mean([tick_positions[j], tick_positions[j+1]]) for j in range(num_categories)]
        tick_positions = sorted(tick_positions)
        ax.set_xticks(tick_positions, labels=['' if i % 2 == 0 else f'{categories[int(i/2)]}' for i in range(2*num_categories+1)])
        ax.set_xlabel(x_axis_label, fontsize=fontsize)
        ax.set_ylabel('Residuals (m)', fontsize=fontsize)
        plt.legend(handles=[patches.Patch(color=colors[i], label=settings[i]) for i in range(len(settings))],
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.1),
                   ncol=2,
                   fontsize=fontsize)
        
        for i in range(len(ax.xaxis.get_major_ticks())):
            if i % 2 != 0:
                ax.xaxis.get_major_ticks()[i].tick1line.set_visible(False)

        plt.savefig(f'figures/maps-stratified-by-{feature}-{resolution}m.png', bbox_inches='tight', pad_inches=0.2)
        plt.close()

if __name__ == '__main__':
    stratify_by_feature(feature='height', categories=['0-3', '3-6', '6-10', '10-30'], x_axis_label='LiDAR-Derived Height (m)')
    stratify_by_feature(feature='river', categories=['0-100', '100-300', '300-600', '600-1000', '1000-2000', '2000-3000'], x_axis_label='Distance to River (m)')
