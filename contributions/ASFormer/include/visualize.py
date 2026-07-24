import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
from matplotlib import colormaps
from utils import get_labels_start_end_time


def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        f.close()
    return content


def survey(results, category_names, colors, ax, fig_size=(25, 3)):
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    
    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data))
    
    for i, (colname, color) in enumerate(zip(category_names, colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh([""], widths, left=starts, height=0.75, label=colname, color=color)

        r, g, b = color
        text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
        ax.bar_label(rects, labels=[colname], label_type='center', color=text_color, rotation=90 if fig_size[1] >=3 else 0 )
    return ax


def get_labels(dataset_name):
    if "cataract101" in dataset_name:
        labels = ['action_start', 'incision', 'viscous_agent_injection', 'rhexis', 'hydrodissection', 'phacoemulsificiation', 'irrigation_aspiration', 
                  'capsule_polishing', 'lens_implant_settingup', 'viscous_agent_removal', 'tonifying_antibiotics', 'action_end']
    else:
        labels = ['background', 'main_incision_entry', 'cautery', 'peritomy', 'tunnel_suture', 'hydroprocedure', 'conjunctival_cautery', 'tunnel', 
                  'nucleus_prolapse', 'OVD_IOL_insertion', 'sideport', 'scleral_groove', 'OVD_injection', 'cortical_wash', 'OVD_wash', 'stromal_hydration', 
                  'nucleus_delivery', 'incision', 'capsulorrhexis', 'AB_injection_and_wash', 'SR_bridle_suture']
    return labels


def get_colormap(labels, cm_name='Blues'):
    colormap = colormaps[cm_name]
    colors = colormap(np.linspace(0, 1, len(labels)))
    color_mapping = {labels[i]: colors[i, :][:3] for i in range(len(labels))}
    return color_mapping


def plot_action_list(data, labels, title=None):
    color_mapping = get_colormap(labels)
    if data[-1] == '': 
        data = data[:-1]
        
    labels_gt, start_gt, end_gt = get_labels_start_end_time(data)
    data_gt = {"Data": [end_gt[i] - start_gt[i] for i, _ in enumerate(start_gt)]}
    plt.rcParams.update({'font.size': 16})
        
    fig, axs = plt.subplots(2, 1, figsize=(25, 6), sharex=True, sharey=True, layout="tight")
    axs[0].axis('off')
    axs[1].axis('off')
    if title:
        axs[0].set_title(title)

    colors_gt = [color_mapping[elm] for elm in labels_gt]
    ax = survey(data_gt, labels_gt, colors_gt, axs[0])
    #fig.savefig(output_dir + case + ".png")
    fig.delaxes(axs[1])
    return ax
