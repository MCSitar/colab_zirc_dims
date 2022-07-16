# -*- coding: utf-8 -*-
"""
Functions for creating, returning 'exploratory' matplotlib plots
"""

import numpy as np
from matplotlib import pyplot as plt

__all__ = ['plot_exp_bar_whisker',
           'plot_exp_histogram',
           'plot_exp_XY']

def plot_exp_bar_whisker(data_dict_input, plot_param, sort_param, sorting_style,
                         plot_dpi = 200):
    """Return an exploratory bar-whisker plot from the input dataset and params.

    Parameters
    ----------
    data_dict_input : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.
    plot_param : str
        Name of data_dict_input column for plotting.
    sort_param : str
        Name of data_dict_input column for sorting samples.
    sorting_style : str
        Sorting style for samples; can be 'Ascending' or 'Descending'.
    plot_dpi : int, optional
        DPI for output plot. The default is 200.

    Returns
    -------
    fig : pyplot figure instance
        A figure for a bar-whisker plot created from the input dataset.

    """

    fig, ax = plt.subplots()
    sample_medians = []
    sorting_medians = []
    usable_keys = list(data_dict_input.keys())

    for each_key in usable_keys:
        sample_medians.append(np.median(data_dict_input[each_key][plot_param]))
        sorting_medians.append(np.median(data_dict_input[each_key][sort_param]))
    sorted_keys = [x for _,x in sorted(zip(sample_medians, usable_keys))]
    if plot_param != sort_param:
        sorted_keys = [x for _,x in sorted(zip(sorting_medians, usable_keys))]
    if sorting_style == 'Descending':
        sorted_keys = sorted_keys[::-1]

    n_labels = []
    for each_key in sorted_keys:
        each_n=len(data_dict_input[each_key][plot_param])
        n_labels.append(str(each_key) + '\n' + '(n = ' + str(each_n) + ')')
        plot_data = [data_dict_input[each_key][plot_param] for each_key in sorted_keys]

    plot_positions = np.asarray(range(len(plot_data)), dtype=object)
    div_lines = []
    ax.boxplot(plot_data, showfliers=False,
               positions = plot_positions, widths = 0.6,
               boxprops=dict(facecolor=(0,0,1,0.5)),
               medianprops=dict(color='black'),
               patch_artist=True)
    ax.set_xticks(range(0, len(sorted_keys)))
    ax.set_xticklabels(n_labels, fontsize=11)

    ax_lim = max([max(sample_data) for sample_data in plot_data])
    ax_lim += 0.05 * ax_lim
    if ax_lim >= 15000:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 2500))
    elif ax_lim >= 10000:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 1000))
    elif ax_lim >= 5000:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 500))
    elif ax_lim >= 2500:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 250))
    elif ax_lim >= 1000:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 100))
    elif ax_lim > 200:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 50))
    elif ax_lim <= 200:
        y_ax_ticks = list(range(0, round(ax_lim) + 1, 25))

    if plot_param in ['Eccentricity', 'Circularity']:
        ax_lim = 1
        y_ax_ticks = [val * 0.1 for val in range(0, 11)]
    for pos in plot_positions[:-1]:
        xs = [pos + .5 for i in range(0,2)]
        ys = [-50, 2*ax_lim]
        div_lines.append([xs, ys])

    ax.set_yticks(y_ax_ticks)
    ax.yaxis.set_tick_params(labelsize=11)
    for idx, sample in enumerate(plot_data):
        y = np.asarray(sample, dtype=object)
        x = np.random.normal(plot_positions[idx], 0.06, size=len(y))
        ax.plot(x, y, color = 'darkblue', marker ='.', alpha=0.2,
                  linestyle = 'None', zorder = 1000 + idx)
    for div_line in div_lines:
        ax.plot(*div_line, color='k', marker='None', linestyle='solid', alpha=0.2)

    ax.set_ylim(0, ax_lim)
    y_label_str = str(plot_param).replace("^2", "²")
    ax.set_ylabel(y_label_str, fontsize=13,
                  fontweight='bold')


    #turn off visible ticks without turning off labels
    ax.tick_params(bottom=False, top=False)

    fig.set_size_inches(1 + len(usable_keys), 8.5, forward=True)
    fig.tight_layout()
    fig.set_dpi(int(plot_dpi))
    return fig

def plot_exp_histogram(input_dataset, color_dict, plot_param, num_bins,
                       histtype, hist_alpha, plot_dpi = 200):
    """Plot an exploratory histogram using the input dataset and parameters.

    Parameters
    ----------
    input_dataset : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.
    color_dict : dict
        Dict of format {sample1: 'hex_color_string', ...}.
    plot_param : str
        Column in input_dataset for plotting.
    num_bins : int
        Number of bins for histogram.
    histtype : str
        Type of histogram; options are 'bar' for traditional histogram,
        'barstacked' for stacked histogram.
    hist_alpha : float
        Float between 0 and 1 (inclusive) indicating the alpha for histogram
        fill.
    plot_dpi : int, optional
        DPI for output plot. The default is 200.

    Returns
    -------
    fig : pyplot figure instance
        A figure for a histogram plot created from the input dataset.

    """
    full_data_max = max([max(vals[plot_param]) for vals in input_dataset.values()])
    full_data_min = min([min(vals[plot_param]) for vals in input_dataset.values()])
    #max_round_up, min_round_down = int(full_data_max) + 1, int(full_data_min)
    bin_width = (full_data_max - full_data_min) / num_bins
    #to avoid floating point range problem with small numbers, multiply all by 1000
    bins_1000x = list(range(round(full_data_min*1000),
                            round(full_data_max*1000 + bin_width*1000),
                            round(bin_width*1000)))
    bins = [bin_val/1000 for bin_val in bins_1000x]
    fig, axes = plt.subplots(2)
    leg_handles, leg_labels = [], []
    for i in range(2):
        ax = axes[i]
        if i == 0:
            if histtype == 'bar':
                for each_sample, each_sample_vals in input_dataset.items():
                    ax.hist(each_sample_vals[plot_param], bins = bins,
                            alpha=hist_alpha,
                            label = each_sample, edgecolor="black",
                            facecolor = color_dict[each_sample],
                            histtype = histtype)
            elif histtype == 'barstacked':
                total_data, total_colors, total_labels = [],  [], []
                for each_sample, each_sample_vals in input_dataset.items():
                    total_data.append(each_sample_vals[plot_param])
                    total_colors.append(color_dict[each_sample])
                    total_labels.append(each_sample)
                ax.hist(total_data, bins=bins, label=total_labels,
                        color=total_colors, alpha=hist_alpha,
                        histtype = histtype, edgecolor="black")

            leg_handles, leg_labels = ax.get_legend_handles_labels()
            x_label_str = str(plot_param).replace("^2", "²")
            ax.set_xlabel(x_label_str, fontweight='bold')
            ax.set_ylabel('Count', fontweight='bold')

        if i == 1:
            ax.legend(leg_handles, leg_labels,
                      ncol=int(len(input_dataset.keys())/4) + 1,
                      loc='lower center',
                      frameon = False)
            ax.axis('off')
    fig.set_dpi(int(plot_dpi))
    return fig

def plot_exp_XY(input_dataset, color_dict, x_param, y_param, xy_alpha, equal_axes,
                plot_dpi = 200):
    """Plot an exploratory X-Y chart using the input dataset and entered
       parameters.

    Parameters
    ----------
    input_dataset : dict
        Dict of format {sample1: {header1: [row1, row2,...],...}, ...}.
    color_dict : dict
        Dict of format {sample1: 'hex_color_string', ...}.
    x_param : str
        Column in input_dataset for plotting on X axis.
    y_param : str
        Column in input_dataset for plotting on Y axis.
    xy_alpha : float
        Float between 0 and 1 (inclusive) indicating the alpha for plot
        markers.
    equal_axes : bool
        If True, X and Y axes will be set equal with equal aspect ratio.
    plot_dpi : int, optional
        DPI for output plot. The default is 200.

    Returns
    -------
    fig : pyplot figure instance
        A figure for a X-Y chart created from the input dataset.
    """

    fig, axes = plt.subplots(2)
    leg_handles, leg_labels = [], []
    for i in range(2):
        ax = axes[i]
        if i == 0:
            for each_sample, each_sample_vals in input_dataset.items():
                ax.scatter(each_sample_vals[x_param],
                           each_sample_vals[y_param],
                           alpha=xy_alpha,
                           color = color_dict[each_sample],
                           label = each_sample)
            leg_handles, leg_labels = ax.get_legend_handles_labels()
            x_label_str = str(x_param).replace("^2", "²")
            ax.set_xlabel(x_label_str, fontweight='bold')
            y_label_str = str(y_param).replace("^2", "²")
            ax.set_ylabel(y_label_str, fontweight='bold')
            if equal_axes:
                total_min_max = [*ax.get_xlim(), *ax.get_ylim()]
                total_min, total_max = min(total_min_max), max(total_min_max)
                ax.set_xlim(total_min, total_max)
                ax.set_ylim(total_min, total_max)
                ax.set_aspect('equal')

        if i == 1:
            ax.legend(leg_handles, leg_labels,
                      ncol=int(len(input_dataset.keys())/4) + 1,
                      loc='lower center',
                      frameon = False)
            ax.axis('off')
    fig.set_dpi(int(plot_dpi))
    return fig
