import json
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def plot_ecoff_distributions(y_low_log, y_high_log, mu_hat, sigma_hat, z_99, lower_cap_value=-10, upper_cap_value=5):
    """
    Plot the distribution of wild-type MIC values and the fitted Gaussian distribution.
    
    Args:
        y_low_log (array-like): Lower bounds of log-transformed MIC intervals.
        y_high_log (array-like): Upper bounds of log-transformed MIC intervals.
        mu_hat (float): Estimated mean of the fitted distribution.
        sigma_hat (float): Estimated standard deviation of the fitted distribution.
        z_99 (float): The 99th percentile of the fitted distribution in log scale.
        cap_value (float, optional): Maximum value to cap the data for plotting. Defaults to 10.
    """
    # Cap np.inf values in the log-transformed MIC arrays
    y_low_log = np.clip(y_low_log, lower_cap_value, upper_cap_value)
    y_high_log = np.clip(y_high_log, lower_cap_value, upper_cap_value)
    
    # WT MIC Intervals
    wt_intervals = [(low, high) for low, high in zip(y_low_log, y_high_log)]
    unique_wt_intervals = sorted(set(wt_intervals))
    wt_mic_counts = [wt_intervals.count(interval) for interval in unique_wt_intervals]
    wt_midpoints = [(low + high) / 2 for low, high in unique_wt_intervals]
    wt_widths = [high - low for low, high in unique_wt_intervals]
    total_wt = sum(wt_mic_counts)
    
    # WT Fitted Curve
    wt_x_values = np.linspace(min(y_low_log), max(y_high_log), 1000)
    wt_fitted_curve = (1 / (sigma_hat * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((wt_x_values - mu_hat) / sigma_hat) ** 2)
    wt_densities = [count / (total_wt * width) for count, width in zip(wt_mic_counts, wt_widths)]

    # Calculate the widest x-range
    x_min = min(y_low_log)
    x_max = max(y_high_log)

    # Create a single plot for WT
    fig, ax_wt = plt.subplots(1, 1, figsize=(6, 4))

    # Plot WT distribution
    ax_wt.bar(wt_midpoints, height=wt_densities, width=wt_widths,
              align='center', edgecolor='black', color='skyblue', alpha=0.9, label='WT MIC Intervals')
    ax_wt.plot(wt_x_values, wt_fitted_curve, color='darkBlue', linewidth=2, 
               label=f'WT Fitted Curve\n(mu={mu_hat:.2f}, sigma={sigma_hat:.2f})')
    if z_99 is not None:
        ax_wt.axvline(x=z_99, color='red', linestyle='--', linewidth=2, label='99th Percentile (log2(ECOFF))')
    ax_wt.set_xlabel('log2(MIC)', fontsize=12)
    ax_wt.set_ylabel('Density', fontsize=12)
    ax_wt.legend(frameon=False)
    ax_wt.spines['top'].set_visible(False)
    ax_wt.spines['right'].set_visible(False)
    ax_wt.set_title('WT MIC Distribution')
    ax_wt.set_xlim([x_min, x_max])

    plt.tight_layout()
    plt.show()