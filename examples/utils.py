import json
import piezo
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
    fig, ax_wt = plt.subplots(1, 1, figsize=(7, 5))

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


def confusion_matrix(labels, predictions, classes):
    """
    Creates a confusion matrix for given labels and predictions with specified classes.

    Parameters:
    labels (list): Actual labels.
    predictions (list): Predicted labels.
    classes (list): List of all classes.

    Returns:
    np.ndarray: Confusion matrix.
    """
    cm = np.zeros((len(classes), len(classes)), dtype=int)
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    for label, prediction in zip(labels, predictions):
        if label in class_to_index and prediction in class_to_index:
            cm[class_to_index[label], class_to_index[prediction]] += 1

    return cm

def piezo_predict(iso_df, catalogue_file, drug, U_to_R=False, U_to_S=False, Print=True):
    """
    Predicts drug resistance based on genetic mutations using a resistance catalogue.

    Parameters:
    iso_df (pd.DataFrame): DataFrame containing isolate data with UNIQUEID, PHENOTYPE, and GENE_MUT columns.
    catalogue_file (str): Path to the resistance catalogue file.
    drug (str): The drug for which resistance predictions are to be made.
    U_to_R (bool, optional): If True, treat 'U' predictions as 'R'. Defaults to False.
    U_to_S (bool, optional): If True, treat 'U' predictions as 'S'. Defaults to False.
    Print (bool, optional): If True, prints the confusion matrix, coverage, sensitivity, and specificity. Defaults to True.

    Returns:
    list: Confusion matrix, isolate coverage, sensitivity, specificity, and false negative IDs.
    """
    # Load and parse the catalogue with piezo
    catalogue = piezo.ResistanceCatalogue(catalogue_file)

    # Ensure the UNIQUEID and PHENOTYPE columns are used correctly
    ids = iso_df['UNIQUEID'].unique().tolist()
    labels = iso_df.groupby('UNIQUEID')['PHENOTYPE'].first().reindex(ids).tolist()
    predictions = []

    for id_ in ids:
        # For each sample
        df = iso_df[iso_df['UNIQUEID'] == id_]
        # Predict phenotypes for each mutation via lookup
        mut_predictions = []
        for var in df['MUTATION']:
            if pd.isna(var):
                predict = 'S'
            else:
                predict = catalogue.predict(var)
            if isinstance(predict, dict):
                mut_predictions.append(predict[drug])
            else:
                mut_predictions.append(predict)

        # Make sample-level prediction from mutation-level predictions. R > U > S
        if "R" in mut_predictions:
            predictions.append("R")
        elif "U" in mut_predictions:
            if U_to_R:
                predictions.append("R")
            elif U_to_S:
                predictions.append("S")
            else:
                predictions.append("U")
        else:
            predictions.append("S")

    # Log false negative samples
    FN_id = [
        id_
        for id_, label, pred in zip(ids, labels, predictions)
        if pred == "S" and label == "R"
    ]

    FP_id = [
        id_
        for id_, label, pred in zip(ids, labels, predictions)
        if pred == "R" and label == "S"
    ]

    # Generate confusion matrix for performance analysis
    cm = confusion_matrix(labels, predictions, classes=["R", "S", "U"])

    if "U" not in predictions:
        cm = cm[:2, :2]
    else:
        cm = cm[:2, :]

    if Print:
        print(cm)
    
    # Calculate performance metrics
    sensitivity = cm[0, 0] / (cm[0, 0] + cm[0, 1])
    specificity = cm[1, 1] / (cm[1, 1] + cm[1, 0])
    isolate_cov = (len(labels) - predictions.count("U")) / len(labels)

    if Print:
        print("Catalogue coverage of isolates:", isolate_cov)
        print("Sensitivity:", sensitivity)
        print("Specificity:", specificity)

    return [cm, isolate_cov, sensitivity, specificity, FN_id, FP_id]


def plot_truthtables(truth_table, U_to_S=False, fontsize=10, colors=None, save=None):
    """
    Plots a truth table as a confusion matrix to denote each cell with perfect squares or proportional rectangles.

    Parameters:
    truth_table (pd.DataFrame): DataFrame containing the truth table values.
                                The DataFrame should have the following structure:
                                - Rows: True labels ("R" and "S")
                                - Columns: Predicted labels ("R", "S", and optionally "U")
    U_to_S (bool): Whether to separate the "U" values from the "S" column. If True,
                   an additional column for "U" values will be used.
    fontsize (int): Font size for the text in the plot.
    colors (list): List of four colors for the squares.
                   Defaults to red and green for the diagonal, pink and green for the off-diagonal.

    Returns:
    None
    """

    # Default colors if none provided
    if colors is None:
        if U_to_S:
            colors = ["#e41a1c", "#4daf4a", "#fc9272", "#4daf4a"]
        else:
            colors = ["#e41a1c", "#4daf4a", "#fc9272", "#4daf4a", "#4daf4a", "#4daf4a"]

    # Determine the number of columns for U_to_S condition
    num_columns = 3 if not U_to_S else 2
    num_rows = 2

    # Adjust the figure size to ensure square cells
    figsize = (
        (num_columns / 1.8, num_rows / 1.8)
        if num_columns == 2
        else (num_columns * 1.5 / 1.8, num_rows / 1.8)
    )

    fig = plt.figure(figsize=figsize)
    axes = plt.gca()

    if not U_to_S:
        assert (
            len(colors) == 6
        ), "The length of supplied colors must be 6, one for each cell"
        axes.add_patch(Rectangle((2, 0), 1, 1, fc=colors[4], alpha=0.5))
        axes.add_patch(Rectangle((2, 1), 1, 1, fc=colors[5], alpha=0.5))

        axes.set_xlim([0, 3])
        axes.set_xticks([0.5, 1.5, 2.5])
        axes.set_xticklabels(["S", "R", "U"], fontsize=9)
    else:
        assert (
            len(colors) == 4
        ), "The length of supplied colors must be 4, one for each cell"
        axes.set_xlim([0, 2])
        axes.set_xticks([0.5, 1.5])
        axes.set_xticklabels(["S+U", "R"], fontsize=9)

    # Apply provided colors for the squares
    axes.add_patch(Rectangle((0, 0), 1, 1, fc=colors[0], alpha=0.8))
    axes.add_patch(Rectangle((1, 0), 1, 1, fc=colors[1], alpha=0.8))
    axes.add_patch(Rectangle((1, 1), 1, 1, fc=colors[2], alpha=0.8))
    axes.add_patch(Rectangle((0, 1), 1, 1, fc=colors[3], alpha=0.8))

    axes.set_ylim([0, 2])
    axes.set_yticks([0.5, 1.5])
    axes.set_yticklabels(["R", "S"], fontsize=9)

    # Add text to the plot
    axes.text(
        1.5,
        0.5,
        int(truth_table["R"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        1.5,
        1.5,
        int(truth_table["R"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        1.5,
        int(truth_table["S"]["S"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )
    axes.text(
        0.5,
        0.5,
        int(truth_table["S"]["R"]),
        ha="center",
        va="center",
        fontsize=fontsize,
    )

    if not U_to_S:
        axes.text(
            2.5,
            0.5,
            int(truth_table["U"]["R"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )
        axes.text(
            2.5,
            1.5,
            int(truth_table["U"]["S"]),
            ha="center",
            va="center",
            fontsize=fontsize,
        )

    axes.set_aspect("equal")  # Ensure squares remain squares

    if save != None:
        plt.savefig(save, format="pdf", bbox_inches="tight")

    plt.show()

def FRS_vs_metric(df, cov=True):
    """
    Plots a comparison of performance metrics (Sensitivity, Specificity, and optionally Coverage)
    against Fraction Read Support (FRS).

    Parameters:
    df (pandas.DataFrame): DataFrame containing the performance metrics with columns "FRS",
                           "Sensitivity", "Specificity", and optionally "Coverage".
    cov (bool): If True, includes Coverage in the plot. Defaults to True.

    Returns:
    None

    """
    plt.figure(figsize=(6.69, 2))

    # Plot Sensitivity and Specificity
    sns.lineplot(
        x="FRS", y="Sensitivity", data=df, label="Sensitivity", color="#e41a1c"
    )
    sns.lineplot(
        x="FRS", y="Specificity", data=df, label="Specificity", color="#377eb8"
    )

    # Plot Coverage if specified
    if cov:
        sns.lineplot(
            x="FRS", y="Coverage", data=df, label="Isolate Coverage", color="green"
        )

    # Set x and y ticks
    yticks = [0, 20, 40, 60, 80, 100]
    xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    plt.yticks(yticks)
    plt.xticks(xticks)

    # Add labels and legend
    plt.xlabel("Fraction Read Support (FRS)")
    plt.ylabel("%")
    plt.legend(loc="best", frameon=False, bbox_to_anchor=(0.85, 0.40))

    # Annotate the start and end values
    for line in plt.gca().lines:
        x_data = line.get_xdata()
        y_data = line.get_ydata()
        start_value = y_data[0]
        final_value = y_data[-1]
        plt.annotate(
            f"{start_value:.1f}",
            (x_data[0], start_value),
            textcoords="offset points",
            xytext=(-17, -3),
            ha="center",
        )
        plt.annotate(
            f"{final_value:.1f}",
            (x_data[-1], final_value),
            textcoords="offset points",
            xytext=(17, -3),
            ha="center",
        )

    # Add vertical lines and text annotations
    plt.axvline(x=0.75, color="gray", linestyle="--", label="FRS=0.75")

    plt.axvline(x=0.25, color="gray", linestyle="--", label="FRS=0.25")

    # Despine and grid settings
    sns.despine(top=True, right=True)
    plt.grid(False)
    plt.tight_layout()
    # Show plot
    plt.ylim(40, 100)
    plt.show()


def plot_2_distributions(y_low_log, y_high_log, mu_hat, sigma_hat, z_99,
                         y_low_log_2=None, y_high_log_2=None, mu_hat_2=None, sigma_hat_2=None, z_99_2=None, lower_cap_value=-10, upper_cap_value=5):
    """
    Plot two distributions of MIC intervals along with fitted Gaussian distributions.
    
    Args:
        y_low_log (array-like): Lower bounds of the first log-transformed MIC intervals.
        y_high_log (array-like): Upper bounds of the first log-transformed MIC intervals.
        mu_hat (float): Estimated mean of the first fitted distribution.
        sigma_hat (float): Estimated standard deviation of the first fitted distribution.
        z_99 (float): 99th percentile of the first distribution in log scale.
        y_low_log_2 (array-like, optional): Lower bounds of the second log-transformed MIC intervals.
        y_high_log_2 (array-like, optional): Upper bounds of the second log-transformed MIC intervals.
        mu_hat_2 (float, optional): Estimated mean of the second fitted distribution.
        sigma_hat_2 (float, optional): Estimated standard deviation of the second fitted distribution.
        z_99_2 (float, optional): 99th percentile of the second distribution in log scale.
        cap_value (float, optional): Maximum value to cap the data for plotting. Defaults to 10.
    """
    # Cap np.inf values in both log-transformed MIC arrays
    y_low_log = np.clip(y_low_log, lower_cap_value, upper_cap_value)
    y_high_log = np.clip(y_high_log, lower_cap_value, upper_cap_value)
    if y_low_log_2 is not None and y_high_log_2 is not None:
        y_low_log_2 = np.clip(y_low_log_2, lower_cap_value, upper_cap_value)
        y_high_log_2 = np.clip(y_high_log_2, lower_cap_value, upper_cap_value)
    
    def compute_distribution(y_low, y_high, mu, sigma):
        intervals = [(low, high) for low, high in zip(y_low, y_high)]
        unique_intervals = sorted(set(intervals))
        mic_counts = [intervals.count(interval) for interval in unique_intervals]
        midpoints = [(low + high) / 2 for low, high in unique_intervals]
        widths = [high - low for low, high in unique_intervals]
        x_values = np.linspace(min(y_low), max(y_high), 1000)
        fitted_curve = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_values - mu) / sigma) ** 2) * sum(mic_counts)
        return midpoints, mic_counts, widths, x_values, fitted_curve

    # Calculate the widest x-range based on both distributions (if the second distribution is provided)
    x_min, x_max = min(y_low_log), max(y_high_log)
    if y_low_log_2 is not None and y_high_log_2 is not None:
        x_min, x_max = min(x_min, min(y_low_log_2)), max(x_max, max(y_high_log_2))

    # Compute WT distribution
    wt_midpoints, wt_counts, wt_widths, wt_x_values, wt_fitted_curve = compute_distribution(
        y_low_log, y_high_log, mu_hat, sigma_hat
    )

    # Plot WT distribution on the primary y-axis
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax1.bar(wt_midpoints, height=wt_counts, width=wt_widths, align='center', edgecolor='black', 
            color='skyblue', alpha=0.7, label='WT MIC Intervals')
    ax1.plot(wt_x_values, wt_fitted_curve, color='darkblue', linewidth=2, 
             label=f'WT Fitted Curve\n(mu={mu_hat:.2f}, sigma={sigma_hat:.2f})')
    if z_99 is not None:
        ax1.axvline(x=z_99, color='red', linestyle='--', linewidth=2, label='99th Percentile')

    # Customize primary y-axis for WT counts
    ax1.tick_params(axis='y', labelcolor='Darkblue')

    # Create a secondary y-axis for the second distribution, if provided
    if y_low_log_2 is not None and y_high_log_2 is not None and mu_hat_2 is not None and sigma_hat_2 is not None:
        ax2 = ax1.twinx()  # Create a second y-axis sharing the same x-axis

        # Compute the second distribution
        midpoints_2, counts_2, widths_2, x_values_2, fitted_curve_2 = compute_distribution(
            y_low_log_2, y_high_log_2, mu_hat_2, sigma_hat_2
        )

        # Plot the second distribution on the secondary y-axis
        ax2.bar(midpoints_2, height=counts_2, width=widths_2, align='center', edgecolor='black', 
                color='orange', alpha=0.5, label='Second MIC Intervals')
        ax2.plot(x_values_2, fitted_curve_2, color='darkorange', linewidth=2, 
                 label=f'Second Fitted Curve\n(mu={mu_hat_2:.2f}, sigma={sigma_hat_2:.2f})')
        if z_99_2 is not None:
            ax2.axvline(x=z_99_2, color='purple', linestyle='--', linewidth=2, label='99th Percentile (log2(ECOFF) - Second)')
        
        # Customize secondary y-axis for the second distribution counts
        ax2.tick_params(axis='y', labelcolor='darkorange')

    # Final plot adjustments
    ax1.set_xlabel('log2(MIC)', fontsize=12)
    ax1.legend(loc="upper left", frameon=False)
    if y_low_log_2 is not None:
        ax2.legend(loc="upper right", frameon=False)
    ax1.spines['top'].set_visible(False)
    if y_low_log_2 is not None:
        ax2.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax1.set_xlim([x_min, x_max])

    plt.tight_layout()
    plt.show()


def plot_catalogue_counts(df, figsize=(6, 2.5)):
    """
    Plots a horizontal bar chart showing the counts of R, S, and U predictions per gene.

    Parameters:
    df (pd.DataFrame): DataFrame containing columns 'MUTATION' and 'PREDICTION'.
    figsize (tuple): Figure size in the format (width, height).

    """
    # Extract the gene names
    df["GENE"] = df["MUTATION"].apply(lambda x: x.split("@")[0])

    # Count the occurrences of each prediction type per gene
    count_data = df.groupby(["GENE", "PREDICTION"]).size().unstack(fill_value=0)
    count_data.sort_values(by="R", ascending=True, inplace=True)
    colors = {"S": "#377eb8", "R": "#e41a1c", "U": "#aaaaaa"}

    # Plot the chart
    fig, ax = plt.subplots(figsize=figsize)
    bars = count_data.plot(
        kind="barh",
        stacked=True,
        color=[colors["S"], colors["R"], colors["U"]],
        edgecolor="none",
        width=0.8,
        alpha=0.8,
        ax=ax,
    )

    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    # ax.legend(frameon=False, fontsize="small")
    ax.get_legend().remove()
    ax.set_xlabel("Number of genetic variants associated with a phenotype")
    ax.set_ylabel("")
    ax.set_yticklabels(count_data.index, fontstyle="italic")
    ax.set_xticks([0, 50, 100, 150, 200])
    ax.set_xlim([0, 220])
    # ax.axes.get_yaxis().set_visible(False)

    # Add data labels for R counts
    for i, (gene, row) in enumerate(count_data.iterrows()):
        if "R" in row and row["R"] > 0:
            ax.text(
                row["R"] + 2, i, f'{row["R"]}', va="center", ha="left", color="#e41a1c"
            )
        if "S" in row and row["S"] > 0:
            ax.text(
                row["S"] + 2, i, f'{row["S"]}', va="center", ha="left", color="#377eb8"
            )
    ax.legend(frameon=False)
    plt.show()

def plot_fitted_distribution(effects, df, x_min, x_max):

    for _, row in effects.iterrows():
        mutation_name = row['Mutation']
        log2_mic = row['effect_size']  # Assuming log2(MIC) is stored in 'effect_size'
        mic = row['MIC']  # Actual MIC value in the 'MIC' column

        # Filter the DataFrame directly for the current mutation
        mutation_df = df[df['MUTATION'] == mutation_name]

        if len(mutation_df) > 3:
            # Extract the intervals directly from the DataFrame
            mutation_intervals = list(zip(mutation_df['y_low_log'], mutation_df['y_high_log']))

            # Handle np.inf by replacing high values with an arbitrarily large width
            processed_intervals = []
            for low, high in mutation_intervals:
                if high == np.inf:
                    processed_intervals.append((low, x_max))  # Cap the high value at the plot limit
                else:
                    processed_intervals.append((low, high))

            # Get unique intervals for the current mutation
            unique_intervals = sorted(set(processed_intervals))

            # Calculate counts for each unique interval
            mutation_mic_counts = [processed_intervals.count(interval) for interval in unique_intervals]

            # Extract the midpoints and widths for plotting the bars
            interval_midpoints = [
                (low + (high if high != x_max else x_max)) / 2
                for low, high in unique_intervals
            ]
            interval_widths = [
                (high - low if high != x_max else x_max - low)
                for low, high in unique_intervals
            ]

            plt.figure(figsize=(4, 2))  # Create a new figure for each mutation

            # Step 1: Plot the histogram of calculated MIC intervals for this mutation
            plt.bar(interval_midpoints, height=mutation_mic_counts, width=interval_widths,
                    align='center', edgecolor='black', color='skyblue', label='True MIC Distribution')

            plt.axvline(x=0, linestyle='--', color='orange')

            # Step 2: Overlay the fitted normal distribution for the current mutation
            x_values = np.linspace(x_min, x_max, 100)

            # Generate the normal distribution using log2(MIC) (effect size) and std
            y_values = norm.pdf(x_values, loc=log2_mic, scale=row['effect_std'])

            # Scale the normal distribution to match the height of the histogram
            y_values *= max(mutation_mic_counts) / max(y_values)

            # Plot the fitted curve
            plt.plot(x_values, y_values, label=f'Fitted Curve for {mutation_name}', linestyle='-', color='red')

            # Add text annotation for log2(MIC) and MIC
            annotation_text = f"log2(MIC): {log2_mic:.2f}\nMIC: {mic:.2f}"
            plt.text(x_min + 0.5, max(mutation_mic_counts) * 0.8, annotation_text, fontsize=8, color='black',
                    bbox=dict(facecolor='white', edgecolor='white', alpha=0.7))

            # Customize the plot
            plt.xlabel('log2(MIC)')
            plt.ylabel('Counts')
            plt.title(f'{mutation_name}', fontsize=9)  # Smaller font size
            plt.xlim([x_min, x_max])  # Set the consistent x-axis range

            # Remove top and right spines
            ax = plt.gca()
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Show the plot for this mutation
            plt.show()
