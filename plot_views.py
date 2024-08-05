import os
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
import pandas as pd
import numpy as np

from plot import style

# Runs
path = "./results"
plots = "./plot/figures"
metrics = ["ssim", "psnr"]
related_work = ["YOGA"]

runs = {
    "L2" : "Final_L2_200epochs_SC_2",
    "G-PCC" : "G-PCC",
    "V-PCC" : "V-PCC",
}


metric_labels = {
    "bpp" : r"bpp",
    "ssim" : r"SSIM",
    "psnr" : r"PSNR [dB]",
}

run_colors = {
    "L2" : style.colors[0],
    "L2_view" : style.colors[3],
    "G-PCC" : style.colors[2],
    "V-PCC" : style.colors[1],
}
linestyles = {
    "L2" : style.linestyles[0],
    "G-PCC" : style.linestyles[2],
    "V-PCC" : style.linestyles[1],
    "L2_view" : style.linestyles[3],
}
markers = {
    "L2" : style.markers[0],
    "G-PCC" : style.markers[2],
    "V-PCC" : style.markers[1],
    "L2_view" : style.markers[3],
}
labels = {
    "L2" : "Ours",
    "L2_view" : "Ours (View)",
    "G-PCC" : "G-PCC",# (tmc13 v23)",
    "V-PCC" : "V-PCC", #(tmc2 v24)",
}
our_keys = ["uniform", "view"]

def plot_experiments():
    """
    Level 0 : Plot all results
    """
    data = load_csvs()

    sequences = data["L2"]["sequence"].unique()
    print(sequences)

    for sequence in sequences:
        for metric in metrics:
            fig = plt.figure()
            ax = fig.add_subplot(111)

            for method, df in data.items():
                df = df[df["sequence"] == sequence]
                for key in our_keys:
                    filtered_df = df[df["key"] == key]
                    if len(filtered_df) == 0:
                        continue

                    bpp = filtered_df["bpp"]
                    y = filtered_df[metric]

                    bjonte_model = Bjontegaard_Model(bpp, y)
                    x_scat, y_scat, x_dat, y_dat = bjonte_model.get_plot_data()

                    if method == "L2":
                        method = "L2" if key == "uniform" else "L2_view"

                    y_lims = [0.9, 0.97] if metric == "ssim" else [32, 40]


                    ax.plot(bpp, y, 
                            label=labels[method],
                            linestyle=linestyles[method],
                            marker=markers[method],
                            linewidth=3,
                            color=run_colors[method])
                    """
                    ax.plot(x_dat, y_dat, 
                            label=labels[method],
                            linestyle=linestyles[method],
                            linewidth=3,
                            color=run_colors[method])
                    ax.scatter(x_scat, y_scat, 
                            s=40,
                            marker=markers[method],
                            color=run_colors[method])
                    """
                    ax.set_ylim(y_lims)
                    ax.set_xlabel(r"bpp")
                    ax.set_ylabel(metric_labels[metric])
                    ax.tick_params(axis='both', which='major', labelsize=20)

            # finish plot
            ax.legend(fontsize=18, bbox_to_anchor=(.54, 0.43))
            ax.grid(visible=True)
            fig.tight_layout()

            folder = os.path.join(plots, "views")
            path = os.path.join(plots, "views", "rd-config_{}_{}.pdf".format(metric, sequence))
            if not os.path.exists(folder):
                os.mkdir(folder)

            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)





def load_csvs():
    data = {}
    for key, run in runs.items():
        data_path = os.path.join(path, run, "view_dep.csv")
        data[key] = pd.read_csv(data_path)

    return data

if __name__ == "__main__":
    plot_experiments()

