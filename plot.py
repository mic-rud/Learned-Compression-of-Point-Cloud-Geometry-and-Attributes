import os
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import pandas as pd
import numpy as np

from plot import style

# Runs
path = "./results"
plots = "./plot/figures"
metrics = ["pcqm", "sym_y_psnr", "sym_p2p_psnr"]
related_work = ["YOGA"]

runs = {
    #"L2_proj" : "Final_L2_200epochs_SC_2_project",

    "L2" : "Final_L2_200epochs_SC_2",
    "SSIM" : "Final_SSIM_200_quadratic",
    "YOGA" : "YOGA",
    "G-PCC" : "G-PCC",
    "V-PCC" : "V-PCC",
}

bd_points = {
    "L2" : [(0,0), (0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (1.0, 1.0)],
    "SSIM" : [(0,0), (0.1, 0.1), (0.2, 0.2), (0.5, 0.5), (1.0, 1.0)],
    "G-PCC" : [(0.125,51), (0.25, 46), (0.5, 40), (0.75, 34), (0.875, 28)], #last: (0.9375, 22) 
    "V-PCC" : [(0,0), (0.1, 0.1), (0.2, 0.2), (22, 27), (22, 16)],
}
pareto_ranges = {
    "longdress":{
        "bpp": [0.0, 1], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "soldier":{
        "bpp": [0.0, 0.9], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "loot":{
        "bpp": [0.0, 0.8], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "redandblack":{
        "bpp": [0.0, 0.8], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "sarah9":{
        "bpp": [0.98, 0.9975], "pcqm": [0.98, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "david9":{
        "bpp": [0.98, 0.9975], "pcqm": [0.98, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "andrew9":{
        "bpp": [0.98, 0.9975], "pcqm": [0.98, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "phil9":{
        "bpp": [0.98, 0.9975], "pcqm": [0.98, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
}

metric_labels = {
    "pcqm" : r"$1 -$ PCQM",
    "sym_y_psnr" : r"Y-PSNR [dB]",
    "sym_yuv_psnr" : r"YUV-PSNR [dB]",
    "sym_p2p_psnr" : r"D1-PSNR [dB]",
}

run_colors = {
    "L2" : style.colors[0],
    "SSIM" : style.colors[1],
    "G-PCC" : style.colors[2],
    "V-PCC" : style.colors[3],
    "YOGA" : style.colors[4],
}
linestyles = {
    "L2" : style.linestyles[0],
    "SSIM" : style.linestyles[1],
    "G-PCC" : style.linestyles[2],
    "V-PCC" : style.linestyles[3],
    "YOGA" : style.linestyles[4],
}
labels = {
    "L2" : "Ours ($L_2$)",
    "SSIM" : "Ours ($SSIM$)",
    "G-PCC" : "G-PCC",# (tmc13 v23)",
    "V-PCC" : "V-PCC", #(tmc2 v24)",
    "YOGA" : "YOGA",
}
def plot_experiments():
    """
    Level 0 : Plot all results
    """
    data = load_csvs()

    # Plot All data separately
    pareto_data = {}
    for key, dataframe in data.items():
        pareto_df = plot_per_run_results(dataframe, key)
        pareto_data[key] = pareto_df

    plot_all_results(dataframe, pareto_data)


def plot_per_run_results(dataframe, key):
    """
    Level 1 : Plot per run results
    """
    
    # Generate the path
    directory = os.path.join(plots, key)
    if not os.path.exists(directory):
        os.mkdir(directory)

    # Filter df for pareto fron
    pareto_df = get_pareto_df(dataframe)

    if key in related_work:
        return pareto_df

    plot_pareto_figs_single(pareto_df, key)
    plot_settings(dataframe, pareto_df, key)

    return pareto_df

    
def plot_all_results(dataframe, pareto_dataframe):
    """
    Level 1 : Plot per run results
    """
    # Plot rd-curves
    plot_rd_figs_all(dataframe)
    plot_pareto_figs_all(pareto_dataframe)


def plot_settings(dataframe, pareto_dataframe, key):
    metrics = ["pcqm", "bpp", "sym_y_psnr", "sym_p2p_psnr"]
    for sequence in dataframe["sequence"].unique():
        df = dataframe[dataframe["sequence"]== sequence].sort_values(by=["q_a", "q_g"])
        pareto_df = pareto_dataframe[pareto_dataframe["sequence"]== sequence]

        x = df["q_a"].values
        y = df["q_g"].values
        X, Y = np.meshgrid(np.linspace(x.min(), x.max(), len(x)), np.linspace(y.min(), y.max(), len(y)))

        for metric in metrics:
            z = df[metric].values
            z_interp = griddata((x, y), z, (X,Y), method="linear")

            fig = plt.figure()
            ax = fig.add_subplot(121, projection="3d")
            ax.plot_surface(X, Y, z_interp)
            ax.set_xlabel("q_a")
            ax.set_ylabel("q_g")
            ax.set_zlabel(metric)
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)


            ax = fig.add_subplot(122)
            ax.contourf(X, Y, z_interp)
            ax.plot(pareto_df["q_a"], pareto_df["q_g"], color="red", marker="x")
            ax.set_xlabel("q_a")
            ax.set_ylabel("q_g")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)


            fig.tight_layout()
            path = os.path.join(plots, key, "single_{}_{}.pdf".format(metric, sequence))
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)


def plot_pareto_figs_single(dataframe, key):
    for sequence in dataframe["sequence"].unique():
        df = dataframe[dataframe["sequence"]== sequence]
        bpp = df["bpp"].values

        for metric in metrics:
            y = df[metric].values

            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(bpp, y)
            ax.set_xlabel("bpp")
            ax.set_ylabel(metric)
            ax.grid(visible=True)
            ax.tick_params(axis='both', which='major', labelsize=18)

            fig.tight_layout()
            path = os.path.join(plots, key, "rd-pareto_{}_{}.pdf".format(metric, sequence))
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)


def plot_pareto_figs_all(pareto_dataframe):
    """
    All figures as used in the publication
    """
    for metric in metrics:
        figs = {}

        for method, df in pareto_dataframe.items():
            for sequence in df["sequence"].unique():
                # Prepare figure
                if sequence in figs.keys():
                    fig, ax = figs[sequence]
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    figs[sequence] = (fig, ax)

                data = df[df["sequence"]== sequence]

                # Filter data
                (bpp_min, bpp_max) = pareto_ranges[sequence]["bpp"]
                (y_min, y_max) = pareto_ranges[sequence][metric]
                filtered_data = data[(data['bpp'] >= bpp_min) & (data['bpp'] <= bpp_max) & (data[metric] >= y_min) & (data[metric] <= y_max)]
                bpp = filtered_data["bpp"]
                y = filtered_data[metric]

                ax.plot(bpp, y, 
                        label=labels[method],
                        linestyle=linestyles[method],
                        linewidth=3,
                        color=run_colors[method])
                ax.set_xlabel(r"bpp")
                ax.set_ylabel(metric_labels[metric])
                ax.tick_params(axis='both', which='major', labelsize=14)
                #if metric in metrics.keys():
                    #ax.set_ylim(metrics[metric][0], y_lims[metric][1])

        for key, items in figs.items():
            fig, ax = items
            ax.legend()
            ax.grid(visible=True)
            fig.tight_layout()
            path = os.path.join(plots, "all", "rd-pareto_{}_{}.pdf".format(metric, key))
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        

def plot_rd_figs_all(dataframes):
    """
    All figures as used in the publication
    """
    for metric in metrics:
        figs = {}

        for method, df in dataframes.items():
            if sequence == "YOGA":
                continue # YOGA has no configs
            for sequence in df["sequence"].unique():
                # Prepare figure
                if sequence in figs.keys():
                    fig, ax = figs[sequence]
                else:
                    fig = plt.figure()
                    ax = fig.add_subplot(111)
                    figs[sequence] = (fig, ax)

                data = df[df["sequence"]== sequence]

                # Filter by settings
                tolerance = 1e-4
                settings = bd_points["method"]
                mask = np.logical_not(np.logical_or.reduce([
                    np.logical_and(np.isclose(df['q_a'], q_a_test, atol=tolerance), np.isclose(df['q_g'], q_g_test, atol=tolerance)) for q_a_test, q_g_test in settings
                ]))

                filtered_data = data[mask]
                print(filtered_data)
                bpp = filtered_data["bpp"]
                y = filtered_data[metric]

                ax.plot(bpp, y, 
                        label=labels[method],
                        linestyle=linestyles[method],
                        linewidth=3,
                        color=run_colors[method])
                ax.set_xlabel(r"bpp")
                ax.set_ylabel(metric_labels[metric])
                ax.tick_params(axis='both', which='major', labelsize=14)
                #if metric in metrics.keys():
                    #ax.set_ylim(metrics[metric][0], y_lims[metric][1])

        for key, items in figs.items():
            fig, ax = items
            ax.legend()
            ax.grid(visible=True)
            fig.tight_layout()
            path = os.path.join(plots, "all", "rd-pareto_{}_{}.pdf".format(metric, key))
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)




def get_pareto_df(dataframe):
    pareto_dataframe = pd.DataFrame()
    for sequence in dataframe["sequence"].unique():
        df = dataframe[dataframe["sequence"]== sequence]
        df = df.sort_values(by=["bpp"])

        pareto_front = []
        pareto_pcqm = 0

        # Iterate through the sorted DataFrame
        for index, row in df.iterrows():
            pcqm = row['pcqm']
            #bpp = row['bpp']
    
            if pcqm >= pareto_pcqm:
                pareto_pcqm = pcqm
                pareto_front.append(index)

        # Create a new DataFrame for the Pareto front
        pareto_df = df.loc[pareto_front]
        pareto_dataframe = pd.concat([pareto_dataframe, pareto_df])
    return pareto_dataframe

def load_csvs():
    data = {}
    for key, run in runs.items():
        data_path = os.path.join(path, run, "test.csv")
        data[key] = pd.read_csv(data_path)

        # Preprocessing
        data[key]["pcqm"] = 1 - data[key]["pcqm"]

    return data

if __name__ == "__main__":
    plot_experiments()

