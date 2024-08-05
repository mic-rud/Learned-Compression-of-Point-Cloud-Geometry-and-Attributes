import os
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy.interpolate import griddata
import scipy.stats as st
import pandas as pd
import numpy as np

from plot import style

# Runs
path = "./results"
plots = "./plot/figures"
metrics = ["pcqm", "sym_y_psnr", "sym_p2p_psnr", "sym_yuv_psnr"]
related_work = ["YOGA"]
top = .97
bottom = .16
left = .22
right = .97
runs = {
    #"L2_proj" : "Final_L2_200epochs_SC_2_project",
    #"SSIM" : "Final_SSIM_200_quadratic",
    "L2" : "Final_L2_200epochs_SC_2",
    "YOGA" : "YOGA",
    "G-PCC" : "G-PCC",
    "V-PCC" : "V-PCC",
    #"L2_log" : "Ablation_L2_200epochs_SC_log_q_map",
}

bd_points = {
    #"L2" : [(0,0), (0.1, 0.2), (0.3, 0.4), (0.4, 0.8)],
    "L2" : [(0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)],
    "L2_log" : [(0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)],
    #"L2_proj" : [(0,0), (0.1, 0.2), (0.2, 0.4), (0.5, 1.0)],
    "SSIM" : [(0,0), (0.1, 0.1),(0.5, 0.5), (1.0, 1.0)],
    "G-PCC" : [(0.125, 51), (0.25, 46), (0.5, 40), (0.75, 34)], #last: (0.9375, 22) 
    "V-PCC" : [(32,42), (28, 37), (24, 32), (20, 27), (16, 22)],
}
pareto_ranges = {
    "longdress":{
        "bpp": [0.0, 1], "pcqm": [0.985, 0.9975], "sym_y_psnr": [22, 30], "sym_yuv_psnr": [22, 1.00], "sym_p2p_psnr": [60, 70],
    },
    "soldier":{
        "bpp": [0.0, 0.9], "pcqm": [0.985, 0.9975], "sym_y_psnr": [22, 35], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [60, 70],
    },
    "loot":{
        "bpp": [0.0, 0.8], "pcqm": [0.985, 0.9975], "sym_y_psnr": [24, 36], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [60, 70],
    },
    "redandblack":{
        "bpp": [0.0, 0.8], "pcqm": [0.985, 0.9975], "sym_y_psnr": [24, 32], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [60, 70],
    },
    "sarah9":{
        "bpp": [0.0, 0.9975], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "david9":{
        "bpp": [0.0, 0.9975], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "andrew9":{
        "bpp": [0.0, 0.9975], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
    "phil9":{
        "bpp": [0.0, 0.9975], "pcqm": [0.985, 0.9975], "sym_y_psnr": [0.98, 1.00], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [0.98, 1.00],
    },
}

metric_labels = {
    "bpp" : r"bpp",
    "pcqm" : r"$1 -$ PCQM",
    "sym_y_psnr" : r"Y-PSNR [dB]",
    "sym_yuv_psnr" : r"YUV-PSNR [dB]",
    "sym_p2p_psnr" : r"D1-PSNR [dB]",
}

run_colors = {
    "L2" : style.colors[0],
    "G-PCC" : style.colors[2],
    "V-PCC" : style.colors[1],
    "YOGA" : style.colors[4],

    "L2_log" : style.colors[1],
    "SSIM" : style.colors[1],
}
linestyles = {
    "L2" : style.linestyles[0],
    "G-PCC" : style.linestyles[2],
    "V-PCC" : style.linestyles[1],
    "YOGA" : style.linestyles[4],

    "L2_log" : style.linestyles[1],
    "SSIM" : style.linestyles[1],
}
markers = {
    "L2" : style.markers[0],
    "G-PCC" : style.markers[2],
    "V-PCC" : style.markers[1],
    "YOGA" : style.markers[4],

    "L2_log" : style.markers[1],
    "SSIM" : style.markers[1],
}
labels = {
    "L2" : "Ours",
    "L2_log" : "L2 log",
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

    plot_rd_figs_all(data)
    compute_bd_deltas(data)
    
    # Timing
    compute_times(data)

    # Plot All data separately
    pareto_data = {}
    for key, dataframe in data.items():
        pareto_df = plot_per_run_results(dataframe, key)
        pareto_data[key] = pareto_df

    plot_all_results(data, pareto_data)


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

            fig = plt.figure(figsize=(4, 3))
            #ax = fig.add_subplot(111, projection="3d")
            #ax.plot_surface(X, Y, z_interp)
            #ax.set_xlabel("q_a")
            #ax.set_ylabel("q_g")
            #ax.set_zlabel(metric)
            #ax.set_ylim(0, 1)
            #ax.set_xlim(0, 1)

            ax = fig.add_subplot(111)
            ranges = {
                "bpp": [0.0, 1], "pcqm": [0.985, 0.9975], "sym_y_psnr": [22, 38], "sym_yuv_psnr": [0.98, 1.00], "sym_p2p_psnr": [62, 78],
            }
            num_levels = {"bpp": 0.1, "pcqm": 0.001, "sym_yuv_psnr": 5, "sym_y_psnr": 2, "sym_p2p_psnr": 2}
            num_levels_bar = {"bpp": 0.2, "pcqm": 0.001, "sym_yuv_psnr": 5, "sym_y_psnr": 4, "sym_p2p_psnr": 4}
            min, max = ranges[metric]
            step = num_levels[metric]
            bar_step = num_levels_bar[metric]
            levels = np.arange(min, max+step, step)
            bar_levels = np.arange(min, max+bar_step, bar_step)

            #cs = ax.contour(X, Y, z_interp, 10, linestyles=":", colors="grey")
            #cs2 = ax.contourf(X, Y, z_interp, 10, cmap=cm.summer)
            cs = ax.contour(X, Y, z_interp, 10, levels=levels, linestyles=":", colors="grey")
            cs2 = ax.contourf(X, Y, z_interp, 10, levels=levels, cmap=cm.summer)
            ax.plot(pareto_df["q_a"], pareto_df["q_g"], color=run_colors[key], marker="o", label=labels[key])

            ax.set_xlabel(r"$q^{(A)}$")
            ax.set_ylabel(r"$q^{(G)}$", rotation=0, ha="right", va="center")
            ax.set_ylim(0, 1)
            ax.set_xlim(0, 1)
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            #ax.legend(fontsize=16)
            ax.xaxis.set_label_coords(0.5, -0.05)
            ax.yaxis.set_label_coords(-0.05, 0.5)

            cbar = fig.colorbar(cs2, boundaries=levels, ticks=bar_levels)
            cbar.ax.set_ylabel(metric_labels[metric])
            
            ticklabels = 18
            ax.tick_params(axis='both', which='major', labelsize=ticklabels)
            cbar.ax.tick_params(axis='both', which='major', labelsize=ticklabels)


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
            ax.tick_params(axis='both', which='major', labelsize=22)

            path = os.path.join(plots, key, "rd-pareto_{}_{}.pdf".format(metric, sequence))
            #fig.tight_layout()
            fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            fig.savefig(path)
            #fig.savefig(path, bbox_inches="tight")
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
                ax.tick_params(axis='both', which='major', labelsize=22)

        for key, items in figs.items():
            fig, ax = items
            ax.legend()
            ax.grid(visible=True)
            #fig.tight_layout()
            path = os.path.join(plots, "all", "rd-pareto_{}_{}.pdf".format(metric, key))

            fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            fig.savefig(path)
            #fig.savefig(path, bbox_inches="tight")
            plt.close(fig)
        
def filter_config_points(data, config):
    tolerance = 1e-2

    mask = np.full(len(data), False)
    for q_g_test, q_a_test in config:
        # Create a mask for the current tuple
        tuple_mask = np.logical_and(
            np.isclose(data['q_a'], q_a_test, atol=tolerance),
            np.isclose(data['q_g'], q_g_test, atol=tolerance)
        )
        mask = np.logical_or(mask, tuple_mask)

    filtered_data = data[mask]
    return filtered_data

def plot_rd_figs_all(dataframes):
    """
    All figures as used in the publication
    """
    for metric in metrics:
        figs = {}

        # Loop through results
        for method, df in dataframes.items():
            if method == "YOGA":
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
                settings = bd_points[method]
                filtered_data = filter_config_points(data, settings)

                bpp = filtered_data["bpp"]
                y = filtered_data[metric]

                bjonte_model = Bjontegaard_Model(bpp, y)
                x_scat, y_scat, x_dat, y_dat = bjonte_model.get_plot_data()

                ax.plot(x_dat, y_dat, 
                        label=labels[method],
                        linestyle=linestyles[method],
                        linewidth=3,
                        color=run_colors[method])
                ax.scatter(x_scat, y_scat, 
                        s=40,
                        marker=markers[method],
                        color=run_colors[method])
                ax.set_xlabel(r"bpp")
                ax.set_ylabel(metric_labels[metric])
                ax.tick_params(axis='both', which='major', labelsize=22)
                """
                if metric == "pcqm":
                    ax.set_ylim([0.98, 0.9975])
                if metric == "sym_y_psnr" and sequence == "andrew9":
                    ax.set_ylim([16, 32])
                """
                

        for key, items in figs.items():
            fig, ax = items
            ax.legend()
            ax.grid(visible=True)
            path = os.path.join(plots, "all", "rd-config_{}_{}.pdf".format(metric, key))
            fig.subplots_adjust(bottom=bottom, top=top, left=left, right=right)
            fig.savefig(path)

            #fig.tight_layout()
            #fig.savefig(path, bbox_inches="tight")
            plt.close(fig)


def compute_bd_deltas(dataframes):
    ref = "G-PCC"
    test = "L2"
    for metric in metrics:
        print(metric)
        # Get G-PCC config for BD Points
        ref_data = dataframes[ref]
        test_data = dataframes[test]
        for sequence in ref_data["sequence"].unique():
            ref_df = ref_data[ref_data["sequence"]== sequence]
            ref_settings = bd_points[ref]
            filtered_ref = filter_config_points(ref_df, ref_settings)

            test_df = test_data[test_data["sequence"]== sequence]
            test_settings = bd_points[test]
            filtered_test = filter_config_points(test_df, test_settings)

            bpp = filtered_ref["bpp"]
            y = filtered_ref[metric]
            ref_model = Bjontegaard_Model(bpp, y)

            bpp = filtered_test["bpp"]
            y = filtered_test[metric]
            test_model = Bjontegaard_Model(bpp, y)

            delta = Bjontegaard_Delta()
            psnr_delta = delta.compute_BD_PSNR(ref_model, test_model)
            rate_delta = delta.compute_BD_Rate(ref_model, test_model)

            print("Sequence: {:<10} \t\tPSNR-Delta: {:.6} \tRate-Delta: {:.4}".format(sequence, psnr_delta, rate_delta))




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


def compute_times(data):
    # Computes the times
    summary_data = []
    for key, results in data.items():
        if key == "YOGA":
            continue

        for sequence in results["sequence"].unique():
            test_sequences = ["loot", "longdress", "soldier", "redandblack"]
            if not sequence in test_sequences:
                continue
            

            if key == "G-PCC":
                #process per rate
                rates = [0.125, 0.25, 0.5, 0.75]
                for rate in rates:
                    t_compress = results[(results["sequence"] == sequence) & (results["q_g"] == rate)]["t_compress"]
                    t_decompress = results[(results["sequence"] == sequence) & (results["q_g"] == rate)]["t_decompress"]

                    conf_compress = st.t.interval(0.95, len(t_compress-1), loc=np.mean(t_compress), scale=st.sem(t_compress))
                    conf_decompress = st.t.interval(0.95, len(t_decompress-1), loc=np.mean(t_decompress), scale=st.sem(t_decompress))

                    summary_data.append([key, sequence, rate, np.mean(t_compress), np.mean(t_compress) - conf_compress[0], np.mean(t_decompress), np.mean(t_decompress) - conf_decompress[0]])
            else:
                # process all
                t_compress = results[results["sequence"] == sequence]["t_compress"]
                t_decompress = results[results["sequence"] == sequence]["t_decompress"]
                conf_compress = st.t.interval(0.95, len(t_compress-1), loc=np.mean(t_compress), scale=st.sem(t_compress))
                conf_decompress = st.t.interval(0.95, len(t_decompress-1), loc=np.mean(t_decompress), scale=st.sem(t_decompress))

                summary_data.append([key, sequence, None, np.mean(t_compress), np.mean(t_compress) - conf_compress[0], np.mean(t_decompress), np.mean(t_decompress) - conf_decompress[0]])


        # Calculate per sequence mean per key and rate
        results = results[results["sequence"].isin(test_sequences)]
        if key == "G-PCC":
            rates = [0.125, 0.25, 0.5, 0.75]
            for rate in rates:
                    t_compress_seq_rate = results[results["q_g"] == rate]["t_compress"]
                    t_decompress_seq_rate = results[results["q_g"] == rate]["t_decompress"]

                    mean_t_compress_seq_rate = np.mean(t_compress_seq_rate)
                    mean_t_decompress_seq_rate = np.mean(t_decompress_seq_rate)

                    summary_data.append([key, "combined", rate, mean_t_compress_seq_rate, np.nan, mean_t_decompress_seq_rate, np.nan])

        t_compress_seq = results["t_compress"]
        t_decompress_seq = results["t_decompress"]

        mean_t_compress_seq = np.mean(t_compress_seq)
        mean_t_decompress_seq = np.mean(t_decompress_seq)

        summary_data.append([key, "combined", None, mean_t_compress_seq, np.nan , mean_t_decompress_seq, np.nan])

    summary_df = pd.DataFrame(summary_data, columns=["key", "sequence", "rate", "mean_t_compress", "conf_compress", "mean_t_decompress", "conf_decompress"])

    print(summary_df)
if __name__ == "__main__":
    plot_experiments()

