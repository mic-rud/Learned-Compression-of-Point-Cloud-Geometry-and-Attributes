import os
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Settings 
matplotlib.use("template")
fontsize = 12
plt.rc("font", family="serif", size=fontsize)
plt.rc("xtick", labelsize=fontsize)
plt.rc("ytick", labelsize=fontsize)
plt.rc("axes", axisbelow=True)
plt.rc("legend", fontsize=fontsize, frameon=True, framealpha=1.0)

matplotlib.rcParams["lines.linewidth"] = 1.5
matplotlib.rcParams["lines.markersize"] = 5

linestyles = ["solid", "dashdot", "dashed", (0, (3, 1, 1,1))]
"""
colors = [
        "#d7191c", 
        "#009E73", 
        "#e69f00", 
        "#2b83ba",
        "#ffffbf"
        ]
"""

colors = {
    "Fixed-Rate" : "#003366",
    "Progressive (Ours)" : "#e31b23", #RPTH palette
    #"G-PCC": "#FFC325"
    #"G-PCC": "#005cab",
    "G-PCC (tmc13)": "#8c9ea3",
}

markers = {
    "Fixed-Rate" : "+",
    "Progressive (Ours)" : "x", 
    "G-PCC (tmc13)": "2",
}
path = "./results"
sequences = ["loot", "redandblack", "longdress", "soldier", "andrew9", "sarah9", "david9", "phil9"]
data_points = {
    "Fixed-Rate" : ["MeanScale_1_lambda100", 
                               "MeanScale_1_lambda200", 
                               "MeanScale_1_lambda400", 
                               "MeanScale_1_lambda800", 
                               "MeanScale_1_lambda1600"],
    "G-PCC (tmc13)" : "G-PCC",
    "Progressive (Ours)" : "MeanScale_5_lambda200-6400_200epochs",
    #"Progressive - Rate (Ours)" : "MeanScale_5_lambda200-3200_rateOnce",
}

cumsum = {
    "Fixed-Rate" : ["MeanScale_1_lambda100", 
                               "MeanScale_1_lambda200", 
                               "MeanScale_1_lambda400", 
                               "MeanScale_1_lambda800", 
                               "MeanScale_1_lambda1600"],
    "G-PCC (tmc13)": ["G-PCC (tmc13)"],
          }
BD_reference = "Fixed-Rate"
# Load data
data_frames = {}
BD_results = []

for key, data_point in data_points.items():
    data = pd.DataFrame()
    if isinstance(data_point, list):
        for i, d in enumerate(data_point):
            data_path = os.path.join(path, d, "test.csv")
            new_data = pd.read_csv(data_path)
            new_data["name"] = d
            new_data["layer"] = i
            data = pd.concat([data, new_data])
        
    else:
        data_path = os.path.join(path, data_point, "test.csv")
        data = pd.read_csv(data_path)
        data["name"] = key


    data_frames[key] = data

"""
for sequence in sequences:
    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence]
        
        ax.plot(data["bpp"], data["sym_y_psnr"], 
                 label=key,
                 color=colors[key],
                 marker=markers[key],)
                

    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("Y PNSR [dB]")
    ax.set_xlim(left=0.0)
    plt.tight_layout()
    fig.savefig("plot/figures/{}_Y-PSNR.pdf".format(str(sequence)), bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence]
        
        ax.plot(data["bpp"], (data["sym_y_psnr"] * 6 + data["sym_u_psnr"] + data["sym_v_psnr"]) / 8, 
                 label=key,
                 color=colors[key],
                 marker=markers[key],)
                

    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("YUV PNSR [dB]")
    ax.set_xlim(left=0.0)
    fig.savefig("plot/figures/{}_YUV-PSNR.pdf".format(str(sequence)), bbox_inches="tight")
    plt.close()


    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        data = data_frame[data_frame["sequence"] == sequence].copy()
        if key == "G-PCC (tmc13)":
            continue

        if key in cumsum.keys():
            run = cumsum[key]
            data = data[data["name"].isin(run)]
            rates = data["bpp"].cumsum()
        else:
            rates = data["bpp"]

        ax.plot(rates, data["sym_y_psnr"], 
                 label=key,
                 color=colors[key],
                 marker=markers[key],)


    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("cumulative bpp")
    ax.set_ylabel("Y PNSR [dB]")
    ax.set_xlim(left=0.0)
    fig.tight_layout()
    fig.savefig("plot/figures/{}_Y-PSNR_cumsum.pdf".format(str(sequence)), bbox_inches="tight")
    plt.close()

    ### Bjontegaard Models
    reference_data = data_frames[BD_reference].copy()
    reference_data = reference_data[reference_data["sequence"] == sequence]
    reference_data_cumulative = reference_data.copy()
    rates_cumulative = reference_data_cumulative["bpp"].cumsum()

    reference_model = Bjontegaard_Model(reference_data["bpp"], reference_data["sym_y_psnr"])
    reference_model_cumulative = Bjontegaard_Model(rates_cumulative, reference_data_cumulative["sym_y_psnr"])
    for key, data_frame in data_frames.items():
        if key == BD_reference:
            continue

        data = data_frame[data_frame["sequence"] == sequence]
        
        if key in cumsum.keys():
            run = cumsum[key]
            data = data[data["name"].isin(run)]
            rates = data["bpp"].cumsum()
        else:
            rates = data["bpp"]

        bd_model = Bjontegaard_Model(data["bpp"], data["sym_y_psnr"])
        delta = Bjontegaard_Delta()
        delta_psnr = delta.compute_BD_PSNR(reference_model, bd_model)
        delta_psnr_cum = delta.compute_BD_PSNR(reference_model_cumulative, bd_model)
        delta_rate = delta.compute_BD_Rate(reference_model, bd_model)
        delta_rate_cum = delta.compute_BD_Rate(reference_model_cumulative, bd_model)
        results = {}
        results["name"] = key
        results["sequence"] = sequence
        results["delta_psnr"] = delta_psnr
        results["delta_psnr_cum"] = delta_psnr_cum
        results["delta_rate"] = delta_rate
        results["delta_rate_cum"] = delta_rate_cum

        BD_results.append(results)

results=pd.DataFrame().from_dict(BD_results)
"""


###### PLOTS FOR DATASET

datasets = {
    "8iVFB": ["loot", "redandblack", "longdress", "soldier"],
    "MVUB": [ "andrew9", "sarah9", "david9", "phil9"]
}

left = 0.15
bottom = 0.16
right = 0.98
top = 0.97
df_per_dataset = pd.DataFrame()
bd_models = {}
bd_models_cumulated = {}
for dataset, sequences in datasets.items():
    fig, ax = plt.subplots(1,1, figsize=(4, 3))

    for key, data_frame in data_frames.items():
        # Filter by dataset
        data_frame = data_frame[data_frame["sequence"].isin(datasets[dataset])]

        rates = []
        y_psnr = []
        u_psnr = []
        v_psnr = []
        layers = data_frame["layer"].unique()
        for layer in layers:
            data = data_frame[data_frame["layer"] == layer]
            rates.append(data["bpp"].mean())
            y_psnr.append(data["sym_y_psnr"].mean())
            u_psnr.append(data["sym_u_psnr"].mean())
            v_psnr.append(data["sym_v_psnr"].mean())
        

        ax.plot(rates, y_psnr, label=key, color=colors[key], marker=markers[key],)
        bd_models[key] = Bjontegaard_Model(rates, y_psnr)

    # Save all the figs
    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("Y PNSR [dB]")
    ax.set_xlim(left=0.0)
    ax.set_ylim([24, 36])
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.savefig("plot/figures/{}_Y-PSNR.pdf".format(str(dataset)))
                
    fig, ax = plt.subplots(1,1, figsize=(4, 3))
    for key, data_frame in data_frames.items():
        # Filter by dataset
        data_frame = data_frame[data_frame["sequence"].isin(datasets[dataset])]

        rates = []
        y_psnr = []
        u_psnr = []
        v_psnr = []
        layers = data_frame["layer"].unique()
        for layer in layers:
            data = data_frame[data_frame["layer"] == layer]
            rates.append(data["bpp"].mean())
            y_psnr.append(data["sym_y_psnr"].mean())
            u_psnr.append(data["sym_u_psnr"].mean())
            v_psnr.append(data["sym_v_psnr"].mean())
        

        # Cum Sum
        if key == "G-PCC (tmc13)":
            continue

        if key in cumsum.keys():
            ax.plot(np.cumsum(rates), y_psnr, label=key, color=colors[key], marker=markers[key],)
            bd_models_cumulated[key] = Bjontegaard_Model(np.cumsum(rates), y_psnr)
        else:
            ax.plot(rates, y_psnr, label=key, color=colors[key], marker=markers[key],)
            bd_models_cumulated[key] = Bjontegaard_Model(rates, y_psnr)

    ax.legend()
    ax.grid(which="both")
    ax.set_xlabel("bpp")
    ax.set_ylabel("Y PNSR [dB]")
    ax.set_xlim(left=0.0)
    ax.set_ylim([24, 36])
    if dataset == "MVUB":
        ax.set_ylim([24, 34])
    fig.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig.savefig("plot/figures/{}_Y-PSNR_cumulated.pdf".format(str(dataset)))
    plt.close()

    fig_delta, ax_delta = plt.subplots(1,1, figsize=(4, 3))
    fig_delta_cum, ax_delta_cum = plt.subplots(1,1, figsize=(4, 3))

    # Bjontegard plots now!
    reference_model = bd_models[BD_reference]
    reference_model_cumulated = bd_models_cumulated[BD_reference]

    for key, bd_model in bd_models.items():
        Delta = Bjontegaard_Delta()
        delta_psnr = Delta.compute_BD_PSNR(reference_model, bd_model)
        delta_rate = Delta.compute_BD_Rate(reference_model, bd_model)

        print(dataset)
        print(key)
        print(delta_psnr)
        print(delta_rate)

        # find rate linspace
        ref_min, ref_max = np.min(reference_model.psnr_values), np.max(reference_model.psnr_values)
        model_min, model_max = np.min(bd_model.psnr_values), np.max(bd_model.psnr_values)

        range = np.linspace(max(ref_min, model_min), min(ref_max, model_max), num=20)

        deltas = []
        for val in range:
            delta = 10**(bd_model.evaluate_rate(val) - reference_model.evaluate_rate(val)) - 1
            deltas.append(delta * 100)

        ax_delta.plot(deltas, range, color=colors[key], marker="o", markersize=2)

    ax_delta.grid(which="both")
    ax_delta.set_xlabel("Rate savings [%]")
    ax_delta.set_ylabel("Y PNSR [dB]")
    ax_delta.set_xlim([-80, 80])
    ax_delta.set_ylim([24, 36])

    fig_delta.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig_delta.savefig("plot/figures/{}_delta-y.pdf".format(str(dataset)))

    for key, bd_model in bd_models.items():
        if key == "G-PCC (tmc13)":
            continue # Not suiting points for G-PCC

        Delta = Bjontegaard_Delta()
        delta_psnr = Delta.compute_BD_PSNR(reference_model, bd_model)
        delta_rate = Delta.compute_BD_Rate(reference_model, bd_model)

        # find rate linspace
        ref_min, ref_max = np.min(reference_model.psnr_values), np.max(reference_model.psnr_values)
        model_min, model_max = np.min(bd_model.psnr_values), np.max(bd_model.psnr_values)

        bd_model = bd_models_cumulated[key]
        delta_psnr = Delta.compute_BD_PSNR(reference_model_cumulated, bd_model)
        delta_rate = Delta.compute_BD_Rate(reference_model_cumulated, bd_model)

        print(dataset)
        print(key)
        print(delta_psnr)
        print(delta_rate)

        deltas = []
        for val in range:
            delta = 10**(bd_model.evaluate_rate(val) - reference_model_cumulated.evaluate_rate(val)) - 1
            deltas.append(delta * 100)

        ax_delta_cum.plot(deltas, range, color=colors[key], marker="o", markersize=2)


    ax_delta_cum.grid(which="both")
    ax_delta_cum.set_xlabel("Rate savings [%]")
    ax_delta_cum.set_ylabel("Y PNSR [dB]")
    ax_delta_cum.set_xlim([-75, 5])
    ax_delta_cum.set_ylim([24, 36])
    if dataset == "MVUB":
        ax_delta_cum.set_ylim([24, 34])

    fig_delta_cum.subplots_adjust(left=left, right=right, top=top, bottom=bottom)
    fig_delta_cum.savefig("plot/figures/{}_delta-y_cum.pdf".format(str(dataset)))