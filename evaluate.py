import os
import time
import yaml

import torch
import open3d as o3d
import pandas as pd
import numpy as np
import copy
from torch.utils.data import DataLoader
import MinkowskiEngine as ME
import subprocess

import utils
from model.model import ColorModel
from data.dataloader import StaticDataset
from metrics.metric import PointCloudMetric

import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

norm = Normalize(vmin=0.0, vmax=10**(-2))
# Paths
base_path = "./results"
data_path = "./data/datasets/full_128" 
ref_paths = {
    "longdress" : "data/datasets/raw/longdress/longdress/Ply/longdress_vox10_1300.ply",
     "soldier" : "data/datasets/raw/soldier/soldier/Ply/soldier_vox10_0690.ply",
     "loot" : "data/datasets/raw/loot/loot/Ply/loot_vox10_1200.ply",
     "redandblack" : "data/datasets/raw/redandblack/redandblack/Ply/redandblack_vox10_1550.ply",
     "phil9" : "data/datasets/raw/phil9/phil9/ply/frame0000.ply",
     "sarah9" : "data/datasets/raw/sarah9/sarah9/ply/frame0000.ply",
     "andrew9" : "data/datasets/raw/andrew9/andrew9/ply/frame0000.ply",
     "david9" : "data/datasets/raw/david9/david9/ply/frame0000.ply",
     }
resolutions ={
    "longdress" : 1023, "soldier" : 1023, "loot" : 1023, "redandblack" : 1023,
     "phil9" : 511, "sarah9" : 511, "andrew9" : 511, "david9" : 511,
}


device_id = 1
experiments = [
    "Ours",
    "V-PCC",
    "G-PCC",
    ]

related_work = [
    "G-PCC",
    "V-PCC"
]

def run_testset(experiments):
    # Device
    device = torch.device(device_id)
    torch.cuda.set_device(device)

    torch.no_grad()
        
    # Dataloader
    test_set = StaticDataset(data_path, split="test", transform=None, partition=False)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    for experiment in experiments:
        experiment_results = []

        # Set model and QPs
        if experiment not in related_work:
            q_as = np.arange(21) * 0.05
            q_gs = np.arange(21) * 0.05

            weight_path = os.path.join(base_path, experiment, "weights.pt")
            config_path = os.path.join(base_path, experiment, "config.yaml")

            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)

            model = ColorModel(config["model"])
            model.load_state_dict(torch.load(weight_path))
            model.to(device)
            model.eval()
            model.update()
        elif experiment == "G-PCC":
            q_as = np.arange(21, 52)
            q_gs = [0.125, 0.1875, 0.25, 0.375, 0.5, 0.75, 0.875, 0.9375]
        elif experiment == "V-PCC":
            #q_as = [42, 37, 32, 27, 22]
            q_as = np.arange(22, 43)
            q_gs = [32, 28, 24, 20, 16]
        

        with torch.no_grad():
            for s, data in enumerate(test_loader):
                for i, q_a in enumerate(q_as):
                    for j, q_g in enumerate(q_gs):
                        # Get info
                        t0 = time.time()
                        sequence = data["cubes"][0]["sequence"][0]

                        # Run model
                        if experiment not in related_work:
                            source_pc, rec_pc, bpp, t_compress, t_decompress = utils.compress_model_ours(experiment,
                                                                                             model,
                                                                                             data,
                                                                                             q_a, 
                                                                                             q_g, 
                                                                                             device,
                                                                                             base_path)
                        else:
                            source_pc, rec_pc, bpp, t_compress, t_decompress = utils.compress_related(experiment,
                                                                                                data,
                                                                                                q_a,
                                                                                                q_g,
                                                                                                base_path)

                        tmp_path = os.path.join(base_path,
                                                experiment)
                        results = utils.pc_metrics(ref_paths[data["cubes"][0]["sequence"][0]], 
                                                     rec_pc, 
                                                     "dependencies/mpeg-pcc-tmc2/bin/PccAppMetrics",
                                                     tmp_path,
                                                     resolution=resolutions[sequence])

                        results["pcqm"] = utils.pcqm(ref_paths[data["cubes"][0]["sequence"][0]], 
                                                     rec_pc, 
                                                     "dependencies/PCQM/build",
                                                     tmp_path)

                        # Save results
                        results["bpp"] = bpp
                        results["sequence"] = data["cubes"][0]["sequence"][0]
                        results["frameIdx"] = data["cubes"][0]["frameIdx"][0].item()
                        results["t_compress"] = t_compress
                        results["t_decompress"] = t_decompress
                        results["q_a"] = q_a
                        results["q_g"] = q_g
                        experiment_results.append(results)

                        torch.cuda.empty_cache()
                        t1 = time.time() - t0
                        total = len(test_loader) * len(q_as) * len(q_gs)
                        done = (s * len(q_as) * len(q_gs)) + (i * len(q_gs)) + j + 1
                        print("[{}/{}] Experiment: {} | Sequence: {} @ q_a:{:.2f} q_g:{:.2f} | {:2f}s | PCQM:{:4f} bpp:{:2f}".format(done,
                                                                                                                                     total,
                                                                                                                                     experiment,
                                                                                                                               sequence, 
                                                                                                                               q_a, 
                                                                                                                               q_g,  
                                                                                                                               t1,
                                                                                                                               results["pcqm"],
                                                                                                                               results["bpp"]))
                        # Renders of the pointcloud
                        point_size = 1.0 if data["cubes"][0]["sequence"][0] in ["longdress", "soldier", "loot", "longdress"] else 2.0
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test",
                                            "{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                        utils.render_pointcloud(rec_pc, path, point_size=point_size)

                        # Renders of the original
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test",
                                            "{}_original_{}.png".format(sequence, "{}"))
                        utils.render_pointcloud(source_pc, path, point_size=point_size)
                        """
                        fig, ax = plt.subplots(figsize=(2, 4), layout='constrained')
                        cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"),
                                cax=ax, orientation='vertical', label='MSE')
                        for t in cbar.ax.get_yticklabels():
                            t.set_fontsize(18)
                        cbar.ax.set_ylabel("MSE", fontsize=18)

                        fig.savefig(os.path.join(base_path, experiment, "renders_test", "error_bar.png"), bbox_inches="tight")
                        exit(0)
                        # Calculate the median x-coordinate
                        median_x = np.median(np.asarray(error_point_cloud.points)[:, 0])

                        # Split error_point_cloud
                        error_points = np.asarray(error_point_cloud.points)
                        left_error_indices = np.where(error_points[:, 0] <= median_x)[1]
                        right_error_indices = np.where(error_points[:, 0] > median_x)[1]

                        left_error_cloud = error_point_cloud.select_by_index(left_error_indices)
                        right_error_cloud = error_point_cloud.select_by_index(right_error_indices)

                        # Split rec_pc
                        rec_pc_points = np.asarray(rec_pc.points)
                        left_rec_pc_indices = np.where(rec_pc_points[:, 0] <= median_x)[0]
                        right_rec_pc_indices = np.where(rec_pc_points[:, 0] > median_x)[0]

                        left_rec_pc = rec_pc.select_by_index(left_rec_pc_indices)
                        right_rec_pc = rec_pc.select_by_index(right_rec_pc_indices)

                        split_cloud = o3d.geometry.PointCloud()
                        split_cloud.points = o3d.utility.Vector3dVector(np.concatenate([left_error_cloud.points, right_rec_pc.points], axis=0))
                        split_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([left_error_cloud.colors, right_rec_pc.colors], axis=0))
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test", 
                                            "split_y_{}_{}_{}.png".format(sequence, str(i), "{}"))
                        utils.render_pointcloud(split_cloud, path)

                        # Ply
                        path = os.path.join(base_path,
                                            experiment, 
                                            "plys", 
                                            "{}_{:04d}_rec{}.ply".format(sequence, results["frameIdx"], str(i)))
                        """

                    # Save the results as .csv
                    df = pd.DataFrame(experiment_results)
                    results_path = os.path.join(base_path, experiment, "test.csv")
                    df.to_csv(results_path)





if __name__ == "__main__":
    run_testset(experiments)
