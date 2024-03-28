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

q_as = np.arange(21) * 0.05
q_gs = np.arange(21) * 0.05
q_as = np.arange(6) * 0.2
q_gs = np.arange(6) * 0.2

device_id = 3
experiments = [
    #"03_19_Debug_ColorsSSIM_jitter",
    #"03_18_Debug_ColorsL2_25600"
    #"03_20_Debug_ColorsL2_2models",
    #"03_20_Debug_ColorsL2_2models_scale_noact"
    #"03_21_Debug_ColorsL2_2models_res_proj"
    #"03_25_Debug_ColorsL2_2models_res_proj_256"
    #"03_26_Debug_ColorsL2_2models_res_proj_128"
    "03_25_Debug_ColorsSSIM_2models_res_proj"
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
        weight_path = os.path.join(base_path, experiment, "weights.pt")
        config_path = os.path.join(base_path, experiment, "config.yaml")

        with open(config_path, "r") as config_file:
            config = yaml.safe_load(config_file)

        # Model
        model = ColorModel(config["model"])
        model.load_state_dict(torch.load(weight_path))
        model.to(device)
        model.eval()
        model.update()
        
        experiment_results = []

        with torch.no_grad():
            for _, data in enumerate(test_loader):
                for q_a in q_as[::-1]:
                    for q_g in q_gs[::-1]:
                        # Prepare data
                        points = data["src"]["points"].to(device, dtype=torch.float)
                        colors = data["src"]["colors"].to(device, dtype=torch.float)
                        source = torch.concat([points, colors], dim=2)[0]
                        coordinates = source.clone()

                        # Side info
                        N = source.shape[0]
                        sequence = data["cubes"][0]["sequence"][0]
                        print(sequence)

                        # Q Map
                        Q_map = ME.SparseTensor(coordinates=torch.cat([torch.zeros((N, 1), device=device), points[0]], dim=1), 
                                                features=torch.cat([torch.ones((N,1), device=device) * q_g, torch.ones((N,1), device=device) * q_a], dim=1),
                                                device=source.device)

                        # Compression
                        torch.cuda.synchronize()
                        t0 = time.time()

                        strings, shapes, k, coordinates = model.compress(source, Q_map)

                        torch.cuda.synchronize()
                        t_compress = time.time() - t0

                        # Decompress all rates
                        # Run decompression
                        torch.cuda.synchronize()
                        t0 = time.time()
                        reconstruction = model.decompress(coordinates=coordinates, 
                                                                strings=strings, 
                                                                shape=shapes,
                                                                k=k)
                        torch.cuda.synchronize()
                        t_decompress = time.time() - t0
                    
                        # Rebuild point clouds
                        source_pc = utils.get_o3d_pointcloud(source)
                        rec_pc = utils.get_o3d_pointcloud(reconstruction)

                        # Compute metrics
                        metric = PointCloudMetric(source_pc, rec_pc)
                        results, error_vectors = metric.compute_pointcloud_metrics(drop_duplicates=True)
                        results["pcqm"] = utils.pcqm(ref_paths[data["cubes"][0]["sequence"][0]], 
                                                     rec_pc, 
                                                     "dependencies/PCQM/build")

                        # Save results
                        results["bpp"] = utils.count_bits(strings) / N
                        results["sequence"] = data["cubes"][0]["sequence"][0]
                        results["frameIdx"] = data["cubes"][0]["frameIdx"][0].item()
                        results["t_compress"] = t_compress
                        results["t_decompress"] = t_decompress
                        results["q_a"] = q_a
                        results["q_g"] = q_g
                        experiment_results.append(results)

                        # Renders of the pointcloud
                        point_size = 1.0 if data["cubes"][0]["sequence"][0] in ["longdress", "soldier", "loot", "longdress"] else 2.0
                        path = os.path.join(base_path,
                                            experiment, 
                                            "renders_test", 
                                            "{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                        #utils.render_pointcloud(rec_pc, path, point_size=point_size)

                        torch.cuda.empty_cache()
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
