import os
import time
import yaml

import torch
import open3d as o3d
import pandas as pd
import numpy as np
import copy
from torch.utils.data import DataLoader

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

experiments = [
    #"MeanScale_5_lambda200-6400_200epochs",
    "MeanScale_1_lambda100_v2",
    #"MeanScale_1_lambda100",
    #"MeanScale_1_lambda200",
    #"MeanScale_1_lambda400",
    #"MeanScale_1_lambda800",
    #"MeanScale_1_lambda1600",
    #"MeanScale_1_lambda3200",
]

def run_testset(experiments):
    # Device
    device = torch.device(0)
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
                # Prepare data
                points = data["src"]["points"].to(device, dtype=torch.float)
                colors = data["src"]["colors"].to(device, dtype=torch.float)
                source = torch.concat([points, colors], dim=2)[0]
                coordinates = source.clone()

                # Side info
                N = source.shape[0]
                sequence = data["cubes"][0]["sequence"][0]
                print(sequence)

                # Compression
                torch.cuda.synchronize()
                t0 = time.time()

                strings, shapes = model.compress(source)
                #strings, shapes = model.compress(source, latent_path="temp")

                torch.cuda.synchronize()
                t_compress = time.time() - t0

                # Decompress all rates
                y_strings = []
                z_strings = []
                for i in range(len(strings[0])):
                    y_strings.append(strings[0][i])
                    z_strings.append(strings[1][i])
                    current_strings = [y_strings, z_strings]

                    # Run decompression
                    torch.cuda.synchronize()
                    t0 = time.time()
                    reconstruction = model.decompress(coordinates=coordinates, 
                                                            strings=current_strings, 
                                                            shape=shapes)
                    torch.cuda.synchronize()
                    t_decompress = time.time() - t0
                    
                    # Rebuild point clouds
                    source_pc = utils.get_o3d_pointcloud(source)
                    rec_pc = utils.get_o3d_pointcloud(reconstruction)

                    # Compute metrics
                    metric = PointCloudMetric(source_pc, rec_pc)
                    results, error_vectors = metric.compute_pointcloud_metrics(drop_duplicates=True)

                    # Save results
                    results["bpp"] = utils.count_bits(y_strings) / N
                    results["layer"] = i
                    results["sequence"] = data["cubes"][0]["sequence"][0]
                    results["frameIdx"] = data["cubes"][0]["frameIdx"][0].item()
                    results["t_compress"] = t_compress
                    results["t_decompress"] = t_decompress
                    experiment_results.append(results)

                    # Renders of the pointcloud
                    point_size = 1.0 if data["cubes"][0]["sequence"][0] in ["longdress", "soldier", "loot", "longdress"] else 2.0
                    path = os.path.join(base_path,
                                        experiment, 
                                        "renders_test", 
                                        "{}_{}_{}.png".format(sequence, str(i), "{}"))
                    utils.render_pointcloud(rec_pc, path, point_size=point_size)

                    # Renders of the color-errors
                    path = os.path.join(base_path,
                                        experiment, 
                                        "renders_test", 
                                        "error_y_{}_{}_{}.png".format(sequence, str(i), "{}"))
                    # Y error
                    error_point_cloud = copy.deepcopy(rec_pc)
                    error_vectors = error_vectors["colorAB"]
                    colors_error = ScalarMappable(norm=norm, cmap="magma").to_rgba(error_vectors)
                    error_point_cloud.colors = o3d.utility.Vector3dVector(colors_error[:, 0, :3])
                    utils.render_pointcloud(error_point_cloud, path, point_size=point_size)


                    fig, ax = plt.subplots(figsize=(2, 4), layout='constrained')
                    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap="magma"),
                            cax=ax, orientation='vertical', label='MSE')
                    for t in cbar.ax.get_yticklabels():
                        t.set_fontsize(18)
                    cbar.ax.set_ylabel("MSE", fontsize=18)

                    fig.savefig(os.path.join(base_path, experiment, "renders_test", "error_bar.png"), bbox_inches="tight")
                    exit(0)
                    """
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
