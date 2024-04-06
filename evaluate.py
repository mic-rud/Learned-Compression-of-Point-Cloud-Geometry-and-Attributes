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


device_id = 0
experiments = [
    #"03_19_Debug_ColorsSSIM_jitter",
    #"03_18_Debug_ColorsL2_25600"
    #"03_20_Debug_ColorsL2_2models",
    #"03_20_Debug_ColorsL2_2models_scale_noact"
    #"03_21_Debug_ColorsL2_2models_res_proj"
    #"03_25_Debug_ColorsL2_2models_res_proj_256"
    #"03_26_Debug_ColorsL2_2models_res_proj_128"
    #"03_28_Debug_ColorsL2_2models_q_infer_Dense_quadratic_100",
    #"03_28_Debug_ColorsL2_2models_q_infer_Dense_100"
    #"Final_L2_200epochs_SC_2",
    #"Final_SSIM_200_quadratic",
    "G-PCC"
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
            q_as = np.arange(21, 52, 2)
            q_gs = [0.125, 0.25, 0.5, 0.875, 0.9375]
        elif experiment == "V-PCC":
            q_as = []
            q_gs = []
        

        with torch.no_grad():
            for s, data in enumerate(test_loader):
                for i, q_a in enumerate(q_as):
                    for j, q_g in enumerate(q_gs):
                        # Get info
                        t0 = time.time()
                        sequence = data["cubes"][0]["sequence"][0]

                        # Run model
                        if experiment not in related_work:
                            source_pc, rec_pc, bpp, t_compress, t_decompress = compress_ours(experiment,
                                                                                             model,
                                                                                             data,
                                                                                             q_a, 
                                                                                             q_g, 
                                                                                             device)
                        else:
                            source_pc, rec_pc, bpp, t_compress, t_decompress = compress_related(experiment,
                                                                                                data,
                                                                                                q_a,
                                                                                                q_g)

                        # Compute metrics
                        metric = PointCloudMetric(source_pc, rec_pc, resolution=resolutions[sequence])
                        results, error_vectors = metric.compute_pointcloud_metrics(drop_duplicates=False)
                        results["pcqm"] = utils.pcqm(ref_paths[data["cubes"][0]["sequence"][0]], 
                                                     rec_pc, 
                                                     "dependencies/PCQM/build")

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

def compress_ours(experiment, model, data, q_a, q_g, device):
    """
    Compress a point cloud using our model
    """
    points = data["src"]["points"].to(device, dtype=torch.float)
    colors = data["src"]["colors"].to(device, dtype=torch.float)
    source = torch.concat([points, colors], dim=2)[0]
    N = source.shape[0]

    # Bin path
    bin_path = os.path.join(base_path,
                            experiment,
                            "tmp")
    if not os.path.exists(bin_path): 
        os.mkdir(bin_path)
    bin_path = os.path.join(bin_path, "bitstream.bin")

    # Q Map
    Q_map = ME.SparseTensor(coordinates=torch.cat([torch.zeros((N, 1), device=device), points[0]], dim=1), 
                            features=torch.cat([torch.ones((N,1), device=device) * q_g, torch.ones((N,1), device=device) * q_a], dim=1),
                            device=source.device)

    # Compression
    torch.cuda.synchronize()
    t0 = time.time()

    #strings, shapes, k, coordinates = model.compress(source, Q_map)
    model.compress(source, Q_map, path=bin_path)

    torch.cuda.synchronize()
    t_compress = time.time() - t0

    # Decompress all rates
    # Run decompression
    torch.cuda.synchronize()
    t0 = time.time()
    reconstruction = model.decompress(path=bin_path)
    #reconstruction = model.decompress(coordinates=coordinates, strings=strings, shape=shapes, k=k)
    torch.cuda.synchronize()
    t_decompress = time.time() - t0
                    
    # Rebuild point clouds
    source_pc = utils.get_o3d_pointcloud(source)
    rec_pc = utils.get_o3d_pointcloud(reconstruction)

    bpp = os.path.getsize(bin_path) * 8 / N

    return source_pc, rec_pc, bpp, t_compress, t_decompress

def compress_related(experiment, data, q_a, q_g):
    """
    Compress a point cloud using V-PCC/G-PCC
    """
    path = os.path.join(base_path,
                        experiment,
                        "tmp")
    if not os.path.exists(path):
        os.mkdir(path)
    # Directories
    src_dir = os.path.join(path, "points_enc.ply")
    rec_dir = os.path.join(path, "points_dec.ply")
    bin_dir = os.path.join(path, "points_enc.bin")

    N = data["src"]["points"].shape[1]

    # Data processing
    dtype = o3d.core.float32
    c_dtype = o3d.core.uint8
    points = data["src"]["points"].to(dtype=torch.float)
    colors = torch.clamp(data["src"]["colors"].to(dtype=torch.float) * 255, 0, 255)
    p_tensor = o3d.core.Tensor(points.detach().cpu().numpy()[0, :, :], dtype=dtype)
    p_colors = o3d.core.Tensor(colors.detach().cpu().numpy()[0, :, :], dtype=c_dtype)
    source = o3d.t.geometry.PointCloud(p_tensor)
    source.point.colors = p_colors
    o3d.t.io.write_point_cloud(src_dir, source, write_ascii=True)

    if experiment == "G-PCC":
        # Compress the point cloud using G-PCC

        command = ['./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3',
                '--mode=0',
                '--trisoupNodeSizeLog2=0',
                '--mergeDuplicatedPoints=1',
                '--neighbourAvailBoundaryLog2=8',
                '--intra_pred_max_node_size_log2=6',
                '--positionQuantizationScale={}'.format(q_g),
                '--maxNumQtBtBeforeOt=4',
                '--minQtbtSizeLog2=0',
                '--planarEnabled=1',
                '--planarModeIdcmUse=0',
                '--convertPlyColourspace=1',

                '--transformType=0',
                '--qp={}'.format(q_a),
                '--qpChromaOffset=-2',
                '--bitdepth=8',
                '--attrScale=1',
                '--attrOffset=0',
                '--attribute=color',

                '--uncompressedDataPath={}'.format(src_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user)" in line:
                processing_time_line = line
        t_compress = float(processing_time_line.split()[-2])

        bpp = os.path.getsize(bin_dir) * 8 / N
        # Decode
        command = ['./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3',
                '--mode=1',
                '--convertPlyColourspace=1',
                '--outputBinaryPly=0',
                '--reconstructedDataPath={}'.format(rec_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user)" in line:
                processing_time_line = line
        t_decompress = float(processing_time_line.split()[-2])

        print(t_compress, t_decompress)

        # Read ply (o3d struggles with GBR order)
        utils.remove_gpcc_header(rec_dir)
        rec_pc = o3d.io.read_point_cloud(rec_dir)
        colors = np.asarray(rec_pc.colors)
        colors = colors[:, [2,0,1]]
        rec_pc.colors=o3d.utility.Vector3dVector(colors)

        # Clean up
        os.remove(rec_dir)
        os.remove(src_dir)
        os.remove(bin_dir)
    else: 
        # Compress the point cloud using V-PCC
        pass

    # Reconstruct source
    points = data["src"]["points"]
    colors = data["src"]["colors"]
    source = torch.concat([points, colors], dim=2)[0]
    source_pc = utils.get_o3d_pointcloud(source)
    return source_pc, rec_pc, bpp, t_compress, t_decompress

if __name__ == "__main__":
    run_testset(experiments)
