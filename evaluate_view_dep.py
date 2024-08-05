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
import skimage as ski

from plot import style
from metrics.bjontegaard import Bjontegaard_Delta, Bjontegaard_Model

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm

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
full_bodies = ["longdress", "loot", "redandblack", "soldier"]
    
views = {
    "full_bodies" :
    {
        "front":    [[0, 0, 1], [0, 1, 0]],
        "side":   [[-1, 0, 0], [0, 1, 0]],
    },
    "mvub" :
    {
        "front":  [[0, -1, 0], [0, 0, 1]],
        "side":   [[-1, 0, 0], [0, 0, 1]],
    },
}
view_grads = {
    "loot": [300, 90, 2], #min, max, direction
    "soldier": [200, 50, 2],
    "longdress": [220, 180, 2],
    "redandblack": [250, 50, 2],
    "phil9": [70, 200, 1],
    "david9": [50, 200, 1],
    "sarah9": [100, 200 , 1],
    "andrew9": [170, 220, 1],
}
cut_offs = {
    "loot": [260, 0], #min, max, direction
    "soldier": [235, 0],
    "longdress": [210, 0],
    "redandblack": [310, 0],
    "phil9": [200, 0],
    "david9": [200, 0],
    "sarah9": [200 , 0],
    "andrew9": [200, 0],
}

device_id = 1
experiments = [
    "Final_L2_200epochs_SC_2",
    #"Final_L2_200epochs_SC_2_project",
    #"Final_SSIM_200_quadratic",
    #"V-PCC",
    #"G-PCC",
    ]

related_work = [
    "G-PCC",
    "V-PCC"
]

configs = {
    "Final_L2_200epochs_SC_2" : [(0.05, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8)],
    "G-PCC" : [(0.125, 51), (0.25, 46), (0.5, 40), (0.75, 34)],
    "V-PCC" : [(32,42), (28, 37), (24, 32), (22, 27)],
}


#### PLOT STUFF

def run_view_dep(experiments):
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
            weight_path = os.path.join(base_path, experiment, "weights.pt")
            config_path = os.path.join(base_path, experiment, "config.yaml")

            with open(config_path, "r") as config_file:
                config = yaml.safe_load(config_file)

            model = ColorModel(config["model"])
            model.load_state_dict(torch.load(weight_path))
            model.to(device)
            model.eval()
            model.update()

        with torch.no_grad():
            for s, data in enumerate(test_loader):
                sequence = data["cubes"][0]["sequence"][0]

                if not sequence == "redandblack":
                    continue
                print(sequence)

                
                for r, config in enumerate(configs[experiment]):
                    results_view = {}
                    results_uniform = {}
                    results_roi = {}

                    
                    # Rendering
                    view_conf = views["full_bodies" if sequence in full_bodies else "mvub"]
                    w = 512 if sequence in full_bodies else 1024
                    zoom = 0.6 if sequence in full_bodies else 0.65
                    point_size = 1.0 if data["cubes"][0]["sequence"][0] in ["longdress", "soldier", "loot", "longdress"] else 2.0

                    img_path = os.path.join(base_path, experiment, "renders_view")
                    q_g, q_a = config


                    # code uniform
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
                    results_uniform["bpp"] = bpp
                    results_uniform["q_a"] = q_a
                    results_uniform["q_g"] = q_g
                    results_uniform["key"] = "uniform"
                    results_uniform["sequence"] = sequence

                    q_pc_uni = o3d.geometry.PointCloud()
                    q_pc_uni.points = source_pc.points
                    q_map_a = np.ones((len(source_pc.points),1)) * q_a
                    fill = np.zeros_like(q_map_a)
                    q_colors = np.concatenate([q_map_a, fill, fill], axis=1)
                    q_pc_uni.colors = o3d.utility.Vector3dVector(q_colors)
                    q_pc_uni_path = os.path.join(img_path, "q_pc_uni_{}_{:2}_{}.png".format(sequence,q_a, "{}"))
                    render_pointviews(q_pc_uni, q_pc_uni_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)
                    torch.cuda.empty_cache()

                    ######### Save images (ref, uniform, view ,roi, q_maps)
                    # Reference
                    ref_path = os.path.join(img_path, "ref_{}_{}.png".format(sequence, "{}"))
                    render_pointviews(source_pc, ref_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)

                    # Uniform
                    uniform_path = os.path.join(img_path, "uniform_{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                    render_pointviews(rec_pc, uniform_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)

                    # Compute PSNRs and SSIMs
                    ref_image = ski.io.imread(ref_path.format("front"))
                    uniform_image = ski.io.imread(uniform_path.format("front"))

                    # Color spaces
                    ref_image = ski.color.rgb2yuv(ref_image)
                    uniform_image = ski.color.rgb2yuv(uniform_image)

                    results_uniform["psnr"] = ski.metrics.peak_signal_noise_ratio(ref_image, uniform_image)
                    results_uniform["ssim"] = ski.metrics.structural_similarity(ref_image, uniform_image, channel_axis=2, data_range=1.0)
                
                    # View-dependent
                    if experiment not in related_work:
                        # code view
                        max, min, dir = view_grads[sequence]

                        scores = np.clip((np.asarray(source_pc.points)[:, dir] - min) / (max - min), 0, 1)

                        # Create quality map
                        q_map_a = (q_a * scores).reshape(-1, 1)
                        q_map_g = (q_g * scores).reshape(-1, 1)

                        # Create a render
                        q_pc_view = o3d.geometry.PointCloud()
                        q_pc_view.points = source_pc.points
                        fill = np.zeros_like(q_map_a)
                        q_colors = np.concatenate([q_map_a, fill, fill], axis=1)
                        q_pc_view.colors = o3d.utility.Vector3dVector(q_colors)
                        q_pc_view_path = os.path.join(img_path, "q_pc_view_{}_{:.2}_{}.png".format(sequence, q_a, "{}"))
                        render_pointviews(q_pc_view, q_pc_view_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)
                        
                        _, view_pc, bpp, t_compress, t_decompress = utils.compress_model_ours(experiment,
                                                                                            model,
                                                                                            data,
                                                                                            q_map_a, 
                                                                                            q_map_g, 
                                                                                            device,
                                                                                            base_path)
                        view_path = os.path.join(img_path, "view_{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                        render_pointviews(view_pc, view_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)

                        results_view["bpp"] = bpp
                        results_view["q_a"] = q_a
                        results_view["q_g"] = q_g
                        results_view["key"] = "view"
                        results_view["sequence"] = sequence

                        # Compute PSNRs and SSIMs
                        view_image = ski.io.imread(view_path.format("front"))

                        # Color spaces
                        view_image = ski.color.rgb2yuv(view_image)

                        results_view["psnr"] = ski.metrics.peak_signal_noise_ratio(ref_image, view_image)
                        results_view["ssim"] = ski.metrics.structural_similarity(ref_image, view_image, channel_axis=2, data_range=1.0)

                        torch.cuda.empty_cache()

                        # ### code region of interest
                        plane, dir = cut_offs[sequence]

                        scores = np.where(np.asarray(source_pc.points)[:, dir] < plane, 0, 1)

                        # Create quality map
                        q_map_a = (q_a * scores).reshape(-1, 1)
                        q_map_g = (q_g * scores).reshape(-1, 1)

                        # Create a render
                        q_pc_view = o3d.geometry.PointCloud()
                        q_pc_view.points = source_pc.points
                        fill = np.zeros_like(q_map_a)
                        q_colors = np.concatenate([q_map_a, fill, fill], axis=1)
                        q_pc_view.colors = o3d.utility.Vector3dVector(q_colors)
                        q_pc_view_path = os.path.join(img_path, "q_pc_roi_{}_{:.2}_{}.png".format(sequence, q_a, "{}"))
                        render_pointviews(q_pc_view, q_pc_view_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)
                        
                        _, roi_pc, bpp, t_compress, t_decompress = utils.compress_model_ours(experiment,
                                                                                            model,
                                                                                            data,
                                                                                            q_map_a, 
                                                                                            q_map_g, 
                                                                                            device,
                                                                                            base_path)
                        roi_path = os.path.join(img_path, "roi_{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                        render_pointviews(roi_pc, roi_path, view_conf, h=1024, w=w, zoom=zoom, point_size=point_size)

                        results_roi["bpp"] = bpp
                        results_roi["q_a"] = q_a
                        results_roi["q_g"] = q_g
                        results_roi["key"] = "roi"
                        results_roi["sequence"] = sequence

                        # Compute PSNRs and SSIMs
                        roi_image = ski.io.imread(roi_path.format("front"))

                        # Color spaces
                        roi_image = ski.color.rgb2yuv(roi_image)

                        results_roi["psnr"] = ski.metrics.peak_signal_noise_ratio(ref_image, roi_image)
                        results_roi["ssim"] = ski.metrics.structural_similarity(ref_image, roi_image, channel_axis=2, data_range=1.0)

                        torch.cuda.empty_cache()

                    experiment_results.append(results_uniform)
                    if experiment not in related_work:
                        experiment_results.append(results_view)
                        experiment_results.append(results_roi)

                df = pd.DataFrame(experiment_results)
                results_path = os.path.join(base_path, experiment, "view_dep.csv")
                df.to_csv(results_path)


def render_pointviews(pc, path, settings, h, w, zoom, point_size=1.0):
    """
    Render the point cloud from 6 views along x,y,z axis

    Parameters
    ----------
    pc: o3d.geometry.PointCloud
        Point Cloud to be rendered
    path: str
        Format String with a open key field for formatting
    """
    # Path
    dir, file = os.path.split(path)
    if not os.path.exists(dir):
        os.mkdir(dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True, width=w, height=h)
    vis.add_geometry(pc)
    # Adjust the point size
    render_options = vis.get_render_option()
    render_options.point_size = 2  * point_size # adjust the size as required
    render_options.background_color = [1, 1, 1]

    for key, view in settings.items():
        # Get view control
        view_control = vis.get_view_control()

        # Fit the object into the scene
        view_control.set_front(view[0])
        view_control.set_up(view[1])
        view_control.set_zoom(zoom)
        view_control.change_field_of_view(-30)


        vis.update_renderer()
        
        image_path = path.format(key)
        vis.capture_screen_image(image_path, do_render=True)

    vis.destroy_window()

if __name__ == "__main__":
    run_view_dep(experiments)

"""
source_pc.estimate_normals(
    fast_normal_computation=True)
source_pc.normalize_normals()
source_pc.orient_normals_to_align_with_direction()
normals = source_pc.normals

# Emulate positioning
vis = o3d.visualization.Visualizer()
vis.create_window(visible=True, width=512, height=1024)
vis.add_geometry(source_pc)
view_control = vis.get_view_control()
view_control.set_front(view_conf["side"][1])
view_control.set_up(view_conf["side"][1])
vis.update_renderer()
cam = view_control.convert_to_pinhole_camera_parameters()
position = [200, -1000, 600]

view_vector = position - np.asarray(source_pc.points) 
view_vector_norm = np.linalg.norm(view_vector, axis=1)
view_vector = view_vector / view_vector_norm.reshape(-1,1)

norm = np.linalg.norm(normals, axis=1)
normals = normals / norm.reshape(-1, 1)
"""