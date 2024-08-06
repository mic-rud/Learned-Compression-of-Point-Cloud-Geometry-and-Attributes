import os
import yaml
import shutil
import argparse
import numpy as np
from utils.RawLoader import RawLoader
from utils.Cubes import CubeHandler

def prepare_folders(path):
    # Check and prepare folder
    head, _ = os.path.split(path)
    if not os.path.exists(head):
        raise ValueError("dataset root {} not existant".format(head))
    
    if os.path.exists(path):
        if not len(os.listdir(path)) == 0:
            # Fail save to not overwrite without user confirmation
            response = input("Overwrite {} to continue? (y/n): ".format(path))
            if response == "yes" or response== "y":
                shutil.rmtree(path)
            else:
                raise ValueError("Aborting, invalid prompt")
        else:
            # Delete empyt dir without prompting
            shutil.rmtree(path)

    os.mkdir(path)

def prepare_split(config, split, args):
    split_config = config[split]
    block_size = tuple(config["info"]["block_size"])

    raw_loader = RawLoader(args.raw_data_path, args.raw_config)
    all_paths = []
    num_points = []
    for sequence, frames in split_config.items():
        for frame in frames:
            print("Slicing {}_{}".format(sequence, str(frame)))
            orig_pointcloud =raw_loader.get_pointcloud(sequence, frame)

            # Slice pc
            handler = CubeHandler(sequence, frame)
            handler.slice(orig_pointcloud, block_size)

            # Save files
            num_cubes = len(handler.cubes)
            handler.write(args.dataset_path)
            all_paths.append(handler.get_cube_paths())
            num_points.append(handler.get_num_points())
            print("Saved sliced {}_{}. {} blocks written.".format(sequence, str(frame), num_cubes))

            # Check against original pc
            sliced_pointcloud = handler.get_pointcloud()
            pc_distance = sliced_pointcloud.compute_point_cloud_distance(orig_pointcloud)
            pc_distance = np.mean(np.asarray(pc_distance))
            print("Verifying sliced {}_{}, Pointcloud distance = {:.06f}".format(sequence, str(frame), pc_distance))
            if pc_distance > 1e-10:
                raise ValueError("Something went wrong, Point Cloud Distance to large ({})".format(pc_distance))

    # write the .yaml file of all paths
    split_dict = {}
    all_paths = [item for sublist in all_paths for item in sublist]
    num_points = [item for sublist in num_points for item in sublist]
    for path, points in zip(all_paths, num_points):
        split_dict[path] = points

    split_doc_path = os.path.join(args.dataset_path, "{}.yaml".format(split))
    with open(split_doc_path, "w+") as file:
        yaml.safe_dump(split_dict, file)


def prepare_data(config, args):
    prepare_folders(args.dataset_path)
    for split in ["train", "test", "val"]:
        prepare_split(config, split, args)



def parse_args():
    """
    Parse arguments from command line
    """
    parser = argparse.ArgumentParser(
        prog="Prepare dataset based on config.yaml",
        description="Script for bulk download of all point cloud data"
    )

    # Paths
    parser.add_argument("--config_path", type=str, default="./data/configs/dataset_dev.yaml")
    parser.add_argument("--dataset_path", type=str, default="./data/dataset_dev")
    parser.add_argument("--raw_config", type=str, default="./data/configs/raw_loading.yaml")
    parser.add_argument("--raw_data_path", type=str, default="./data/raw")

    args = parser.parse_args()
    return args

def parse_config(args):
    """
    Parse the config to a dict with datasets, sequences and list of frames
    """
    config_path = args.config_path
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)

    for split , sub_dicts in config.items():
        if split == "info":
            continue # Skip info

        for key, item in sub_dicts.items():
            new_item = []
            if not isinstance(item, str):
                raise ValueError("Cannot parse config, all keys should be str.")

            # Split by commas
            sub_items = item.split(",")

            for sub_item in sub_items:
                # Hanlde start, end, (stride=1) notation
                if ":" in sub_item:
                    elements = sub_item.split(":")
                    if len(elements) == 2:
                        stride = 1
                    elif len(elements) == 3:
                        stride = int(elements[2])

                    sub_range = list(range(int(elements[0]), int(elements[1])+1, stride))
                    new_item += sub_range
                # Handle single indexing
                elif isinstance(sub_item, str):
                    new_item.append(int(sub_item))

            new_item.sort()
            new_item = list(set(new_item)) # Removes redundancies
            config[split][key] = new_item

    return config


if __name__ == "__main__":
    # Parse config.yaml
    args = parse_args()
    dataset_config = parse_config(args)
    prepare_data(dataset_config, args)
