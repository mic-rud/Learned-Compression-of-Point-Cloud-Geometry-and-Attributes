import argparse
import requests
import os
import yaml
from zipfile import ZipFile
import tarfile


def logging(level, process, keyword, text):
    if level == 1:
        print("[{}] [{}] {}".format(process, keyword, text))
    elif level == 2:
        print("\t[{}] [{}] {}".format(process, keyword, text))
    else:
        pass


def download_jpeg_datasets(args, config, dataset):
    config = config[dataset]
    for key, link in config.items():
        logging(1, "Downloading", key, "Starting to download ... ")

        raw_dir = os.path.join(args.data_path, key)
        download_path = os.path.join(args.temp_path, key)

        if os.path.exists(raw_dir):
            logging(2, "Downloading", key, "Already downloaded? Skipping for now - Check for correctness!")
            continue

        if dataset == "uvg-vpc":
            response = download_and_unpack_tar(link, download_path, raw_dir)
        else:
            response = download_and_unpack_zip(link, download_path, raw_dir)

        if response == False:
            logging(2, "Downloading", key, "Download failed, skipping for now")
            continue
        logging(1, "Downloading", key, "Done!")

def download_and_unpack_tar(url, 
                        download_path, 
                        raw_path,
                        chunk_size=1024*1024):
    """
    Downloads and unpacks a given file
    """
    tar_file = download_path + ".tar.gz"

    if not os.path.exists(tar_file):
        response = requests.get(url, stream=True)
        if not response.ok:
            return False
        with open(tar_file, "wb+") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    if not os.path.exists(raw_path):
        with tarfile.open(tar_file, "r") as zippped_content:
            zippped_content.extractall(raw_path)

    return True

def download_and_unpack_zip(url, 
                        download_path, 
                        raw_path, 
                        chunk_size=1024*1024):
    """
    Downloads and unpacks a given file
    """
    zip_file = download_path + ".zip"

    if not os.path.exists(zip_file):
        response = requests.get(url, stream=True)
        if not response.ok:
            return False
        with open(zip_file, "wb+") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    f.write(chunk)

    if not os.path.exists(raw_path):
        with ZipFile(zip_file, "r") as zippped_content:
            zippped_content.extractall(raw_path)

    return True



def parse_args():
    """
    Parse arguments from command line
    """
    parser = argparse.ArgumentParser(
        prog="Bulk download all raw point clouds",
        description="Script for bulk download of all point cloud data"
    )

    # Paths
    parser.add_argument("--data_path", type=str, default="./datasets/raw")
    parser.add_argument("--temp_path", type=str, default="./datasets/tmp")
    parser.add_argument("--config_path", type=str, default="./config/download_paths.yaml")
    parser.add_argument("--datasets", type=str, default="mvub, 8iVFBv2, uvg-vpc")

    args = parser.parse_args()
    return args

    
if __name__ == "__main__":
    # Parse arguments
    args = parse_args()
    with open(args.config_path) as f:
        config = yaml.safe_load(f)

    datasets = [dataset.strip() for dataset in args.datasets.split(",")]
    # Download MVUBO
    if "mvub" in datasets:
        download_jpeg_datasets(args, config, "mvub") 
    # Download 8iVFBv2
    if "8iVFBv2" in datasets:
        download_jpeg_datasets(args, config, "8iVFBv2")
    # Download UVG-VPC
    if "uvg-vpc" in datasets:
        download_jpeg_datasets(args, config, "uvg-vpc")


