import yaml
import os
import argparse
import random
import numpy as np

from tqdm import tqdm

import open3d as o3d 
import MinkowskiEngine as ME

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.dataloader import StaticDataset
from data.transform import build_transforms
from data.utils.util import custom_collate_fn

from metrics.metric import PointCloudMetric

from model.model import ColorModel
from loss import Loss
import utils

TQDM_BAR_FORMAT_TRAIN = "[Train]\t{l_bar}{bar:10}{r_bar}\t"
TQDM_BAR_FORMAT_VAL = "[Val]\t{l_bar}{bar:10}{r_bar}\t"

# Determinism
os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
torch.manual_seed(0)
random.seed(0)
torch.use_deterministic_algorithms(True)
np.random.seed(0)

class Training():
    def __init__(self, config_path):
        self.load_config(config_path)

        # Setup Folders
        self.setup_folders()

        # Training Settings
        self.device = torch.device(int(self.config["device"]))
        torch.cuda.set_device(self.device)
        
        # Model
        self.model = ColorModel(self.config["model"])
        self.model.to(self.device)

        # Optimizer
        model_parameters = [p for n,p in self.model.named_parameters() if not n.endswith(".quantiles")]
        bottleneck_parameters = [p for n,p in self.model.named_parameters() if n.endswith(".quantiles")]

        self.model_optimizer = optim.Adam(model_parameters, 
                                          lr=self.config["model_learning_rate"])
        self.bottleneck_optimizer = optim.Adam(bottleneck_parameters,
                                                lr=self.config["bottleneck_learning_rate"])

        self.model_scheduler = optim.lr_scheduler.StepLR(self.model_optimizer, 
                                                         step_size=self.config["scheduler_step_size"], 
                                                         gamma=self.config["scheduler_gamma"])
        
        # Loss
        self.loss = Loss(self.config["loss"])

        # Data
        train_transform = Compose(build_transforms(self.config["transforms"]["train"]))
        trainset = StaticDataset(self.config["data_path"],
                                 split="train",
                                 transform=train_transform,
                                 min_points=self.config["min_points_train"])
        valset = StaticDataset(self.config["data_path"],
                                 split="val",
                                 transform=None,
                                 partition=False)

        self.train_loader = DataLoader(trainset,
                                       batch_size=self.config["batch_size"],                                       
                                       shuffle=True,
                                       num_workers=8,
                                       pin_memory=False,
                                       collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(valset,
                                     batch_size=1,                                       
                                     shuffle=False)
            


    def load_config(self, config_path):
        """
        Loads the config for the training
        """
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)

    def setup_folders(self):
        """
        Sets up the training:
            - Initialize folder structure for results
        """
        # Initialize Results folder structure
        self.results_directory = os.path.join(self.config["results_path"], self.config["experiment_name"])

        if not os.path.exists(self.results_directory):
            os.mkdir(self.results_directory)

        config_path =  os.path.join(self.results_directory, "config.yaml")
        with open(config_path, "w") as config_file:
            yaml.safe_dump(self.config, config_file)
        
        
    def train(self):
        for epoch in range(self.config["epochs"]):
            # Training
            self.train_epoch(epoch)

            if ((epoch)%10 == 0):
                continue
                self.val_epoch(epoch)

            self.model_scheduler.step()
            self.model.update()

        # Save model after training
        path = os.path.join(self.results_directory,
                            "weights.pt")
        self.save_checkpoint(path)

    def train_epoch(self, epoch):
        self.model.train()

        loss_avg = utils.AverageMeter()
        aux_loss_avg = utils.AverageMeter()

        pbar = tqdm(self.train_loader, bar_format=TQDM_BAR_FORMAT_TRAIN)
        pbar.set_description("[{}: {}/{}]".format(self.config["experiment_name"], 
                                                  str(epoch + 1), 
                                                  str(self.config["epochs"])))
        for i, data in enumerate(pbar):
            self.model_optimizer.zero_grad()
            self.bottleneck_optimizer.zero_grad()

            coords, feats = ME.utils.collation.sparse_collate(data["points"], 
                                                              data["colors"], 
                                                              device=self.device)
            # Input data
            input = ME.SparseTensor(features=feats,
                                    coordinates=coords,
                                    device=self.device)
            
            output = self.model(input)

            # Backward for model
            loss_value, loss_dict = self.loss(input, output)
            loss_avg.update(loss_value.item())
            loss_value.backward()

            torch.cuda.empty_cache() # Needed?

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                          self.config["clip_grad_norm"])

            self.model_optimizer.step()
            
            # Backward for bottleneck
            aux_loss = self.model.aux_loss()
            aux_loss_avg.update(aux_loss.item())
            aux_loss.backward()

            self.bottleneck_optimizer.step()

            # Logging
            pbar_dict = {}
            pbar_dict["Loss"] = "{:.2e}".format(loss_avg.avg)
            pbar_dict["Aux_Loss"] = "{:.2e}".format(aux_loss_avg.avg)
            #for key, value in loss_dict.items():
                #pbar_dict[key] = "{:.2e}".format(self.log.get_avg([self.finished_epochs, "training", key]))
            pbar.set_postfix(pbar_dict)

    def val_epoch(self, epoch):
        self.model.eval()
        self.model.update()
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, bar_format=TQDM_BAR_FORMAT_VAL)

            for _, data in enumerate(pbar):
                # Prepare data
                points = data["src"]["points"].to(self.device, dtype=torch.float)
                colors = data["src"]["colors"].to(self.device, dtype=torch.float)
                source = torch.concat([points, colors], dim=2)[0]
                coordinates = source.clone()

                # Side info
                N = source.shape[0]
                sequence = data["cubes"][0]["sequence"][0]

                # Compression
                strings, shapes = self.model.compress(source)

                # Decompress all rates
                y_strings = []
                z_strings = []
                for i in range(len(strings[0])):
                    y_strings.append(strings[0][i])
                    z_strings.append(strings[1][i])
                    current_strings = [y_strings, z_strings]

                    # Run decompression
                    reconstruction = self.model.decompress(coordinates=coordinates, 
                                                           strings=current_strings, 
                                                           shape=shapes)
                    
                    # Compute bpp
                    bpp = utils.count_bits(current_strings) / N
                    print(bpp)

                    # Rebuild point clouds
                    source_pc = utils.get_o3d_pointcloud(source)
                    rec_pc = utils.get_o3d_pointcloud(reconstruction)

                    # Compute metrics
                    metric = PointCloudMetric(source_pc, rec_pc)
                    results, _ = metric.compute_pointcloud_metrics(drop_duplicates=True)
                    print(results["sym_y_psnr"])

                    # Renders
                    path = os.path.join(self.results_directory, 
                                        "renders_val", 
                                        "{}_{}_{}_{}.png".format(str(epoch), sequence, str(i), "{}"))
                    utils.render_pointcloud(rec_pc, path)

    def save_checkpoint(self, path):
        """
        Save a checkpoint of the model
        
        Parameters
        ----------
        epoch: int
            Current epoch
        """
        self.model.update()
        torch.save(self.model.state_dict(), path)

def parse_options():
    """
    Parse options into a Options object
    """
    parser = argparse.ArgumentParser()
    # Training options
    parser.add_argument("--config", type=str, default="./configs/MeanScale_5_lambda200-3200.yaml", help="Configuration for training")
    #Parse into option class
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # Clean up for debugging
    import shutil
    #shutil.rmtree('./results/Ours_test')
    
    args = parse_options()
    config = args.config
    run = Training(config)
    run.train()