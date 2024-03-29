import yaml
import os
import argparse
import random
import numpy as np

from tqdm import tqdm

import open3d as o3d 
import pandas as pd
import MinkowskiEngine as ME

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

from data.dataloader import StaticDataset
from data.transform import build_transforms
from data.utils.util import custom_collate_fn
from data.q_map import Q_Map

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
np.random.seed(0)
torch.use_deterministic_algorithms(True)

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

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

        if self.config["optimizer"] == "Adam":
            self.model_optimizer = optim.Adam(model_parameters, 
                                            lr=self.config["model_learning_rate"])
        elif self.config["optimizer"] == "SGD":
            self.model_optimizer = optim.SGD(model_parameters, 
                                            lr=self.config["model_learning_rate"],
                                            momentum=0.9)
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
                                       num_workers=12,
                                       pin_memory=False,
                                       worker_init_fn=seed_worker,
                                       collate_fn=custom_collate_fn)
        self.val_loader = DataLoader(valset,
                                     batch_size=1,                                       
                                     shuffle=False)

        self.q_map = Q_Map(self.config["q_map"])

        self.results = []
        self.epoch = 0

        self.check_resume()


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

        # Checkpoints
        ckpts_dir = os.path.join(self.results_directory, "ckpts")
        if not os.path.exists(ckpts_dir):
            os.mkdir(ckpts_dir)



    def check_resume(self):
        path = os.path.join(self.results_directory, "ckpts")
        ckpts = os.listdir(path)
        ckpts.sort()

        if len(ckpts) > 0:
            ckpt_dir = os.path.join(self.results_directory, "ckpts", ckpts[-1])
            self.load_checkpoint(ckpt_dir)
        

        
    def train(self):
        for epoch in range(self.epoch, self.config["epochs"]):
            # Training
            self.train_epoch(epoch)

            if ((epoch + 1)%10 == 0):
                self.val_epoch(epoch)

            self.model_scheduler.step()
            self.model.update()
            self.save_checkpoint(epoch+1)

        # Save model after training
        path = os.path.join(self.results_directory,
                            "weights.pt")
        self.model.update()
        torch.save(self.model.state_dict(), path)

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
            
            Q, Lambda = self.q_map(input)
            output = self.model(input, Q, Lambda)

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

                q_as = range(0, 2)
                q_gs = range(0, 2)
                for q_a in q_as:
                    for q_g in q_gs:
                        # Q Map
                        Q_map = ME.SparseTensor(coordinates=torch.cat([torch.zeros((N, 1), device=self.device), points[0]], dim=1), 
                                                features=torch.cat([torch.ones((N,1), device=self.device) * q_g, torch.ones((N,1), device=self.device) * q_a], dim=1),
                                                device=source.device)

                        # Compression
                        strings, shapes, k, coordinates = self.model.compress(source, Q_map)

                        # Run decompression
                        reconstruction = self.model.decompress(coordinates=coordinates, 
                                                                strings=strings, 
                                                                shape=shapes,
                                                                k=k)
                    
                        # Rebuild point clouds
                        source_pc = utils.get_o3d_pointcloud(source)
                        rec_pc = utils.get_o3d_pointcloud(reconstruction)

                        # Compute metrics
                        metric = PointCloudMetric(source_pc, rec_pc)
                        results, error_vectors = metric.compute_pointcloud_metrics(drop_duplicates=True)

                        # Save results
                        results["bpp"] = utils.count_bits(strings) / N
                        results["sequence"] = data["cubes"][0]["sequence"][0]
                        results["frameIdx"] = data["cubes"][0]["frameIdx"][0].item()
                        results["q_a"] = q_a
                        results["q_g"] = q_g
                        self.results.append(results)

                        # Renders of the pointcloud
                        point_size = 1.0 if data["cubes"][0]["sequence"][0] in ["longdress", "soldier", "loot", "longdress"] else 2.0
                        path = os.path.join(self.results_directory,
                                            "renders_val", 
                                            "{}_a{}_g{}_{}.png".format(sequence, str(q_a), str(q_g), "{}"))
                        utils.render_pointcloud(rec_pc, path, point_size=point_size)

        # Save the results as .csv
        df = pd.DataFrame(self.results)
        results_path = os.path.join(self.results_directory, "val.csv")
        df.to_csv(results_path)

    def save_checkpoint(self, epoch):
        """
        Save a checkpoint of the model
        """
        self.model.update()

        checkpoint = {}
        checkpoint["epoch"] = epoch
        checkpoint["results"] = self.results
        checkpoint["model"] = self.model.state_dict()
        checkpoint["model_optimizer"] = self.model_optimizer.state_dict()
        checkpoint["model_scheduler"] = self.model_scheduler.state_dict()
        checkpoint["bottleneck_optimizer"] = self.bottleneck_optimizer.state_dict()

        path = os.path.join(self.results_directory, "ckpts", "ckpt_{:03d}.pt".format(epoch))
        torch.save(checkpoint, path)

    def load_checkpoint(self, path):
        """
        Load a checkpoint of the model
        
        Parameters
        ----------
        path: str
            Path to load
        """
        checkpoint = torch.load(path)
        self.epoch = checkpoint["epoch"]
        self.results = checkpoint["results"]
        self.model.load_state_dict(checkpoint["model"])
        self.model_optimizer.load_state_dict(checkpoint["model_optimizer"])
        self.model_scheduler.load_state_dict(checkpoint["model_scheduler"])
        self.bottleneck_optimizer.load_state_dict(checkpoint["bottleneck_optimizer"])



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
    args = parse_options()
    config = args.config
    run = Training(config)
    run.train()