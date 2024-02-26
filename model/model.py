import torch
import numpy as np
import MinkowskiEngine as ME
from compressai.models.base import CompressionModel

from .entropy_models import *
from .transforms import *
import utils

class ColorModel(CompressionModel):
    def __init__(self, config):
        super().__init__()

        self.g_a = AnalysisTransform(config["g_a"])
        self.g_s = SparseSynthesisTransform(config["g_s"])

        if config["entropy_model"]["type"] == "FactorizedPrior":
            self.entropy_model = FactorizedPrior(config["entropy_model"])
        elif config["entropy_model"]["type"] == "MeanScaleHyperprior":
            self.entropy_model = MeanScaleHyperpriorScales(config["entropy_model"])
        else:
            self.entropy_model = FactorizedPriorScaled(config["entropy_model"])




    def update(self):
        """
        Update the scale tables of entropy models
        """
        self.entropy_model.update(force=True)



    def aux_loss(self):
        """
        Get the aux loss of the entropy model
        """
        return self.entropy_model.aux_loss()
    


    def forward(self, x):
        """
        Parameters
        ----------
        x : ME.SparseTensor
            Input Tensor of geometry and features
        """
        # Save coords for decoding
        coords = ME.SparseTensor(coordinates=x.C.clone(),
                                 features=torch.ones(x.C.shape[0], 1),
                                 device=x.device)

        # Analysis Transform
        y = self.g_a(x)

        # Entropy Bottleneck
        y_hats, likelihoods = self.entropy_model(y)

        # Synthesis Transform(s)
        x_hats = []
        for y_hat in y_hats:
            x_hat = self.g_s(y_hat, coords=coords)
            x_hats.append(x_hat)
        
        # Building Output dictionaries
        output = {
            "prediction": x_hats,
            "likelihoods": {"y": likelihoods} if not isinstance(likelihoods, tuple) else {"y": likelihoods[0], "z": likelihoods[1]}
        }

        return output

    def compress(self, x, path=None, latent_path=None):
        """
        Compress a point cloud
        
        Parameters
        ----------
        x: torch.tensor, shape Nx6
            Tensor containing the point cloud, N is the number of points.
            Coordinates as first 3 dimensions, colors as last 3
        bin_path: str (default=None)
            path to store the binaries to, if None the compression is mocked
        
        returns
        -------
        strings: list
            List of strings (bitstreams), only returned if path=None
        shape: list
            List of shapes, only returned if path=None
        """
        N = x.shape[0]

        # Build input point cloud from tensors
        batch_vec = torch.zeros((N, 1), device=x.device)
        points = torch.cat([batch_vec, x[:, :3].contiguous()], dim=1)
        colors = x[:, 3:6].contiguous()

        # Minkowski Tensor
        input = ME.SparseTensor(coordinates=points.int(),
                                features=colors)

        # Analysis Transform
        y = self.g_a(input)
        
        # Entropy Coding
        strings, shape = self.entropy_model.compress(y, latent_path=latent_path)

        # Save the bitstream of return data
        if path:
            self.save_bitstream(path=path,
                                points=points, 
                                strings=strings, 
                                shape=shape)
        else:
            return strings, shape




    def decompress(self, 
                   path=None, 
                   coordinates=None, 
                   strings=None, 
                   shape=None):
        """
        Decompress a point cloud bitstream
        
        Parameters
        ----------
        path: str
            Path of the point cloud bitstream
        geometry: torch.tensor, Nx3
            Point Cloud geometry required to decode the attributes
        strings: list
            List of strings (bitstreams), only returned if path=None
        shape: list
            List of shapes, only returned if path=None
        
        returns
        -------
        x_hat: torch.tensor, Nx6
            Decompressed and reconstructed point cloud
        """
        N = coordinates.shape[0]

        # Decode the bitstream
        if path:
            strings, shape = self.load_bitstream(path)

        # Prepare the coordinates
        batch_vec = torch.zeros((N, 1), 
                                device=coordinates.device)
        coordinates = torch.cat([batch_vec, coordinates[:, :3].int()], 
                                dim=1)
        mock_features = torch.ones((coordinates.shape[0], 1), 
                                   device=coordinates.device)

        coordinates = ME.SparseTensor(coordinates=coordinates,
                                      features=mock_features)
    
        #latent_coordinates = utils.downsampled_coordinates(coordinates.C.clone(), factor=8)
        latent_coordinates = self.g_s.down_conv(coordinates)
        latent_coordinates = self.g_s.down_conv(latent_coordinates)
        latent_coordinates = self.g_s.down_conv(latent_coordinates)
        latent_coordinates_2 = ME.SparseTensor(coordinates=latent_coordinates.C.clone(), features=latent_coordinates.F.clone(), tensor_stride=8, device=coordinates.device)
        latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
        latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
        points = [latent_coordinates.C, latent_coordinates_2.C]

        # Entropy Decoding
        y_hat = self.entropy_model.decompress(points, strings, shape)

        # Synthesis transform
        x_hat = self.g_s(y_hat, coords=coordinates)

        # Rebuild reconstruction to torch tensor
        features = torch.clamp(torch.round(x_hat.F * 255), 0.0, 255.0) / 255
        x_hat = torch.concat([x_hat.C[:, 1:4], features], dim=1)
        return x_hat
        


    def load_bitstream(self,
                       path):
        """
        Load the bitstream from disk

        Parameters
        ----------
        path: str
            Path to the bitstream

        returns
        -------
        strings: list
            List of strings to be saved, each element is saved to a separate bitstream
        shape: list
            Shapes of the feature representation required for decoding
        """
        # Prepare paths
        path, ending = path.split(".", 1)
        aux_path = path + "_aux." + ending
        string_path = path + "_strings_{}" + ending

        # Strings
        strings = []
        for i in range(2):
            strings_path = string_path.format(i)
            with open(strings_path, "rb") as f:
                string = f.read()
                strings.append([string])

        # Aux info
        with open(aux_path, "rb") as f:
            shape = np.frombuffer(f.read(4), dtype=np.int32)

        return strings, shape



    def save_bitstream(self, 
                       path,
                       strings, 
                       shape):
        """
        Save the bitstream to file

        Parameters
        ----------
        path: str
            Path to store the data to
        strings: list
            List of strings to be saved, each element is saved to a separate bitstream
        shape: list
            Shapes of the feature representation required for decoding
        """
        # Prepare paths
        path, ending = path.split(".", 1)
        aux_path = path + "_aux." + ending
        string_path = path + "_strings_{}" + ending

        # Save strings
        for i, string in enumerate(strings):
            strings_path = string_path.format(i)
            with open(strings_path, "wb") as f:
                f.write(string[0])

        # Save aux_info
        with open(aux_path, "wb") as f:
            f.write(np.array(shape, dtype=np.int32).tobytes())


                
