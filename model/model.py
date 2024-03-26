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

        self.entropy_model = MeanScaleHyperprior(config["entropy_model"])
        if "entropy_model_map" in config.keys():
            self.entropy_model_map = MeanScaleHyperprior(config["entropy_model_map"])
        else:
            self.entropy_model_map = None


    def update(self):
        """
        Update the scale tables of entropy models
        """
        self.entropy_model.update(force=True)
        if self.entropy_model_map is not None:
            self.entropy_model_map.update(force=True)



    def aux_loss(self):
        """
        Get the aux loss of the entropy model
        """
        if self.entropy_model_map is None:
            return self.entropy_model.aux_loss()
        else:
            return self.entropy_model.aux_loss() + self.entropy_model_map.aux_loss()
    


    def forward(self, x, Q, Lambda):
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

        # Pad input tensor
        x = ME.SparseTensor(coordinates=x.C.clone(),
                            features=torch.cat([torch.ones((x.C.shape[0], 1), device=x.device), x.F], dim=1))

        # Analysis Transform
        y, Q, k = self.g_a(x, Q)

        # Entropy Bottleneck
        if not self.entropy_model_map:
            y = ME.SparseTensor(coordinates=y.c,
                                features=torch.cat[y.F, Q.features_at_coordinates(y.C.float(), dim=1)],
                                tensor_stride=y.tensor_stride)
            y_hat, likelihoods = self.entropy_model(y)
            # Split coords after entropy coding
            Q_hat = ME.SparseTensor(coordinates=y_hat.C, features=y_hat.F[:, 128:], device=y.device, tensor_stride=8)
            y_hat = ME.SparseTensor(coordinates=y_hat.C, features=y_hat.F[:, :128], device=y.device, tensor_stride=8)
            likelihoods = {"y": likelihoods[0], "z": likelihoods[1]}
        else:
            y_hat, y_likelihoods = self.entropy_model(y)
            Q_hat, Q_likelihoods = self.entropy_model_map(Q)
            likelihoods = {"y": [y_likelihoods[0], Q_likelihoods[0]], "z" : [y_likelihoods[1], Q_likelihoods[1]]}


        # Synthesis Transform(s)
        x_hat, points, predictions = self.g_s(y_hat, Q_hat, coords=coords, k=k)
        
        # Building Output dictionaries
        output = {
            "prediction": x_hat,
            "points": points,
            "occ_predictions": predictions,
            "q_map": Lambda,
            "likelihoods": likelihoods
        }

        return output

    def compress(self, x, Q, path=None, latent_path=None):
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
                                features=torch.cat([torch.ones((N, 1), device=x.device), colors], dim=1))

        # Analysis Transform
        y, Q, k = self.g_a(input, Q)
        
        # Entropy Bottleneck
        if not self.entropy_model_map:
            y = ME.SparseTensor(coordinates=y.C,
                                features=torch.cat([y.F, Q.features_at_coordinates(y.C.float())], dim=1),
                                tensor_stride=y.tensor_stride)
            points, strings, shape = self.entropy_model.compress(y)
            # Split coords after entropy coding
        else:
            points, y_strings, y_shape = self.entropy_model.compress(y)
            _, Q_strings, Q_shape = self.entropy_model_map.compress(Q)
            strings = [y_strings, Q_strings]
            shape = [y_shape, Q_shape]

        coordinates = y.C

        # Save the bitstream of return data
        if path:
            print("Not implemented in model.py")
            self.save_bitstream(path=path,
                                points=points, 
                                strings=strings, 
                                shape=shape)
        else:
            return strings, shape, k, coordinates




    def decompress(self, 
                   path=None, 
                   coordinates=None, 
                   strings=None, 
                   shape=None,
                   k=None):
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
        k: list
            Number of points at each stage
        
        returns
        -------
        x_hat: torch.tensor, Nx6
            Decompressed and reconstructed point cloud
        """
        # Decode the bitstream
        if path:
            strings, shape = self.load_bitstream(path)

        # Prepare the coordinates
        latent_coordinates_2 = ME.SparseTensor(coordinates=coordinates.clone(), features=torch.ones((coordinates.shape[0], 1)), tensor_stride=8, device=coordinates.device)
        latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
        latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
        points = [coordinates, latent_coordinates_2.C]

        # Entropy Decoding
        if not self.entropy_model_map:
            y_hat = self.entropy_model.decompress(points, strings, shape)
            # Split coords after entropy coding
            Q_hat = ME.SparseTensor(coordinates=y_hat.C, features=y_hat.F[:, 128:], device=coordinates.device, tensor_stride=8)
            y_hat = ME.SparseTensor(coordinates=y_hat.C, features=y_hat.F[:, :128], device=coordinates.device, tensor_stride=8)
        else:
            y_strings, Q_strings = strings[0], strings[1]
            y_shape, Q_shape = shape[0], shape[1]
            y_hat = self.entropy_model.decompress(points, y_strings, y_shape)
            Q_hat = self.entropy_model_map.decompress(points, Q_strings, Q_shape)

        # Synthesis transform
        x_hat = self.g_s(y_hat, Q_hat, k=k)

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