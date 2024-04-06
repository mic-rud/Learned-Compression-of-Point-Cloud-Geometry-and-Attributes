import torch
import numpy as np
import MinkowskiEngine as ME
from compressai.models.base import CompressionModel

import os
import subprocess
import open3d as o3d
from bitstream import BitStream

from .entropy_models import *
from .transforms import *
import utils

class ColorModel(CompressionModel):
    def __init__(self, config):
        super().__init__()

        self.g_a = AnalysisTransform(config["g_a"])
        self.g_s = SparseSynthesisTransform(config["g_s"])

        if "entropy_model_map" in config.keys():
            self.entropy_model = MeanScaleHyperprior(config["entropy_model"])
            self.entropy_model_map = MeanScaleHyperprior(config["entropy_model_map"])
        else:
            self.entropy_model = MeanScaleHyperprior_Map(config["entropy_model"])
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
        if isinstance(self.entropy_model, MeanScaleHyperprior_Map):
            y_hat, Q_hat, likelihoods = self.entropy_model(y)
            # Split coords after entropy coding
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

    def compress(self, x, Q, path=None):
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
        if isinstance(self.entropy_model, MeanScaleHyperprior_Map):
            points, strings, shape = self.entropy_model.compress(y)
        else:
            points, y_strings, y_shape = self.entropy_model.compress(y)
            _, Q_strings, Q_shape = self.entropy_model_map.compress(Q)
            strings = [y_strings, Q_strings]
            shape = [y_shape, Q_shape]

        coordinates = y.C

        # Save the bitstream of return data
        if path:
            self.save_bitstream(path=path,
                                points=coordinates, 
                                strings=strings, 
                                shape=shape,
                                k=k)
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
        device = self.g_s.down_conv.kernel.device # look up the model device
        # Decode the bitstream
        if path:
            coordinates, strings, shape, k = self.load_bitstream(path)
            coordinates = coordinates.to(device)
            batch_vec = torch.zeros((coordinates.shape[0], 1), device=coordinates.device)
            coordinates = torch.cat([batch_vec, coordinates.contiguous()], dim=1)

        # Prepare the coordinates
        latent_coordinates_2 = ME.SparseTensor(coordinates=coordinates.clone(), features=torch.ones((coordinates.shape[0], 1)), tensor_stride=8, device=coordinates.device)
        latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
        latent_coordinates_2 = self.g_s.down_conv(latent_coordinates_2)
        points = [coordinates, latent_coordinates_2.C]

        # Entropy Decoding
        if isinstance(self.entropy_model, MeanScaleHyperprior_Map):
            y_hat, Q_hat = self.entropy_model.decompress(points, strings, shape)
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
        




    def save_bitstream(self, 
                       path,
                       points,
                       strings, 
                       shape,
                       k):
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
        #path, ending = path.split(".", 1)

        # Initialize Bitstream
        stream = BitStream()

        # Encode points with G-PCC
        points_bitstream = self.gpcc_encode(points, path)

        ## Write header
        # Shape
        stream.write(shape, np.int32)
        stream.write(len(points_bitstream), np.int32)

        # String lengths
        for i, string in enumerate(strings):
            stream.write(len(string[0]), np.int32)
        for i, ks in enumerate(k):
            stream.write(ks, np.int32)

        # Write content
        stream.write(points_bitstream)

        for i, string in enumerate(strings):
            stream.write(string[0])


        bit_string = stream.__str__()
        byte_array = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

        with open(path, "wb") as binary:
            binary.write(byte_array)



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
        stream = BitStream()
        with open(path, "rb") as binary:
            data = binary.read()

        stream = BitStream()
        stream.write(data, bytes)

        # Header
        shape = [int(stream.read(np.uint32))]
        len_points_bitstream = stream.read(np.uint32)
        len_string_1 = stream.read(np.uint32)
        len_string_2 = stream.read(np.uint32)
        string_lengths = [len_string_1, len_string_2]
        k1 = stream.read(np.uint32)
        k2 = stream.read(np.uint32)
        k3 = stream.read(np.uint32)
        k = [[k1], [k2], [k3]]

        # Payload
        points_bitstream = stream.read(int(len_points_bitstream)*8)

        strings = []
        for i in range(2):
            string = stream.read(int(string_lengths[i])*8)
            bit_string = string.__str__()
            byte_string = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))
            strings.append([byte_string])

        coordinates = self.gpcc_decode(points_bitstream, path)

        return coordinates, strings, shape, k


    def gpcc_encode(self, points, directory):
        """
        Encode a list of points with G-PCC
        """
        directory, _ = os.path.split(directory)
        tmp_dir = os.path.join(directory, "points_enc.ply")
        bin_dir = os.path.join(directory, "points_enc.bin")

        # Save points as ply
        dtype = o3d.core.float32
        p_tensor = o3d.core.Tensor(points.detach().cpu().numpy()[:, 1:], dtype=dtype)
        pc = o3d.t.geometry.PointCloud(p_tensor)
        o3d.t.io.write_point_cloud(tmp_dir, pc, write_ascii=True)

        # G-PCC
        subp=subprocess.Popen('./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3'+ 
                                ' --mode=0' + 
                                ' --positionQuantizationScale=1' + 
                                ' --trisoupNodeSizeLog2=0' + 
                                ' --neighbourAvailBoundaryLog2=8' + 
                                ' --intra_pred_max_node_size_log2=6' + 
                                ' --inferredDirectCodingMode=0' + 
                                ' --maxNumQtBtBeforeOt=4' +
                                ' --uncompressedDataPath='+tmp_dir + 
                                ' --compressedStreamPath='+bin_dir, 
                                shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Read stdout and stderr
        stdout, stderr = subp.communicate()

        # Print the outputs
        if subp.returncode != 0:
            print("Error occurred:")
            print(stderr.decode())
            c=subp.stdout.readline()

        # Read the bytes to return
        with open(bin_dir, "rb") as binary:
            data = binary.read()
        
        # Clean up
        os.remove(tmp_dir)
        os.remove(bin_dir)

        return data



    def gpcc_decode(self, bin, directory):
        directory, _ = os.path.split(directory)
        tmp_dir = os.path.join(directory, "points_dec.ply")
        bin_dir = os.path.join(directory, "points_dec.bin")
        
        # Write to file
        bit_string = bin.__str__()
        byte_array = bytes(int(bit_string[i:i+8], 2) for i in range(0, len(bit_string), 8))

        with open(bin_dir, "wb") as binary:
            binary.write(byte_array)
        subp=subprocess.Popen('./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3'+ 
                                ' --mode=1'+ 
                                ' --compressedStreamPath='+bin_dir+ 
                                ' --reconstructedDataPath='+tmp_dir+
                                ' --outputBinaryPly=0',
                                shell=True, stdout=subprocess.PIPE)
        c=subp.stdout.readline()
        while c:
            c=subp.stdout.readline()
            #print(c)
    
        # Load ply
        pcd = o3d.io.read_point_cloud(tmp_dir)
        points = torch.tensor(pcd.points)

        # Clean up
        os.remove(tmp_dir)
        os.remove(bin_dir)
        return points
