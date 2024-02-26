import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

import utils

from compressai.entropy_models import EntropyBottleneck, GaussianConditional 
from compressai.models.base import CompressionModel

class FactorizedPrior(CompressionModel):
    """
    Factorized Prior Entropy Bottleneck

    Uses a factorized prior model for the latents
    """
    def __init__(self, config):
        super().__init__()
        self.C = config["C_in"]
        self.C_bottleneck = config["C_bottleneck"]
        self.num_bottlenecks = config["num_bottlenecks"]

        self.entropy_bottlenecks = []
        self.extracts = []
        self.expands = []

        for i in range(self.num_bottlenecks):
            # Entropy Bottleneck
            entropy_bottleneck = EntropyBottleneck(self.C_bottleneck)
            self.entropy_bottlenecks.append( entropy_bottleneck )
            self.add_module(f'entropy_bottleneck_{i}', entropy_bottleneck)

            # Extract layers
            extract = nn.Sequential(
                ME.MinkowskiConvolution(self.C, self.C, kernel_size=1, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(self.C, self.C, kernel_size=3, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(self.C, self.C_bottleneck, kernel_size=1, dimension=3),
            ) 
            """
            extract = nn.Sequential(
                ME.MinkowskiChannelwiseConvolution(self.C, kernel_size=3, dimension=3),
                ME.MinkowskiTanh(),
                #ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiChannelwiseConvolution(self.C, kernel_size=3, dimension=3),
                ME.MinkowskiTanh(),
                #ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiChannelwiseConvolution(self.C, kernel_size=3, dimension=3)
            ) 
            """
            self.extracts.append(extract)
            self.add_module(f'extract_{i}',extract)

            expand = nn.Sequential(
                ME.MinkowskiConvolution(self.C_bottleneck, self.C, kernel_size=1, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(self.C, self.C, kernel_size=3, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(self.C, self.C, kernel_size=1, dimension=3)
            )
            """
            expand = nn.Sequential(
                #ME.MinkowskiChannelwiseConvolution(self.C_bottleneck, self.C, kernel_size=3, dimension=3),
                ME.MinkowskiChannelwiseConvolution(self.C, kernel_size=3, dimension=3),
                #ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiTanh(),
                #ME.MinkowskiChannelwiseConvolution(self.C, self.C, kernel_size=3, dimension=3),
                ME.MinkowskiChannelwiseConvolution(self.C, kernel_size=3, dimension=3),
                #ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiTanh(),
                #ME.MinkowskiChannelwiseConvolution(self.C, self.C, kernel_size=3, dimension=3)
                ME.MinkowskiChannelwiseConvolution(self.C, kernel_size=3, dimension=3)
            )
            """

            self.expands.append(expand)
            self.add_module(f'expand_{i}',expand)

    
    def forward(self, y):
        # Entropy model
        y_hats, y_likelihoods = [], []
        residuals = []

        y_base = ME.SparseTensor(coordinates=y.C, 
                                 features=y.F.clone(), 
                                 device=y.device, 
                                 tensor_stride=8)
        for i in range(self.num_bottlenecks):
            # Extract residual
            r_i = self.extracts[i](y_base)

            # Entropy bottleneck
            r_hat, likelihood = self.entropy_bottlenecks[i](r_i.F.t().unsqueeze(0))
            r_hat = ME.SparseTensor(coordinates=r_i.C,
                                    features=r_hat[0].t(),
                                    device=y.device,
                                    tensor_stride=8)

            # Expand
            r_i_hat = self.expands[i](r_hat)

            # Sum over residuals
            y_hat = r_i_hat.features_at_coordinates(y.C.float())
            with torch.no_grad():
                for residual in residuals:
                    y_hat += residual.features_at_coordinates(y.C.float())
                        
            # Append to y_hats
            y_hat = ME.SparseTensor(coordinates=y.C,
                                    features=y_hat.clone(),
                                    device=y.device,
                                    tensor_stride=8)
            y_hats.append(y_hat)

            y_likelihoods.append(likelihood)

            # Rounded residual 
            with torch.no_grad():
                r_round, likelihoods_round = self.entropy_bottlenecks[i](r_i.F.t().unsqueeze(0), training=False)
                r_round = ME.SparseTensor(coordinates=r_i.C,
                                          features=r_round[0].t(),
                                          tensor_stride=8)
                r_i_round = self.expands[i](r_round)
                residuals.append(r_i_round)

                #y_likelihoods_round.append(likelihoods_round)

                # Subtract from y_base
                y_base = ME.SparseTensor(coordinates=y_base.C,
                                        features=y_base.F - r_i_round.F,
                                        tensor_stride=8,
                                        device=y_base.device)

        return y_hats, y_likelihoods

    def compress(self, y):
        """
        Compression of a latent space
        
        Parameters
        ----------
        y: MinkowskiEngine.SparseTensor
            Sparse Tensor containing the latent features

        returns
        ----------
        strings: list
            List of strings representing the compressed bitstream
        shape: np.array
            Shape of the compressed latent features
        """
        shapes = []
        y_strings = []
        y = utils.sort_tensor(y)
        y_base = ME.SparseTensor(coordinates=y.C.clone(), 
                                 features=y.F.clone(), 
                                 device=y.device, 
                                 tensor_stride=8)
        #ys = torch.split(y.F, int(self.C / self.num_bottlenecks), dim=1)
        for i in range(self.num_bottlenecks):
            """
            y_base = ME.SparseTensor(coordinates=y.C.clone(), 
                                 features=ys[i], 
                                 device=y.device, 
                                 tensor_stride=8)
            """
            # Extract residual
            r_i = self.extracts[i](y_base)

            # Shape
            shape = [r_i.F.shape[0]]
            shapes.append(shape)

            # Entropy bottleneck compress
            r_i = utils.sort_tensor(r_i)
            y_string = self.entropy_bottlenecks[i].compress(r_i.F.t().unsqueeze(0))
            y_strings.append(y_string)
            
            # Entropy bottleneck decompress
            r_hat = self.entropy_bottlenecks[i].decompress(y_string, shape)
            r_hat = ME.SparseTensor(coordinates=r_i.C,
                                    features=r_hat[0].t(),
                                    tensor_stride=8,
                                    device=r_hat.device)
            
            # Expand
            r_i_hat = self.expands[i](r_hat)

            # Rounded residual 
            y_base = ME.SparseTensor(coordinates=y_base.C,
                                     features=y_base.F - r_i_hat.features_at_coordinates(y_base.C.float()),
                                     tensor_stride=8,
                                     device=y_base.device)
            # Render a pc
            import open3d as o3d
            import numpy as np
            coordinates = y_base.C.cpu().numpy()
            features = y_base.F.cpu().numpy()
            for i in range(features.shape[1]):
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                color_value = features[:, i:i+1]
                color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                utils.render_pointcloud(point_cloud, "temp/image_{}_{}.png".format(i, "{}"))

        strings = [y_strings]
        shapes = [shape]
        return strings, shapes

    def decompress(self, points, strings, shapes):
        """
        Decompression routine, the length of strings determines the quality of decompression

        Parameters
        ----------
        points: torch.tensor, Nx4
            Tensor of points for the latent features with indices B,x,y,z
        strings: list
            List of strings representing the bitstream
        shapes: list
            List of shapes, containing the dimension of the compressed features
        
        returns
        -------
        y_hat: MinkowskiEngine.sparseTensor
            Sparse Tensor of the decompressed representation
        
        """
        points = points[0]
        assert isinstance(strings, list) and len(strings) == 1
        points = utils.sort_points(points)
        # Get the points back
        y_strings = strings[0]
        shape = shapes[0]

        residuals = []
        for i in range(len(y_strings)):
            # Decompress Strings
            r_hat = self.entropy_bottlenecks[i].decompress(y_strings[i], shape)
            r_hat = ME.SparseTensor(features=r_hat[0].t(),
                                    coordinates=points,
                                    tensor_stride=8,
                                    device=points.device)

            r_hat = self.expands[i](r_hat)

            residuals.append(r_hat.F.clone())

        y_hat_feats = residuals[-1]
        for res in residuals[:-1]:
            y_hat_feats += res

        y_hat = ME.SparseTensor(features=y_hat_feats, 
                                coordinates=points, 
                                tensor_stride=8, 
                                device=points.device)

        return y_hat


class FactorizedPriorScaled(CompressionModel):
    """
    Factorized Prior Entropy Bottleneck

    Uses a factorized prior model for the latents
    """
    def __init__(self, config):
        super().__init__()
        self.C = config["C_in"]
        self.C_bottleneck = config["C_bottleneck"]
        self.num_bottlenecks = config["num_bottlenecks"]

        self.entropy_bottlenecks = []
        self.extracts = []
        self.expands = []

        for i in range(self.num_bottlenecks):
            # Entropy Bottleneck
            entropy_bottleneck = EntropyBottleneck(self.C_bottleneck)
            self.entropy_bottlenecks.append( entropy_bottleneck )
            self.add_module(f'entropy_bottleneck_{i}', entropy_bottleneck)

            # Extract layers
            if i >= 1:
                extract = nn.Sequential(
                    ME.MinkowskiConvolution(self.C, self.C_bottleneck * 2, kernel_size=3, dimension=3),
                    ME.MinkowskiReLU(inplace=False),
                    ME.MinkowskiConvolution(self.C_bottleneck * 2, self.C_bottleneck * 2,  kernel_size=3, dimension=3, stride=2),
                    ME.MinkowskiReLU(inplace=False),
                    ME.MinkowskiConvolution(self.C_bottleneck * 2, self.C_bottleneck,  kernel_size=3, dimension=3)
                )
                self.extracts.append(extract)
                self.add_module(f'extract_{i}',extract)
                 

            if i >= 1:
                expand = nn.Sequential(
                    ME.MinkowskiConvolution(self.C_bottleneck, self.C_bottleneck * 2, kernel_size=3, dimension=3),
                    ME.MinkowskiReLU(inplace=False),
                    ME.MinkowskiGenerativeConvolutionTranspose(self.C_bottleneck*2, self.C_bottleneck*2, kernel_size=3, dimension=3),
                    ME.MinkowskiReLU(inplace=False),
                    ME.MinkowskiConvolution(self.C_bottleneck * 2, self.C, kernel_size=3, dimension=3)
                )
                self.expands.append(expand)
                self.add_module(f'expand_{i}',expand)

        self.down_conv = ME.MinkowskiConvolution(in_channels=self.C_bottleneck, out_channels=1, kernel_size=3, stride=2, dimension=3)

    
    def forward(self, y):
        # Entropy model
        y_hats, y_likelihoods = [], []
        residuals = []

        y_base = ME.SparseTensor(coordinates=y.C, 
                                 features=y.F.clone(), 
                                 device=y.device, 
                                 tensor_stride=8)
        for i in range(self.num_bottlenecks):
            # Extract residual
            if i >= 1:
                r_i = self.extracts[i-1](y_base)
            else:
                r_i = y_base 

            # Entropy bottleneck
            r_hat, likelihood = self.entropy_bottlenecks[i](r_i.F.t().unsqueeze(0))
            r_hat = ME.SparseTensor(coordinates=r_i.C,
                                    features=r_hat[0].t(),
                                    device=y.device,
                                    tensor_stride=8)

            # Expand
            if i >= 1:
                r_i_hat = self.expands[i-1](r_hat)
            else:
                r_i_hat = r_hat 

            # Sum over residuals
            y_hat = r_i_hat.features_at_coordinates(y.C.float())
            with torch.no_grad():
                for residual in residuals:
                    y_hat += residual.features_at_coordinates(y.C.float())
                        
            # Append to y_hats
            y_hat = ME.SparseTensor(coordinates=y.C,
                                    features=y_hat.clone(),
                                    device=y.device,
                                    tensor_stride=8)
            y_hats.append(y_hat)

            y_likelihoods.append(likelihood) # Try for likelihood once

            # Rounded residual 
            with torch.no_grad():
                r_round, likelihoods_round = self.entropy_bottlenecks[i](r_i.F.t().unsqueeze(0), training=False)
                r_round = ME.SparseTensor(coordinates=r_i.C,
                                          features=r_round[0].t(),
                                          tensor_stride=8)
                if i >= 1:
                    r_i_round = self.expands[i-1](r_round)
                else:
                    r_i_round = r_round

                residuals.append(r_i_round)

                # Subtract from y_base
                y_base = ME.SparseTensor(coordinates=y_base.C,
                                        features=y_base.F - r_i_round.features_at_coordinates(y_base.C.float()),
                                        tensor_stride=8,
                                        device=y_base.device)

        return y_hats, y_likelihoods

    def compress(self, y):
        """
        Compression of a latent space
        
        Parameters
        ----------
        y: MinkowskiEngine.SparseTensor
            Sparse Tensor containing the latent features

        returns
        ----------
        strings: list
            List of strings representing the compressed bitstream
        shape: np.array
            Shape of the compressed latent features
        """
        shapes = []
        y_strings = []
        y = utils.sort_tensor(y)
        y_base = ME.SparseTensor(coordinates=y.C.clone(), 
                                 features=y.F.clone(), 
                                 device=y.device, 
                                 tensor_stride=8)
        #ys = torch.split(y.F, int(self.C / self.num_bottlenecks), dim=1)
        for i in range(self.num_bottlenecks):
            """
            y_base = ME.SparseTensor(coordinates=y.C.clone(), 
                                 features=ys[i], 
                                 device=y.device, 
                                 tensor_stride=8)
            """
            # Extract residual
            if i >= 1:
                r_i = self.extracts[i-1](y_base)
            else:
                r_i = y_base 

            # Shape
            shape = [r_i.F.shape[0]]
            shapes.append(shape)

            # Entropy bottleneck compress
            r_i = utils.sort_tensor(r_i)
            y_string = self.entropy_bottlenecks[i].compress(r_i.F.t().unsqueeze(0))
            y_strings.append(y_string)
            
            # Entropy bottleneck decompress
            r_hat = self.entropy_bottlenecks[i].decompress(y_string, shape)
            
            # Expand
            if i >= 1:
                r_hat = ME.SparseTensor(coordinates=r_i.C,
                                    features=r_hat[0].t(),
                                    tensor_stride=16,
                                    device=r_hat.device)
                r_i_hat = self.expands[i-1](r_hat)
            else:
                r_i_hat = ME.SparseTensor(coordinates=r_i.C,
                                    features=r_hat[0].t(),
                                    tensor_stride=8,
                                    device=r_hat.device)

            # Rounded residual 
            y_base = ME.SparseTensor(coordinates=y_base.C,
                                     features=y_base.F - r_i_hat.features_at_coordinates(y_base.C.float()),
                                     tensor_stride=8,
                                     device=y_base.device)
            # Render a pc
            import open3d as o3d
            import numpy as np
            coordinates = y_base.C.cpu().numpy()
            features = y_base.F.cpu().numpy()
            for i in range(8):
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                color_value = features[:, i:i+1]
                color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                utils.render_pointcloud(point_cloud, "temp/image_{}_{}.png".format(i, "{}"))

        strings = [y_strings]
        return strings, shapes

    def decompress(self, points, points2, strings, shapes):
        """
        Decompression routine, the length of strings determines the quality of decompression

        Parameters
        ----------
        points: torch.tensor, Nx4
            Tensor of points for the latent features with indices B,x,y,z
        strings: list
            List of strings representing the bitstream
        shapes: list
            List of shapes, containing the dimension of the compressed features
        
        returns
        -------
        y_hat: MinkowskiEngine.sparseTensor
            Sparse Tensor of the decompressed representation
        
        """
        assert isinstance(strings, list) and len(strings) == 1
        points = utils.sort_points(points)
        points2 = utils.sort_points(points2)
        # Get the points back
        y_strings = strings[0]

        residuals = []
        for i in range(len(y_strings)):
            # Decompress Strings
            r_hat = self.entropy_bottlenecks[i].decompress(y_strings[i], shapes[i])

            # Expand
            if i >= 1:
                r_hat = ME.SparseTensor(features=r_hat[0].t(),
                                    coordinates=points2,
                                    tensor_stride=16,
                                    device=points.device)
                r_hat = self.expands[i-1](r_hat)
            else:
                r_hat = ME.SparseTensor(features=r_hat[0].t(),
                                    coordinates=points,
                                    tensor_stride=8,
                                    device=points.device)

            residuals.append(r_hat.features_at_coordinates(points.float()).clone())

        y_hat_feats = residuals[-1]
        for res in residuals[:-1]:
            y_hat_feats += res

        y_hat = ME.SparseTensor(features=y_hat_feats, 
                                coordinates=points, 
                                tensor_stride=8, 
                                device=points.device)

        return y_hat

class SortedMinkowskiConvolution(ME.MinkowskiConvolution):

    def forward(self, input):
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (input.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        input = ME.SparseTensor(
            features=input.F[sorted_coords_indices],
            coordinates=input.C[sorted_coords_indices],
            tensor_stride=input.tensor_stride,
            device=input.device
        )

        output = super().forward(input)
        
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (output.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        output = ME.SparseTensor(
            features=output.F[sorted_coords_indices],
            coordinates=output.C[sorted_coords_indices],
            tensor_stride=output.tensor_stride,
            device=output.device
        )

        return output

class SortedMinkowskiGenerativeConvolutionTranspose(ME.MinkowskiGenerativeConvolutionTranspose):

    def forward(self, input):
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (input.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        input = ME.SparseTensor(
            features=input.F[sorted_coords_indices],
            coordinates=input.C[sorted_coords_indices],
            tensor_stride=input.tensor_stride,
            device=input.device
        )

        output = super().forward(input)
        
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=output.device) 
        sortable_vals = (output.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        output = ME.SparseTensor(
            features=output.F[sorted_coords_indices],
            coordinates=output.C[sorted_coords_indices],
            tensor_stride=output.tensor_stride,
            device=output.device
        )

        return output

class SortedMinkowskiLeakyReLU(ME.MinkowskiLeakyReLU):
    def forward(self, input):
        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=input.device) 
        sortable_vals = (input.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        input = ME.SparseTensor(
            features=input.F[sorted_coords_indices],
            coordinates=input.C[sorted_coords_indices],
            tensor_stride=input.tensor_stride,
            device=input.device
        )

        output = super().forward(input)

        # Sort the coordinates
        weights = torch.tensor([1e12, 1e8, 1e4, 1], device=output.device) 
        sortable_vals = (output.C * weights).sum(dim=1)
        sorted_coords_indices = sortable_vals.argsort()

        output = ME.SparseTensor(
            features=output.F[sorted_coords_indices],
            coordinates=output.C[sorted_coords_indices],
            tensor_stride=output.tensor_stride,
            device=output.device
        )

        return output


class MeanScaleHyperpriorScales(CompressionModel):
    def __init__(self, config):
        super().__init__()

        self.C_in = config["C_in"]
        self.C_bottleneck = config["C_bottleneck"]
        self.C_hyper_bottleneck = config["C_bottleneck"]
        self.num_entropy_bottlenecks = config["num_bottlenecks"]

        (
            self.entropy_bottlenecks,
            self.gaussian_conditionals,
            self.h_a_list,
            self.h_s_list,
            self.expands_list,
            self.extracts_list,
        ) = self._create_entropy_bottlenecks()

        self._register_modules()

    def _register_modules(self):
        for i, entropy_bottleneck in enumerate(self.entropy_bottlenecks):
            self.add_module(f'entropy_bottleneck_{i}', entropy_bottleneck)

        for i, gaussian_conditional in enumerate(self.gaussian_conditionals):
            self.add_module(f'gaussian_conditional_{i}', gaussian_conditional)

        for i, h_a in enumerate(self.h_a_list):
            self.add_module(f'h_a_{i}', h_a)

        for i, h_s in enumerate(self.h_s_list):
            self.add_module(f'h_s_{i}', h_s)

        for i, expands in enumerate(self.expands_list):
            self.add_module(f'expands_{i}', expands)

        for i, extracts in enumerate(self.extracts_list):
            self.add_module(f'extracts_{i}', extracts)

    def _create_entropy_bottlenecks(self):
        entropy_bottlenecks = []
        gaussian_conditionals = []
        h_a_list = []
        h_s_list = []
        expands_list = []
        extracts_list = []

        for i in range(self.num_entropy_bottlenecks):
            entropy_bottleneck = EntropyBottleneck(self.C_hyper_bottleneck)
            gaussian_conditional = GaussianConditional(None)
            h_a = self._build_h_a(self.C_bottleneck, self.C_bottleneck)
            h_s = self._build_h_s(self.C_hyper_bottleneck)
            expands = self._build_expands(in_channels=self.C_bottleneck, out_channels=self.C_in)
            extracts = self._build_extracts(in_channels=self.C_in, out_channels=self.C_bottleneck)

            entropy_bottlenecks.append(entropy_bottleneck)
            gaussian_conditionals.append(gaussian_conditional)
            h_a_list.append(h_a)
            h_s_list.append(h_s)
            expands_list.append(expands)
            extracts_list.append(extracts)

        return entropy_bottlenecks, gaussian_conditionals, h_a_list, h_s_list, expands_list, extracts_list

    def _build_expands(self, in_channels, out_channels):
        expand_layers = [
            ME.MinkowskiConvolution(in_channels, in_channels, kernel_size=1, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dimension=3)
        ]
        return nn.Sequential(*expand_layers)

    def _build_extracts(self, in_channels, out_channels):
        extract_layers = [
            ME.MinkowskiConvolution(in_channels=in_channels, out_channels=out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=in_channels, out_channels=in_channels, kernel_size=3, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=1, dimension=3)
        ]
        return nn.Sequential(*extract_layers)

    def _build_h_a(self, in_channels, out_channels):
        layers = [
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiLeakyReLU(inplace=False),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiLeakyReLU(inplace=False),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
        ]
        return nn.Sequential(*layers)

    def _build_h_s(self, in_channels):
        layers = [
            SortedMinkowskiConvolution(in_channels, in_channels, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose(in_channels, in_channels, kernel_size=2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),
            SortedMinkowskiConvolution(in_channels, in_channels, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose(in_channels, self.C_bottleneck * 3 // 2, kernel_size=2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),
            SortedMinkowskiConvolution(self.C_bottleneck * 3 // 2, self.C_bottleneck * 2, kernel_size=3, stride=1, dimension=3, bias=False)
        ]
        return nn.Sequential(*layers)

    def forward(self, y):
        z_likelihoods = []
        y_likelihoods = []
        y_hats = []
        y_res_list = []
        y_base = ME.SparseTensor( coordinates=y.C, features=y.F.clone(), device=y.device, tensor_stride=8)
        for i in range(self.num_entropy_bottlenecks):
            # Extract features from y
            y_extract = self.extracts_list[i](y_base)

            # Analysis
            z = self.h_a_list[i](y_extract)

            # Entropy bottleneck
            z_hat, z_likelihood = self.entropy_bottlenecks[i](z.F.t().unsqueeze(0))
            z_hat = ME.SparseTensor(features=z_hat[0].t(), coordinates=z.C, tensor_stride=32, device=z.device)
            z_likelihoods.append(z_likelihood)
            
            # Synthesis of z_hat
            gaussian_params = self.h_s_list[i](z_hat)
            gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())

            # Split the tensor into two tensors: scales_hat and means_hat
            means_hat, scales_hat = gaussian_params_feats.chunk(2, dim=1) 
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            # Gaussian Conditional
            y_res, y_likelihood = self.gaussian_conditionals[i](y_extract.F.t().unsqueeze(0), scales_hat, means=means_hat)
            y_res = ME.SparseTensor(features=y_res[0].t(), coordinates=y.C, tensor_stride=8, device=y.device)
            y_likelihoods.append(y_likelihood)

            # Expand and add previous
            y_res = self.expands_list[i](y_res)

            # Sum over residuals
            y_hat = y_res.features_at_coordinates(y.C.float())
            with torch.no_grad():
                for y_rem_res in y_res_list:
                    y_hat += y_rem_res.features_at_coordinates(y.C.float())

            # Append to list of y_hats
            y_hat = ME.SparseTensor(features=y_hat.clone(), coordinates=y.C, tensor_stride=8, device=y.device)
            y_hats.append(y_hat)
            #y_res_list.append(y_res)
            

            with torch.no_grad():
                # Bypass bottleneck
                z_round, _ = self.entropy_bottlenecks[i](z.F.t().unsqueeze(0), training=False)
                z_round = ME.SparseTensor(features=z_round[0].t(), coordinates=z.C, tensor_stride=32, device=z.device)

                # Synthesis of z_round
                gaussian_params_round = self.h_s_list[i](z_round)
                gaussian_params_round_feats = gaussian_params_round.features_at_coordinates(y.C.float())

                # Split the tensor into two tensors: scales_hat and means_hat
                means_round, scales_round = gaussian_params_round_feats.chunk(2, dim=1) 
                scales_round = scales_round.t().unsqueeze(0)
                means_round = means_round.t().unsqueeze(0)

                # Gaussian Conditional
                y_round, _ = self.gaussian_conditionals[i](y_extract.F.t().unsqueeze(0), scales_round, means=means_round, training=False)
                y_round = ME.SparseTensor(features=y_round[0].t(), coordinates=y.C, tensor_stride=8, device=y.device)

                # Expand
                y_res_round = self.expands_list[i](y_round)
                y_res_list.append(y_res_round)
            
            y_base = ME.SparseTensor(features=y_base.F.clone() - y_res_round.features_at_coordinates(y_base.C.float()).detach(), coordinates=y_base.C, tensor_stride=8, device=y.device)

        return y_hats, (y_likelihoods, z_likelihoods)




    def compress(self, y, latent_path=None):
        shapes = []
        y_strings, z_strings = [], []

        y_points = y.C.clone()

        # Downsample and round coordinates by 4
        """
        processed_points = (y_points) // 32
        unique_map, inverse_map = ME.utils.quantization.unique_coordinate_map(processed_points.int())
        z_points = y_points[unique_map]
        """

        y_points = utils.sort_points(y_points)
        #z_points = sort_points(z_points)
        #y_chunks = y.F.chunk(self.num_entropy_bottlenecks, dim=1)

        y_base = ME.SparseTensor( coordinates=y_points, features=y.features_at_coordinates(y_points.float()).clone(), device=y.device, tensor_stride=8)
        for i in range(self.num_entropy_bottlenecks):
            # Extract
            y_extract = self.extracts_list[i](y_base)

            # Hyper Analysis
            z = self.h_a_list[i](y_extract)
            if i == 0:
                z_points = z.C.clone()
                z_points = utils.sort_points(z_points)

            shape = [z.F.shape[0]]
            shapes.append(shape)

            # Bottleneck
            z_string = self.entropy_bottlenecks[i].compress(z.features_at_coordinates(z_points.float()).t().unsqueeze(0))
            z_strings.append(z_string)
            z_hat_feats = self.entropy_bottlenecks[i].decompress(z_string, shape)

            # Reconstruct z_hat
            z_hat = ME.SparseTensor(features=z_hat_feats[0].t(), coordinates=z_points, tensor_stride=32, device=z.device)

            gaussian_params = self.h_s_list[i](z_hat)
        
            # Find the right scales
            gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())
            means_hat, scales_hat = gaussian_params_feats.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)
        
            # Gaussian Conditional
            indexes = self.gaussian_conditionals[i].build_indexes(scales_hat)
            y_string = self.gaussian_conditionals[i].compress(y_extract.F.t().unsqueeze(0), indexes, means=means_hat)
            y_strings.append(y_string)



            # Decompress and subtract
            indexes = self.gaussian_conditionals[i].build_indexes(scales_hat)
            y_res = self.gaussian_conditionals[i].decompress(y_string, indexes, means=means_hat)
            y_res_tensor = ME.SparseTensor(features=y_res[0].t(), coordinates=y_points, tensor_stride=8, device=y.device)




            y_res_tensor = self.expands_list[i](y_res_tensor)

            y_base = ME.SparseTensor(features=y_base.features_at_coordinates(y_points.float()) - y_res_tensor.features_at_coordinates(y_points.float()), coordinates=y_points, tensor_stride=8, device=y.device)

            if latent_path is not None:
                import open3d as o3d
                import matplotlib.pyplot as plt
                import numpy as np
                coordinates = y_base.C.cpu().numpy()

                scales = scales_hat.cpu().numpy()[0]
                means = means_hat.cpu().numpy()[0]
                quantized_features = y_res.cpu().numpy() 
                unquantized_features = y_extract.F.t().cpu().numpy() 

                residuum_raw = unquantized_features - quantized_features
                quantized_features = (quantized_features - means ) / scales
                unquantized_features_shift = (unquantized_features - means) / scales
                residuum_shifted = unquantized_features_shift - quantized_features

                # Find index with biggest variance
                variances = np.var(unquantized_features, axis=1)
                index = np.argsort(variances)[-1]
                lowest = np.argsort(variances)[0]

                fig, ax = plt.subplots(3,2)

                counts, bins = np.histogram(unquantized_features[index], 100)
                print(unquantized_features.shape)
                ax[0, 0].hist(bins[:-1], bins, weights=counts)
                ax[0, 0].set_title("High variance - Features")
                ax[0, 0].set_xlim([-10, 10])
                counts, bins = np.histogram(residuum_raw[0, index], 100)
                ax[1, 0].hist(bins[:-1], bins, weights=counts)
                ax[1, 0].set_title("High variance - Residuals")
                counts, bins = np.histogram(residuum_shifted[0, index], 100)
                ax[2, 0].hist(bins[:-1], bins, weights=counts)
                ax[2, 0].set_title("High variance - Residuals (Shifted)")

                counts, bins = np.histogram(unquantized_features[lowest], 100)
                ax[0, 1].hist(bins[:-1], bins, weights=counts)
                ax[0, 1].set_title("Low variance - Features")
                ax[0, 1].set_xlim([-10, 10])
                counts, bins = np.histogram(residuum_raw[0, lowest], 100)
                ax[1, 1].hist(bins[:-1], bins, weights=counts)
                ax[1, 1].set_title("Low variance - Residuals")
                counts, bins = np.histogram(residuum_shifted[0, lowest], 100)
                ax[2, 1].hist(bins[:-1], bins, weights=counts)
                ax[2, 1].set_title("Low variance - Residuals (Shifted)")

                plt.tight_layout()
                plt.savefig("temp/bins_{}.pdf".format(i))

                # Residuum at bottleneck
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                color_value = residuum_raw[0, index:index+1].T
                color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                utils.render_pointcloud(point_cloud, "temp/residual_high_n{}_{}.png".format(i, "{}"), point_size=8)

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                color_value = residuum_raw[0, lowest:lowest+1].T
                color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                utils.render_pointcloud(point_cloud, "temp/residual_low_n{}_{}.png".format(i, "{}"), point_size=8)


                # Residuum after non-linearity
                feats = y_base.F.cpu().numpy()
                variances = np.var(feats, axis=0)
                index = np.argsort(variances)[-1]
                lowest = np.argsort(variances)[1]

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                color_value = feats[:, index:index+1]
                color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                utils.render_pointcloud(point_cloud, "temp/nonlin_residual_high_n{}_{}.png".format(i, "{}"), point_size=8)

                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                color_value = feats[:, lowest:lowest+1]
                color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                utils.render_pointcloud(point_cloud, "temp/nonlin_residual_low_n{}_{}.png".format(i, "{}"), point_size=8)
        # Points are needed, to be compressed later

        # Pack it
        points = [y_points, z_points]
        strings = [y_strings, z_strings]
        return strings, shapes


    def decompress(self, points, strings, shapes):
        assert isinstance(strings, list) and len(strings) == 2

        # Get the points back
        y_strings, z_strings = strings[0], strings[1]
        y_points, z_points = points[0], points[1]

        y_points = utils.sort_points(y_points)
        z_points = utils.sort_points(z_points)

        residuals = []
        for i in range(len(y_strings)):

            z_hat_feats = self.entropy_bottlenecks[i].decompress(z_strings[i], shapes[i])
            z_hat = ME.SparseTensor(features=z_hat_feats[0].t(),
                                    coordinates=z_points,
                                    tensor_stride=32,
                                    device=z_points.device)
            # Decompress y_hat
            gaussian_params = self.h_s_list[i](z_hat)
            gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())

            means_hat, scales_hat = gaussian_params_feats.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            indexes = self.gaussian_conditionals[i].build_indexes(scales_hat)
            y_hat_feats = self.gaussian_conditionals[i].decompress(y_strings[i], indexes, means=means_hat)

            y_hat = ME.SparseTensor(features=y_hat_feats[0].t(),
                                    coordinates=y_points,
                                    tensor_stride=8,
                                    device=y_points.device)

            y_hat = self.expands_list[i](y_hat)
            residuals.append(y_hat.F.clone())

        y_hat_feats = residuals[-1]
        for res in residuals[:-1]:
            y_hat_feats += res

        y_hat = ME.SparseTensor(features=y_hat_feats, coordinates=y_points, tensor_stride=8, device=y_points.device)
        return y_hat


class MeanScaleHyperpriorScalesDirect(CompressionModel):
    def __init__(self, config):
        super().__init__()

        self.C_in = config["C_in"]
        self.C_bottleneck = config["C_bottleneck"]
        self.C_hyper_bottleneck = config["C_bottleneck"]
        self.num_entropy_bottlenecks = config["num_bottlenecks"]

        (
            self.entropy_bottlenecks,
            self.gaussian_conditionals,
            self.h_a_list,
            self.h_s_list,
        ) = self._create_entropy_bottlenecks()

        self._register_modules()

    def _register_modules(self):
        for i, entropy_bottleneck in enumerate(self.entropy_bottlenecks):
            self.add_module(f'entropy_bottleneck_{i}', entropy_bottleneck)

        for i, gaussian_conditional in enumerate(self.gaussian_conditionals):
            self.add_module(f'gaussian_conditional_{i}', gaussian_conditional)

        for i, h_a in enumerate(self.h_a_list):
            self.add_module(f'h_a_{i}', h_a)

        for i, h_s in enumerate(self.h_s_list):
            self.add_module(f'h_s_{i}', h_s)

    def _create_entropy_bottlenecks(self):
        entropy_bottlenecks = []
        gaussian_conditionals = []
        h_a_list = []
        h_s_list = []

        for i in range(self.num_entropy_bottlenecks):
            entropy_bottleneck = EntropyBottleneck(self.C_hyper_bottleneck)
            gaussian_conditional = GaussianConditional(None)
            h_a = self._build_h_a(self.C_bottleneck, self.C_bottleneck)
            h_s = self._build_h_s(self.C_hyper_bottleneck)

            entropy_bottlenecks.append(entropy_bottleneck)
            gaussian_conditionals.append(gaussian_conditional)
            h_a_list.append(h_a)
            h_s_list.append(h_s)

        return entropy_bottlenecks, gaussian_conditionals, h_a_list, h_s_list


    def _build_h_a(self, in_channels, out_channels):
        layers = [
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiLeakyReLU(inplace=False),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
            ME.MinkowskiLeakyReLU(inplace=False),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=2, stride=2, dimension=3),
            ME.MinkowskiConvolution(out_channels, out_channels, kernel_size=3, dimension=3),
        ]
        return nn.Sequential(*layers)

    def _build_h_s(self, in_channels):
        layers = [
            SortedMinkowskiConvolution(in_channels, in_channels, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose(in_channels, in_channels, kernel_size=2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),
            SortedMinkowskiConvolution(in_channels, in_channels, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose(in_channels, self.C_bottleneck * 3 // 2, kernel_size=2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),
            SortedMinkowskiConvolution(self.C_bottleneck * 3 // 2, self.C_bottleneck * 2, kernel_size=3, stride=1, dimension=3, bias=False)
        ]
        return nn.Sequential(*layers)

    def forward(self, y):
        z_likelihoods = []
        y_likelihoods = []
        y_hats = []
        y_res_list = []
        y_base = ME.SparseTensor( coordinates=y.C, features=y.F.clone(), device=y.device, tensor_stride=8)
        for i in range(self.num_entropy_bottlenecks):
            # Extract features from y
            y_extract = y_base

            # Analysis
            z = self.h_a_list[i](y_extract)

            # Entropy bottleneck
            z_hat, z_likelihood = self.entropy_bottlenecks[i](z.F.t().unsqueeze(0))
            z_hat = ME.SparseTensor(features=z_hat[0].t(), coordinates=z.C, tensor_stride=32, device=z.device)
            z_likelihoods.append(z_likelihood)
            
            # Synthesis of z_hat
            gaussian_params = self.h_s_list[i](z_hat)
            gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())

            # Split the tensor into two tensors: scales_hat and means_hat
            means_hat, scales_hat = gaussian_params_feats.chunk(2, dim=1) 
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            # Gaussian Conditional
            y_res, y_likelihood = self.gaussian_conditionals[i](y_extract.F.t().unsqueeze(0), scales_hat, means=means_hat)
            y_res = ME.SparseTensor(features=y_res[0].t(), coordinates=y.C, tensor_stride=8, device=y.device)
            y_likelihoods.append(y_likelihood)

            # Sum over residuals
            y_hat = y_res.features_at_coordinates(y.C.float())
            with torch.no_grad():
                for y_rem_res in y_res_list:
                    y_hat += y_rem_res.features_at_coordinates(y.C.float())

            # Append to list of y_hats
            y_hat = ME.SparseTensor(features=y_hat.clone(), coordinates=y.C, tensor_stride=8, device=y.device)
            y_hats.append(y_hat)
            #y_res_list.append(y_res)
            

            with torch.no_grad():
                # Bypass bottleneck
                z_round, _ = self.entropy_bottlenecks[i](z.F.t().unsqueeze(0), training=False)
                z_round = ME.SparseTensor(features=z_round[0].t(), coordinates=z.C, tensor_stride=32, device=z.device)

                # Synthesis of z_round
                gaussian_params_round = self.h_s_list[i](z_round)
                gaussian_params_round_feats = gaussian_params_round.features_at_coordinates(y.C.float())

                # Split the tensor into two tensors: scales_hat and means_hat
                means_round, scales_round = gaussian_params_round_feats.chunk(2, dim=1) 
                scales_round = scales_round.t().unsqueeze(0)
                means_round = means_round.t().unsqueeze(0)

                # Gaussian Conditional
                y_round, _ = self.gaussian_conditionals[i](y_extract.F.t().unsqueeze(0), scales_round, means=means_round, training=False)
                y_round = ME.SparseTensor(features=y_round[0].t(), coordinates=y.C, tensor_stride=8, device=y.device)

                # Expand
                y_res_round = y_round
                y_res_list.append(y_res_round)
            
            y_base = ME.SparseTensor(features=y_base.F.clone() - y_res_round.features_at_coordinates(y_base.C.float()).detach(), coordinates=y_base.C, tensor_stride=8, device=y.device)

        return y_hats, (y_likelihoods, z_likelihoods)




    def compress(self, y, latent_path=None):
        shapes = []
        y_strings, z_strings = [], []

        y_points = y.C.clone()

        # Downsample and round coordinates by 4
        """
        processed_points = (y_points) // 32
        unique_map, inverse_map = ME.utils.quantization.unique_coordinate_map(processed_points.int())
        z_points = y_points[unique_map]
        """

        y_points = utils.sort_points(y_points)
        #z_points = sort_points(z_points)
        #y_chunks = y.F.chunk(self.num_entropy_bottlenecks, dim=1)

        y_base = ME.SparseTensor( coordinates=y_points, features=y.features_at_coordinates(y_points.float()).clone(), device=y.device, tensor_stride=8)
        for i in range(self.num_entropy_bottlenecks):
            # Extract
            y_extract = y_base

            # Hyper Analysis
            z = self.h_a_list[i](y_extract)
            if i == 0:
                z_points = z.C.clone()
                z_points = utils.sort_points(z_points)

            shape = [z.F.shape[0]]
            shapes.append(shape)

            # Bottleneck
            z_string = self.entropy_bottlenecks[i].compress(z.features_at_coordinates(z_points.float()).t().unsqueeze(0))
            z_strings.append(z_string)
            z_hat_feats = self.entropy_bottlenecks[i].decompress(z_string, shape)

            # Reconstruct z_hat
            z_hat = ME.SparseTensor(features=z_hat_feats[0].t(), coordinates=z_points, tensor_stride=32, device=z.device)

            gaussian_params = self.h_s_list[i](z_hat)
        
            # Find the right scales
            gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())
            means_hat, scales_hat = gaussian_params_feats.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)
        
            # Gaussian Conditional
            indexes = self.gaussian_conditionals[i].build_indexes(scales_hat)
            y_string = self.gaussian_conditionals[i].compress(y_extract.F.t().unsqueeze(0), indexes, means=means_hat)
            y_strings.append(y_string)



            # Decompress and subtract
            indexes = self.gaussian_conditionals[i].build_indexes(scales_hat)
            y_res = self.gaussian_conditionals[i].decompress(y_string, indexes, means=means_hat)
            y_res_tensor = ME.SparseTensor(features=y_res[0].t(), coordinates=y_points, tensor_stride=8, device=y.device)

            # Prepare latents for storage
            if latent_path is not None:
                import numpy as np
                quantized_features = y_res.cpu().numpy()
                #quantized_features = (quantized_features - means_hat.cpu().numpy()) / scales_hat.cpu().numpy()
                np.save(latent_path + "/quantized_features.npy", quantized_features)

                unquantized_features = y_extract.F.t().cpu().numpy()
                #unquantized_features = (unquantized_features - means_hat.cpu().numpy()) / scales_hat.cpu().numpy()
                np.save(latent_path + "/unquantized_features.npy", unquantized_features)

                residuum = unquantized_features - quantized_features
                np.save(latent_path + "/residual_features.npy", residuum)



            y_base = ME.SparseTensor(features=y_base.features_at_coordinates(y_points.float()) - y_res_tensor.features_at_coordinates(y_points.float()), coordinates=y_points, tensor_stride=8, device=y.device)

            if latent_path is not None:
                import open3d as o3d
                import numpy as np
                coordinates = y_base.C.cpu().numpy()
                residuum = y_base.F.cpu().numpy()
                for j in range(12):
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                    color_value = residuum[:, i:i+1]
                    color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                    utils.render_pointcloud(point_cloud, "temp/residual_f{}_n{}_{}.png".format(j, i, "{}"), point_size=8)

                features = y_extract.F.cpu().numpy()
                for j in range(12):
                    point_cloud = o3d.geometry.PointCloud()
                    point_cloud.points = o3d.utility.Vector3dVector(coordinates[:, 1:] / 8)
                    color_value = features[:, i:i+1]
                    color_value = (color_value - np.min(color_value)) / (np.max(color_value) - np.min(color_value))
                    point_cloud.colors = o3d.utility.Vector3dVector(np.concatenate([color_value, color_value, color_value], axis=1))
                    utils.render_pointcloud(point_cloud, "temp/features_f{}_n{}_{}.png".format(j, i, "{}"), point_size=8)


        # Points are needed, to be compressed later

        # Pack it
        points = [y_points, z_points]
        strings = [y_strings, z_strings]
        return strings, shapes


    def decompress(self, points, strings, shapes):
        assert isinstance(strings, list) and len(strings) == 2

        # Get the points back
        y_strings, z_strings = strings[0], strings[1]
        y_points, z_points = points[0], points[1]

        y_points = utils.sort_points(y_points)
        z_points = utils.sort_points(z_points)

        residuals = []
        for i in range(len(y_strings)):

            z_hat_feats = self.entropy_bottlenecks[i].decompress(z_strings[i], shapes[i])
            z_hat = ME.SparseTensor(features=z_hat_feats[0].t(),
                                    coordinates=z_points,
                                    tensor_stride=32,
                                    device=z_points.device)
            # Decompress y_hat
            gaussian_params = self.h_s_list[i](z_hat)
            gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())

            means_hat, scales_hat = gaussian_params_feats.chunk(2, dim=1)
            scales_hat = scales_hat.t().unsqueeze(0)
            means_hat = means_hat.t().unsqueeze(0)

            indexes = self.gaussian_conditionals[i].build_indexes(scales_hat)
            y_hat_feats = self.gaussian_conditionals[i].decompress(y_strings[i], indexes, means=means_hat)

            y_hat = ME.SparseTensor(features=y_hat_feats[0].t(),
                                    coordinates=y_points,
                                    tensor_stride=8,
                                    device=y_points.device)

            residuals.append(y_hat.F.clone())

        y_hat_feats = residuals[-1]
        for res in residuals[:-1]:
            y_hat_feats += res

        y_hat = ME.SparseTensor(features=y_hat_feats, coordinates=y_points, tensor_stride=8, device=y_points.device)
        return y_hat