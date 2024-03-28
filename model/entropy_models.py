import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import MinkowskiEngine as ME

from utils import sort_points, sort_tensor

from compressai.entropy_models import EntropyBottleneck, GaussianConditional 
from compressai.models.base import CompressionModel

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

class MeanScaleHyperprior(CompressionModel):
    def __init__(self, config):
        """
        Paramters
        ---------
            C_bottleneck: int
                Number of channels in the bottlneck
            C_hyper_bottlneck: int
                Number of channels in the bottlneck of the hyperprior model
            N: int
                Number of channels in between
        """
        super().__init__()
        C_bottleneck = config["C_bottleneck"]
        C_hyper_bottleneck = config["C_hyper_bottleneck"]
        self.entropy_bottleneck = EntropyBottleneck(C_hyper_bottleneck)
        self.gaussian_conditional = GaussianConditional(None)

        self.h_a = nn.Sequential(
            ME.MinkowskiConvolution( in_channels=C_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
            ME.MinkowskiLeakyReLU(inplace=False),

            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
            ME.MinkowskiLeakyReLU(inplace=False),

            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
        )

        self.h_s = nn.Sequential(
            SortedMinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size = 2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),

            SortedMinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_bottleneck*3//2, kernel_size = 2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),

            SortedMinkowskiConvolution( in_channels=C_bottleneck*3//2, out_channels=C_bottleneck*2, kernel_size=3, stride=1, dimension=3, bias=True)
        )


    def forward(self, y):
        z = self.h_a(y)

        # Entropy model
        z_hat, z_likelihoods = self.entropy_bottleneck(z.F.t().unsqueeze(0))

        # Reconstruct z_hat
        z_hat = ME.SparseTensor(features=z_hat[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        # Hyper synthesis
        gaussian_params = self.h_s(z_hat)

        # Find the right scales
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)
        y_hat, y_likelihoods = self.gaussian_conditional(y.F.t().unsqueeze(0), scales_hat, means=means_hat)

        y_hat = ME.SparseTensor(features=y_hat[0].t(), 
                                coordinates=y.C,
                                tensor_stride=8,
                                device=y.device)

        return y_hat, (y_likelihoods, z_likelihoods)



    def compress(self, y):
        # Hyper analysis
        z = self.h_a(y)

        # Sort points
        y = sort_tensor(y)
        z = sort_tensor(z)

        # Entropy model
        shape = [z.F.shape[0]]

        z_strings = self.entropy_bottleneck.compress(z.F.t().unsqueeze(0))
        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)

        # Reconstruct z_hat
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        gaussian_params = self.h_s(z_hat)
        
        # Find the right scales
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())

        # Compress all that
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y.F.t().unsqueeze(0), indexes, means=means_hat)

        # Points are needed, to be compressed later
        y_points = y.C
        z_points = z.C

        # Pack it
        points = [y_points, z_points]
        strings = [y_strings, z_strings]
        return points, strings, shape


    def decompress(self, points, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # Get the points back
        y_points, z_points = points[0], points[1]
        y_points = sort_points(y_points)
        z_points = sort_points(z_points)
        y_strings, z_strings = strings[0], strings[1]

        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(),
                                coordinates=z_points,
                                tensor_stride=32,
                                device=z_points.device)
        # Decompress y_hat
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())

        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat_feats = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)

        y_hat = ME.SparseTensor(features=y_hat_feats[0].t(),
                                coordinates=y_points,
                                tensor_stride=8,
                                device=y_points.device)
        return y_hat


class MeanScaleHyperprior_Map(CompressionModel):
    def __init__(self, config):
        """
        Paramters
        ---------
            C_bottleneck: int
                Number of channels in the bottlneck
            C_hyper_bottlneck: int
                Number of channels in the bottlneck of the hyperprior model
            N: int
                Number of channels in between
        """
        super().__init__()
        C_bottleneck = config["C_bottleneck"]
        C_hyper_bottleneck = config["C_hyper_bottleneck"]
        C_Q = config["C_Q"]
        self.entropy_bottleneck = EntropyBottleneck(C_hyper_bottleneck)
        self.gaussian_conditional = GaussianConditional(None)

        self.h_a = nn.Sequential(
            ME.MinkowskiConvolution( in_channels=C_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
            ME.MinkowskiLeakyReLU(inplace=False),

            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
            ME.MinkowskiLeakyReLU(inplace=False),

            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=2, dimension=3),
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, dimension=3), 
        )

        self.h_s = nn.Sequential(
            SortedMinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size = 2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),

            SortedMinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=1, dimension=3, bias=True),
            SortedMinkowskiGenerativeConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_bottleneck*3//2, kernel_size = 2, stride=2, bias=True, dimension=3),
            SortedMinkowskiLeakyReLU(inplace=False),

            SortedMinkowskiConvolution( in_channels=C_bottleneck*3//2, out_channels=C_bottleneck*2, kernel_size=3, stride=1, dimension=3, bias=True)
        )

        self.h_q = nn.Sequential(
            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=1, dimension=3, bias=True),
            ME.MinkowskiConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size = 3, stride=2, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),

            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size=3, stride=1, dimension=3, bias=True),
            ME.MinkowskiConvolutionTranspose( in_channels=C_hyper_bottleneck, out_channels=C_hyper_bottleneck, kernel_size = 3, stride=2, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),

            ME.MinkowskiConvolution( in_channels=C_hyper_bottleneck, out_channels=C_Q, kernel_size = 3, stride=1, bias=True, dimension=3),
        )


    def forward(self, y):
        z = self.h_a(y)

        # Entropy model
        z_hat, z_likelihoods = self.entropy_bottleneck(z.F.t().unsqueeze(0))

        # Reconstruct z_hat
        z_hat = ME.SparseTensor(features=z_hat[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        # Hyper synthesis
        gaussian_params = self.h_s(z_hat)
        Q_hat = self.h_q(z_hat)

        # Find the right scales
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)
        y_hat, y_likelihoods = self.gaussian_conditional(y.F.t().unsqueeze(0), scales_hat, means=means_hat)

        y_hat = ME.SparseTensor(features=y_hat[0].t(), 
                                coordinates=y.C,
                                tensor_stride=8,
                                device=y.device)

        return y_hat, Q_hat, (y_likelihoods, z_likelihoods)



    def compress(self, y):
        # Hyper analysis
        z = self.h_a(y)

        # Sort points
        y = sort_tensor(y)
        z = sort_tensor(z)

        # Entropy model
        shape = [z.F.shape[0]]

        z_strings = self.entropy_bottleneck.compress(z.F.t().unsqueeze(0))
        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)

        # Reconstruct z_hat
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(), 
                                coordinates=z.C,
                                tensor_stride=32,
                                device=z.device)

        gaussian_params = self.h_s(z_hat)
        
        # Find the right scales
        gaussian_params_feats = gaussian_params.features_at_coordinates(y.C.float())

        # Compress all that
        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y.F.t().unsqueeze(0), indexes, means=means_hat)

        # Points are needed, to be compressed later
        y_points = y.C
        z_points = z.C

        # Pack it
        points = [y_points, z_points]
        strings = [y_strings, z_strings]
        return points, strings, shape


    def decompress(self, points, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2

        # Get the points back
        y_points, z_points = points[0], points[1]
        y_points = sort_points(y_points)
        z_points = sort_points(z_points)
        y_strings, z_strings = strings[0], strings[1]

        z_hat_feats = self.entropy_bottleneck.decompress(z_strings, shape)
        z_hat = ME.SparseTensor(features=z_hat_feats[0].t(),
                                coordinates=z_points,
                                tensor_stride=32,
                                device=z_points.device)
        # Decompress y_hat
        Q_hat = self.h_q(z_hat)
        gaussian_params = self.h_s(z_hat)
        gaussian_params_feats = gaussian_params.features_at_coordinates(y_points.float())

        scales_hat, means_hat = gaussian_params_feats.chunk(2, dim=1)
        scales_hat = scales_hat.t().unsqueeze(0)
        means_hat = means_hat.t().unsqueeze(0)

        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat_feats = self.gaussian_conditional.decompress(y_strings, indexes, means=means_hat)

        y_hat = ME.SparseTensor(features=y_hat_feats[0].t(),
                                coordinates=y_points,
                                tensor_stride=8,
                                device=y_points.device)
        return y_hat, Q_hat

