import torch
import torch.nn as nn
import MinkowskiEngine as ME

import utils


class AnalysisTransform(nn.Module):
    """
    Simple Analysis Module consisting of 3 blocks on Non-linear transformations
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information for the transformation
            keys:
                C_in: Number of input channels
                N1: Filters in the 1st layer
                N2: Filters in the 2nd layer
                N3: Filters in the 3rd layer
        """
        super().__init__()

        C_in = config["C_in"]
        N1 = config["N1"]
        N2 = config["N2"]
        N3 = config["N3"]

        self.block_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=C_in, 
                                    out_channels=N1, 
                                    kernel_size=3, 
                                    stride=1, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiConvolution(in_channels=N1, 
                                    out_channels=N2, 
                                    kernel_size=3, 
                                    stride=2, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.block_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2, 
                                    out_channels=N2, 
                                    kernel_size=3, 
                                    stride=1, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiConvolution(in_channels=N2, 
                                    out_channels=N3, 
                                    kernel_size=3, 
                                    stride=2, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.block_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3, 
                                    out_channels=N3, 
                                    kernel_size=3, 
                                    stride=1, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiConvolution(in_channels=N3, 
                                    out_channels=N3, 
                                    kernel_size=3, 
                                    stride=2, 
                                    bias=True, 
                                    dimension=3),
        )

    def forward(self, x):
        """
        Forward pass for the analysis transform

        Parameters
        ----------
        x: ME.SparseTensor
            Sparse Tensor containing the orignal features

        returns
        -------
        x: ME.SparseTensor
            Sparse Tensor containing the latent features
        """
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        return x



class SparseSynthesisTransform(torch.nn.Module):
    """
    Sparse Decoder/ Synthesis Transform module for Attribute Compression
    Operates by pruning voxels after each upsampling step using the original point cloud geometry.
    """
    def __init__(self, config):
        """
        Parameters
        ----------
        config : dict
            Dictionary containing information for the transformation
            keys:
                C_in: Number of input channels
                N1: Filters in the 3rd layer
                N2: Filters in the 2nd layer
                N3: Filters in the 1st layer
        """
        super().__init__()

        C_out = config["C_out"]
        N1 = config["N1"]
        N2 = config["N2"]
        N3 = config["N3"]

        self.up_1 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N3, 
                                                               out_channels=N3, 
                                                               kernel_size=3, 
                                                               stride=2, 
                                                               bias=True, 
                                                               dimension=3)
        self.block_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3, 
                                    out_channels=N3, 
                                    kernel_size=3, 
                                    stride=1, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )


        self.up_2 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N3, 
                                                               out_channels=N2, 
                                                               kernel_size=3, 
                                                               stride=2, 
                                                               bias=True, 
                                                               dimension=3)
        self.block_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2, 
                                    out_channels=N2, 
                                    kernel_size=3, 
                                    stride=1, 
                                    bias=True, 
                                    dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.up_3 = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N2, 
                                                               out_channels=N1, 
                                                               kernel_size=3, 
                                                               stride=2, 
                                                               bias=True, 
                                                               dimension=3)
        self.block_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1, 
                                    out_channels=C_out, 
                                    kernel_size=3, 
                                    stride=1, 
                                    dimension=3)
        )


        self.down_conv = ME.MinkowskiConvolution(in_channels=1, 
                                                 out_channels=1, 
                                                 kernel_size=3, 
                                                 stride=2, 
                                                 dimension=3)
        self.eq_conv = ME.MinkowskiConvolution(in_channels=1, 
                                                 out_channels=1, 
                                                 kernel_size=3, 
                                                 stride=1, 
                                                 dimension=3)

        self.prune = ME.MinkowskiPruning()


    def _prune_coords(self, x, occupied_points):
        """
        Prunes the coordinates after upsampling, only keeping points coinciding with occupied points

        Parameters
        ----------
        x: ME.SparseTensor
            Upsampled point cloud with features
        occupied_points: ME.SparseTensor
            Sparse Tensor containing the coordinates to keep

        returns
        -------
        x: ME.SparseTensor
            Pruned tensor with features
        """
        # Define Scaling Factors
        scaling_factors = torch.tensor([1, 1e5, 1e10, 1e15], dtype=torch.int64, device=x.C.device)

        # Transform to unique indices
        x_flat = (x.C.to(torch.int64) * scaling_factors).sum(dim=1)
        guide_flat = (occupied_points.to(torch.int64) * scaling_factors).sum(dim=1)

        # Prune
        mask = torch.isin(x_flat, guide_flat)
        x = self.prune(x, mask)

        return x
    


    def forward(self, x, coords=None):
        """
        Forward pass for the synthesis transform

        Parameters
        ----------
        x: ME.SparseTensor
            Sparse Tensor containing the latent features
        coords: ME.SparseTensor
            Sparse Tensor containing coordinates of the upsampled point cloud

        returns
        -------
        x: ME.SparseTensor
            Sparse Tensor containing the upsampled features at location of coords
        """
        # Compute downsampled coordinates for pruning
        with torch.no_grad():
            #coords = coords.C.to(torch.int32)
            points_1 = self.down_conv(coords)
            points_2 = self.down_conv(points_1)
            #points_0 = utils.downsampled_coordinates(coords.clone(), factor=1)
            #points_1 = utils.downsampled_coordinates(coords.clone(), factor=2)
            #points_2 = utils.downsampled_coordinates(coords.clone(), factor=4)

        x = self.up_1(x)
        x = self._prune_coords(x, points_2.C)
        x = self.block_1(x)

        x = self.up_2(x)
        x = self._prune_coords(x, points_1.C)
        x = self.block_2(x)


        x = self.up_3(x)
        x = self._prune_coords(x, coords.C)
        x = self.block_3(x)


        return x