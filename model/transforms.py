import torch
import torch.nn as nn
import MinkowskiEngine as ME

import utils
from model.blocks import *


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

        # Model
        self.pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=C_in, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.down_1 = ME.MinkowskiConvolution(in_channels=N1, out_channels=N2, kernel_size=3, stride=2, bias=True, dimension=3)
        self.down_2 = ME.MinkowskiConvolution(in_channels=N2, out_channels=N3, kernel_size=3, stride=2, bias=True, dimension=3)
        self.down_3 = ME.MinkowskiConvolution(in_channels=N3, out_channels=N3, kernel_size=3, stride=2, bias=True, dimension=3)

        self.scale_1 = ScaledBlock(N2, encode=True)
        self.scale_2 = ScaledBlock(N3, encode=True)
        self.scale_3 = ScaledBlock(N3, encode=True)

        self.post_conv = ME.MinkowskiConvolution(in_channels=N3, out_channels=N3, kernel_size=3, stride=1, bias=True, dimension=3)

        # Conditions
        self.q_pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2, out_channels=N1//4, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.q_down_1 = ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N2//4, kernel_size=3, stride=2, bias=True, dimension=3)
        self.q_down_2 = ME.MinkowskiConvolution(in_channels=N2//4, out_channels=N3//4, kernel_size=3, stride=2, bias=True, dimension=3)
        self.q_down_3 = ME.MinkowskiConvolution(in_channels=N3//4, out_channels=N3//4, kernel_size=3, stride=2, bias=True, dimension=3)

        self.q_layers_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2//4, out_channels=N2//4, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.q_layers_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3//4, out_channels=N3//4, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.q_layers_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3//4, out_channels=N3//4, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        
        self.q_predict_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2//4, out_channels=N2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2, out_channels=N2*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.q_predict_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3//4, out_channels=N3, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N3, out_channels=N3*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.q_predict_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3//4, out_channels=N3, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N3, out_channels=N3*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )


    def forward(self, x, Q):
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
        # Pre-Conv
        x = self.pre_conv(x)
        Q = self.q_pre_conv(Q)

        # Layer 1
        x = self.down_1(x)
        Q = self.q_down_1(Q)

        Q = self.q_layers_1(Q)
        beta_gamma = self.q_predict_1(Q)
        x = self.scale_1(x, beta_gamma)

        # Layer 2
        x = self.down_2(x)
        Q = self.q_down_2(Q)

        Q = self.q_layers_2(Q)
        beta_gamma = self.q_predict_2(Q)
        x = self.scale_2(x, beta_gamma)

        # Layer 3
        x = self.down_3(x)
        Q = self.q_down_3(Q)

        Q = self.q_layers_3(Q)
        beta_gamma = self.q_predict_3(Q)
        x = self.scale_3(x, beta_gamma)

        x = self.post_conv(x)

        # Concat quality and features for compression
        x = ME.SparseTensor(coordinates=x.C,
                            features=torch.cat([x.F, Q.features_at_coordinates(x.C.float())], dim=1),
                            tensor_stride=x.tensor_stride)

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

        # Model
        self.pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.up_1 = GenerativeUpBlock(N1, N1)
        self.up_2 = GenerativeUpBlock(N1, N2)
        self.up_3 = GenerativeUpBlock(N2, N3)

        self.scale_1 = ScaledBlock(N1, encode=False)
        self.scale_2 = ScaledBlock(N1, encode=False)
        self.scale_3 = ScaledBlock(N2, encode=False)

        self.post_conv = ME.MinkowskiConvolution(in_channels=N3, out_channels=C_out, kernel_size=3, stride=1, bias=True, dimension=3)

        # Condition
        self.q_pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N1//4, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.q_up_1 = GenerativeUpBlock(N1//4, N1//4)
        self.q_up_2 = GenerativeUpBlock(N1//4, N2//4)
        self.q_up_3 = GenerativeUpBlock(N2//4, N2//4)

        self.q_layers_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N1//4, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.q_layers_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N1//4, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.q_layers_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2//4, out_channels=N2//4, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        
        self.q_predict_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.q_predict_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N1//4, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.q_predict_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N2//4, out_channels=N2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2, out_channels=N2*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )


        # Auxiliary
        self.down_conv = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3)



    def forward(self, y, coords=None):
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
            points_1 = self.down_conv(coords)
            points_2 = self.down_conv(points_1)

        # Split coords after entropy coding
        Q = ME.SparseTensor(coordinates=y.C, features=y.F[:, 128:], device=y.device, tensor_stride=8)
        x = ME.SparseTensor(coordinates=y.C, features=y.F[:, :128], device=y.device, tensor_stride=8)

        # Pre-Conv
        x = self.pre_conv(x)
        Q = self.q_pre_conv(Q)

        # Layer 1
        Q = self.q_layers_1(Q)
        beta_gamma = self.q_predict_1(Q)
        x = self.scale_1(x, beta_gamma)

        Q = self.q_up_1(Q, points_2)
        x = self.up_1(x, points_2)

        # Layer 2
        Q = self.q_layers_2(Q)
        beta_gamma = self.q_predict_2(Q)
        x = self.scale_2(x, beta_gamma)

        Q = self.q_up_2(Q, points_1)
        x = self.up_2(x, points_1)

        # Layer 3
        Q = self.q_layers_3(Q)
        beta_gamma = self.q_predict_3(Q)
        x = self.scale_3(x, beta_gamma)

        Q = self.q_up_3(Q, coords)
        x = self.up_3(x, coords)

        # Post Conv
        x = self.post_conv(x)

        return x

