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

        self.scale_1 = ScaledBlock(N2, encode=True, scale=True)
        self.scale_2 = ScaledBlock(N3, encode=True, scale=True)
        self.scale_3 = ScaledBlock(N3, encode=True, scale=True)

        self.post_conv = ME.MinkowskiConvolution(in_channels=N3, out_channels=N3, kernel_size=3, stride=1, bias=True, dimension=3)

        # Conditions
        self.condition_encoder = ConditionEncoder(C_in = 2, 
                                                  N_scales=[N2, N2, N3],
                                                  N_features=[2, 2, 2, 2])

    def count_per_batch(self, x):
        batch_indices = torch.unique(x.C[:, 0])  # Get unique batch IDs
        k_per_batch = []
        for batch_idx in batch_indices:
            k = (x.C[:, 0] == batch_idx).sum().item()
            k_per_batch.append(k)
        return k_per_batch
        
        

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
        k = []
        k.append(self.count_per_batch(x))
        Q, beta_gammas = self.condition_encoder(Q)

        # Pre-Conv
        x = self.pre_conv(x)

        # Layer 1
        x = self.down_1(x)
        x = self.scale_1(x, beta_gammas[0])
        k.append(self.count_per_batch(x))

        # Layer 2
        x = self.down_2(x)
        x = self.scale_2(x, beta_gammas[1])
        k.append(self.count_per_batch(x))

        # Layer 3
        x = self.down_3(x)
        x = self.scale_3(x, beta_gammas[2])

        x = self.post_conv(x)

        # Concat quality and features for compression
        Q = ME.SparseTensor(coordinates=x.C,
                            features=Q.features_at_coordinates(x.C.float()),
                            tensor_stride=x.tensor_stride)

        k.reverse()
        return x, Q, k


        


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

        self.up_1 = GenerativeUpBlock(N1, N1, predict=True)
        self.up_2 = GenerativeUpBlock(N1, N2, predict=True)
        self.up_3 = GenerativeUpBlock(N2, N3, predict=True)

        self.scale_1 = ScaledBlock(N1, encode=False, scale=True)
        self.scale_2 = ScaledBlock(N1, encode=False, scale=True)
        self.scale_3 = ScaledBlock(N2, encode=False, scale=True)

        self.post_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N3, out_channels=N3, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N3, out_channels=N3//2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N3//2, out_channels=C_out, kernel_size=3, stride=1, bias=True, dimension=3),
        )

        # Condition
        self.q_pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2, out_channels=16, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=16, out_channels=16, kernel_size=1, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=16, out_channels=2, kernel_size=3, stride=1, bias=True, dimension=3),
        )

        self.q_up_1 = GenerativeUpBlock(2, 2)
        self.q_up_2 = GenerativeUpBlock(2, 2)
        self.q_up_3 = GenerativeUpBlock(2, 2)

        """
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
        """
        
        self.q_predict_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.q_predict_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N1, out_channels=N1*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.q_predict_3 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=2, out_channels=N2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2, out_channels=N2, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N2, out_channels=N2*2, kernel_size=3, stride=1, bias=True, dimension=3),
        )


        # Auxiliary
        self.down_conv = ME.MinkowskiConvolution(in_channels=1, out_channels=1, kernel_size=3, stride=2, dimension=3)



    def forward(self, x, Q, coords=None, k=None):
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
        # Pre-Conv
        x = self.pre_conv(x)
        Q = self.q_pre_conv(Q)

        # Layer 1
        #Q = self.q_layers_1(Q)
        beta_gamma = self.q_predict_1(Q)
        x = self.scale_1(x, beta_gamma)

        x, predict_2, up_coords = self.up_1(x, k=k[0])
        Q = self.q_up_1(Q, up_coords)

        # Layer 2
        #Q = self.q_layers_2(Q)
        beta_gamma = self.q_predict_2(Q)
        x = self.scale_2(x, beta_gamma)

        x, predict_1, up_coords = self.up_2(x, k=k[1])
        Q = self.q_up_2(Q, up_coords)

        # Layer 3
        #Q = self.q_layers_3(Q)
        beta_gamma = self.q_predict_3(Q)
        x = self.scale_3(x, beta_gamma)

        x, predict_final, up_coords = self.up_3(x, k=k[2])
        Q = self.q_up_3(Q, up_coords)

        # Post Conv
        x = self.post_conv(x)

        if coords is not None:
            predictions = [predict_2, predict_1, predict_final]
            with torch.no_grad():
                points_1 = self.down_conv(coords)
                points_2 = self.down_conv(points_1)
            points = [points_2, points_1, coords]
            return x, points, predictions

        else:
            return x

