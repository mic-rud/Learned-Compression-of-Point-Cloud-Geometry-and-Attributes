import torch
import torch.nn as nn
import MinkowskiEngine as ME

import utils

class ScaledBlock(torch.nn.Module):
    def __init__(self, N, encode=True):
        super().__init__()
        self.encode = encode

        self.conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

    def forward(self, x, condition):
        x = self.conv_1(x)

        # Scale and shift
        beta, gamma = condition.features_at_coordinates(x.C.float()).chunk(2, dim=1)

        if self.encode:
            feats = x.F * nn.functional.sigmoid(beta) + gamma
        else:
            feats = (x.F * nn.functional.sigmoid(beta)) - gamma

        x = ME.SparseTensor(coordinates=x.C, 
                            features=feats, 
                            device=x.device, 
                            tensor_stride=x.tensor_stride)

        x = self.conv_2(x)
        return x



class ConvBlock(torch.nn.Module):
    def __init__(self, N):
        super().__init__()

        self.conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        return x




class GenerativeUpBlock(torch.nn.Module):
    def __init__(self, N_in, N_out):
        super().__init__()

        self.conv = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N_in, out_channels=N_out, kernel_size=3, stride=2, bias=True, dimension=3)
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
        guide_flat = (occupied_points.C.to(torch.int64) * scaling_factors).sum(dim=1)

        # Prune
        mask = torch.isin(x_flat, guide_flat)
        x = self.prune(x, mask)

        return x


    def forward(self, x, coords):
        x = self.conv(x)
        x = self._prune_coords(x, coords)
        return x


class ConditionEncoder(nn.Module):
    def __init__(self, C_in, N_scales, N_features):
        super().__init__()
        self.num_stages = len(N_features)

        self.pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=C_in, out_channels=N_features[0], kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.conv_layers = nn.ModuleList() 
        self.predict_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()  
        
        for i in range(self.num_stages):
            conv_layer = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_features[i], kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_features[i], kernel_size=3, stride=1, bias=True, dimension=3),
            )
            self.conv_layers.append(conv_layer)
            self._register_layers(conv_layer, f'conv_layers_{i}')
            
            predict_layer = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_scales[i], kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(in_channels=N_scales[i], out_channels=N_scales[i], kernel_size=3, stride=1, bias=True, dimension=3),
            )
            self.predict_layers.append(predict_layer)
            self._register_layers(predict_layer, f'predict_layers_{i}')

            down_layer = ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_features[i+1], kernel_size=3, stride=2, bias=True, dimension=3)
            self.down_layers.append(down_layer)
            self._register_layers(down_layer, f'down_layers_{i}')



    def _register_layers(self, layer, name):
        for id, param in layer.named_parameters():
            self.register_parameter(f'{name}_{id}', param)



    def forward(self, Q):
        Q = self.pre_conv(Q)

        beta_gammas = []
        for i in range(self.num_stages):
            Q = self.down_layers[i](Q)
            Q = self.conv_layers[i](Q)
            beta_gamma = self.predict_layers[i](Q)

            beta_gammas.append(beta_gamma)
        
        return Q, beta_gammas
        



class ConditionDecoder(nn.Module):
    def __init__(self, C_in, N_scales, N_features):
        super().__init__()
        self.num_stages = len(N_features)

        self.conv_layers = nn.ModuleList()
        self.predict_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()

        for i in range(self.num_stages):
            conv_layer = nn.Sequential(
                    ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_features[i], kernel_size=3, stride=1, bias=True, dimension=3),
                    ME.MinkowskiReLU(inplace=False),
                    ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_features[i], kernel_size=3, stride=1, bias=True, dimension=3),
                )
            self.conv_layers.append(conv_layer)
            self._register_layers(conv_layer, f"conv_layers_{i}")

            predict_layer = nn.Sequential(
                    ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_scales[i], kernel_size=3, stride=1, bias=True, dimension=3),
                    ME.MinkowskiReLU(inplace=False),
                    ME.MinkowskiConvolution(in_channels=N_scales[i], out_channels=N_scales[i], kernel_size=3, stride=1, bias=True, dimension=3),
                )
            self.predict_layers.append(predict_layer)
            self._register_layers(predict_layer, f"predict_layer_{i}")

            up_layer = GenerativeUpBlock(N_features[i], N_features[i+1])
            self.up_layers.append(up_layer)
            self._register_layers(up_layer, f"up_layer_{i}")



    def _register_layers(self, layer, name):
        for id, param in layer.named_parameters():
            self.register_parameter(f'{name}_{id}', param)
 

    def forward(self, Q, coords=None):
        beta_gammas = []
        for i in range(self.num_stages):
            Q = self.conv_layers[i](Q)
            beta_gamma = self.predict_layers[i](Q)

            beta_gammas.append(beta_gamma)

            Q = self.up_layers[i](Q, coords)
        
        return Q, beta_gammas
        


