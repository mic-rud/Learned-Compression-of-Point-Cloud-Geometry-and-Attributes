import torch
import torch.nn as nn
import MinkowskiEngine as ME
import copy

import utils



class ScaledBlock(torch.nn.Module):
    def __init__(self, N, encode=True, scale=True):
        super().__init__()
        self.encode = encode
        self.scale = scale

        self.conv_1 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
        )
        self.conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N, out_channels=N, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )
        self.gdn = MinkowskiGDN(N, inverse=(not self.encode))

    def forward(self, x, condition):
        x_res = ME.SparseTensor(coordinates=x.C,
                                features=x.F.clone(),
                                device=x.device,
                                tensor_stride=x.tensor_stride)
        x = self.conv_1(x)

        # Scale and shift
        beta, gamma = condition.features_at_coordinates(x.C.float()).chunk(2, dim=1)

        if self.scale:
            feats = x.F * beta + gamma
         

        x = ME.SparseTensor(coordinates=x.C, 
                            features=feats, 
                            device=x.device, 
                            tensor_stride=x.tensor_stride)

        x = self.conv_2(x)
        x = ME.SparseTensor(coordinates=x.C,
                                features=x.F + x_res.features_at_coordinates(x.C.float()),
                                device=x.device,
                                tensor_stride=x.tensor_stride)
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
    def __init__(self, N_in, N_out, predict=False, dense=True):
        super().__init__()
        self.dense = dense

        self.conv = ME.MinkowskiGenerativeConvolutionTranspose(in_channels=N_in, out_channels=N_out, kernel_size=3, stride=2, bias=True, dimension=3)
        self.conv_2 = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
            ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3)
        )
        self.prune = ME.MinkowskiPruning()

        self.predict = predict
        if self.predict:
            self.occ_predict = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(in_channels=N_out, out_channels=N_out, kernel_size=3, stride=1, bias=True, dimension=3),
            )


    def _prune_coords(self, x, occupied_points=None):
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
    
    def _topk_prediction(self, prediction, k):
        """
        Mask the top-k elements for each batch in prediction to get attributes at predicted points.
        """
        batch_indices = torch.unique(prediction.C[:, 0])  # Get unique batch IDs
        pred_occupancy_mask = torch.zeros_like(prediction.F[:, 0], dtype=torch.bool)

        for batch_idx in batch_indices:
            # Mask for current batch
            current_batch_mask = prediction.C[:, 0] == batch_idx

            # Extract the predictions for the current batch and get top-k
            current_preds = prediction.F[current_batch_mask, 0]
            current_k = k[batch_idx]
            _, top_indices = torch.topk(current_preds, int(current_k))
    
            # Use advanced indexing to set the top-k indices to True
            indices_for_current_batch = torch.nonzero(current_batch_mask).squeeze()
            pred_occupancy_mask[indices_for_current_batch[top_indices]] = True

        return pred_occupancy_mask

    def forward(self, x, coords=None, k=None):
        x = self.conv(x)

        if self.predict:
            if self.dense:
                # Standard dense upsampling
                x = self.conv_2(x)

                predictions = self.occ_predict(x)

                occupancy_mask = self._topk_prediction(predictions, k)
                up_coords = predictions.C[occupancy_mask]

                x = self._prune_coords(x, up_coords)
            else:
                predictions = self.occ_predict(x)
                # Not dense ablation
                occupancy_mask = self._topk_prediction(predictions, k)
                up_coords = predictions.C[occupancy_mask]

                x = self._prune_coords(x, up_coords)
                x = self.conv_2(x)

            return x, predictions, up_coords

        else:
            x = self._prune_coords(x, coords)
            return x



class ConditionEncoder(nn.Module):
    def __init__(self, C_in, N_scales, N_features):
        super().__init__()
        self.num_stages = len(N_scales)

        self.pre_conv = nn.Sequential(
            ME.MinkowskiConvolution(in_channels=C_in, out_channels=N_features[0], kernel_size=3, stride=1, bias=True, dimension=3),
            ME.MinkowskiReLU(inplace=False),
        )

        self.conv_layers = nn.ModuleList() 
        self.predict_layers = nn.ModuleList()
        self.down_layers = nn.ModuleList()  
        
        for i in range(self.num_stages):
            down_layer = ME.MinkowskiConvolution(in_channels=N_features[i], out_channels=N_features[i+1], kernel_size=3, stride=2, bias=True, dimension=3)
            self.down_layers.append(down_layer)
            self._register_layers(down_layer, f'down_layers_{i}')

            conv_layer = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=N_features[i+1], out_channels=N_features[i+1], kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(in_channels=N_features[i+1], out_channels=N_features[i+1], kernel_size=3, stride=1, bias=True, dimension=3),
            )
            self.conv_layers.append(conv_layer)
            self._register_layers(conv_layer, f'conv_layers_{i}')
            
            predict_layer = nn.Sequential(
                ME.MinkowskiConvolution(in_channels=N_features[i+1], out_channels=N_scales[i], kernel_size=3, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(in_channels=N_scales[i], out_channels=N_scales[i], kernel_size=1, stride=1, bias=True, dimension=3),
                ME.MinkowskiReLU(inplace=False),
                ME.MinkowskiConvolution(in_channels=N_scales[i], out_channels=N_scales[i]*2, kernel_size=3, stride=1, bias=True, dimension=3),
            )
            self.predict_layers.append(predict_layer)
            self._register_layers(predict_layer, f'predict_layers_{i}')




    def _register_layers(self, layer, name):
        for id, param in layer.named_parameters():
            param_name = f'{name}_{id}'.replace('.', '_')
            self.register_parameter(param_name, param)



    def forward(self, Q):
        Q = self.pre_conv(Q)

        beta_gammas = []
        for i in range(self.num_stages):
            Q = self.down_layers[i](Q)
            #Q = self.conv_layers[i](Q)
            beta_gamma = self.predict_layers[i](Q)

            beta_gammas.append(beta_gamma)
        
        return Q, beta_gammas
        





from compressai.layers import GDN
import torch.nn.functional as F
class MinkowskiGDN(GDN):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.unique(x.C[:, 0])  # Get unique batch IDs
        _, C = x.F.size()

        beta = self.beta_reparam(self.beta)
        gamma = self.gamma_reparam(self.gamma)
        gamma = gamma.reshape(C, C, 1)

        for batch_idx in batch_indices:
            # Mask for current batch
            current_batch_mask = x.C[:, 0] == batch_idx

            current_feats = x.F[current_batch_mask]
            norm = F.conv1d(torch.abs(current_feats.T).unsqueeze(0), gamma, beta)
            norm = norm[0].T

            if not self.inverse:
                norm = 1.0 / norm

            out = current_feats * norm
            x.F[current_batch_mask] = out
        return x



