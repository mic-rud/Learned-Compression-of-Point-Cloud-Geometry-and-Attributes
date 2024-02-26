import torch
import math
import MinkowskiEngine as ME

class Loss():
    """
    Wrapper holding loss functions
    """

    def __init__(self, config):
        """
        Parameters
        ----------
        config: dict
            Dictionary containing loss configurations
        """
        self.losses = {}
        for id, setting in config.items():
            key = setting["type"]
            setting["id"] = id

            # Match key to the respective class
            match key:
                case "BPPLoss":
                    self.losses[id] = BPPLoss(setting)
                case "ColorLoss":
                    self.losses[id] = ColorLoss(setting)
                case "StackedBPPLoss":
                    self.losses[id] = StackedBPPLoss(setting)
                case "StackedColorLoss":
                    self.losses[id] = StackedColorLoss(setting)
                    
    def __call__(self, gt, pred):
        """
        Call the loss function to return sum of all losses
        
        Parameters
        ----------
        gt: ME.SparseTensor
            Ground truth point cloud
        pred: dict
            Dictionary containing information for computing the loss

        returns
        -------
        total_loss: torch.tensor
            Total loss after adding and weighting
        losses: dict
            Dictionary containing the loss value per loss
        """
        total_loss = 0
        losses = {}
        for _, loss in self.losses.items():
            loss_item = loss(gt, pred)
            losses[loss.identifier] = loss_item
            total_loss += loss_item
        
        return total_loss, losses
        
class BPPLoss():
    """
    BPP loss
    """
    def __init__(self, config):
        self.weight = config["weight"]
        self.identifier = config["id"]
        self.key = config["key"]

    def __call__(self, gt, pred):
        loss = 0.0
        likelihoods = pred["likelihoods"][self.key]
        if isinstance(likelihoods, list):
            likelihoods = torch.cat(likelihoods, dim=-1)
        num_points = gt.C.shape[0]

        bits = torch.log(likelihoods).sum() / (- math.log(2) * num_points)
        loss += bits


        return loss.mean() * self.weight
    

class StackedBPPLoss():
    """
    Stacked BPP loss
    """
    def __init__(self, config):
        self.weight = config["weight"]
        self.identifier = config["id"]
        self.key = config["key"]

    def __call__(self, gt, pred):
        num_points = gt.C.shape[0]

        loss = 0.0
        likelihoods = pred["likelihoods"][self.key]
        cur_likelihoods = []

        likelihoods = torch.cat(likelihoods, dim=-1)

        for likelihood in likelihoods:
            if isinstance(likelihood, list):
                likelihood = torch.cat(likelihood, dim=-1)

            cur_likelihoods.append(likelihood)
            cur_likelihood = torch.cat(cur_likelihoods, dim=-1)

            #bits = torch.log(likelihood).sum() / (- math.log(2) * num_points)
            bits = torch.log(cur_likelihood).sum() / (- math.log(2) * num_points)
            loss += bits

        return loss.mean() * self.weight

class ColorLoss():
    """
    ColorLoss using L2/L1 on GT voxel locations
    """
    def __init__(self, config):
        self.weight = config["weight"]
        self.identifier = config["id"]
        if config["loss"] == "L1":
            self.loss_func = torch.nn.L1Loss(reduction="mean")
        elif config["loss"] == "L2":
            self.loss_func = torch.nn.MSELoss(reduction="mean")

    def __call__(self, gt, pred):
        prediction = pred["prediction"]

        pred_colors = prediction.features_at_coordinates(gt.C.float())
        gt_colors = gt.F

        color_loss = self.loss_func(gt_colors, pred_colors) 

        return self.weight * color_loss    
    
class StackedColorLoss():
    def __init__(self, config):
        self.weights = config["weight"]
        self.identifier = config["id"]
        if config["loss"] == "L1":
            self.loss_func = torch.nn.L1Loss(reduction="mean")
        elif config["loss"] == "L2":
            self.loss_func = torch.nn.MSELoss(reduction="mean")

    def __call__(self, gt, pred):
        predictions = pred["prediction"]
        loss = 0.0

        for i, prediction in enumerate(predictions):
            gt_colors = gt.F
            pred_colors = prediction.features_at_coordinates(gt.C.float())

            # ColorLoss
            color_loss = self.loss_func(gt_colors, pred_colors) 
            print(color_loss)
            loss += self.weights[i] * color_loss

        return loss