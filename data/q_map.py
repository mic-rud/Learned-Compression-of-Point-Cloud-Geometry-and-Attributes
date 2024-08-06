import torch
import random
import math
import MinkowskiEngine as ME


class Q_Map2(object):
    """
    Generator for Quality maps
    """
    def __init__(self, config):
        self.a_A = math.log2(config["lambda_A_max"] + config["lambda_A_min"])
        self.b_A = config["lambda_A_min"] - 1

        self.a_G = math.log2(config["lambda_G_max"] + config["lambda_G_min"])
        self.b_G = config["lambda_G_min"] - 1


    def __call__(self, geometry):
        """
        Documentation
        
        Parameters
        ----------
        geometry: ME.SparseTensor
            Description
        
        returns
        -------
        q_map: ME.SparseTensor
            Q_Map of the data
        """
        batch_indices = torch.unique(geometry.C[:, 0])
        N = geometry.C.shape[0]

        q_map = ME.SparseTensor(coordinates=geometry.C, features=torch.zeros(N, 2), device=geometry.device)
        for batch_idx in batch_indices:
            mask = geometry.C[:,0]==batch_idx
            self.random_q_map(geometry, q_map, mask)

        # Scale 
        lambda_map = self.scale_q_map(q_map)
        return q_map, lambda_map

    def scale_q_map(self, q_map):
        lambda_map_features = q_map.F.clone()
        lambda_map_features[:, 0] = 2**(lambda_map_features[:, 0] * self.a_G) + self.b_G
        lambda_map_features[:, 1] = 2**(lambda_map_features[:, 1] * self.a_A) + self.b_A

        lambda_map = ME.SparseTensor(coordinates=q_map.C, 
                                features=lambda_map_features, 
                                coordinate_manager=q_map.coordinate_manager)
        return lambda_map

        
    def random_q_map(self, geometry, q_map, mask):
        """
        Builds a random Q_map for training
        
        Parameters
        ----------
        args: Datatype
            Description
        """
        coordinates = geometry.C[mask]
        features = geometry.F[mask]
        for i in range(2):
            choice = random.choice(range(2))

            if choice == 0:
                q_feats = self.gradient(coordinates)
            elif choice == 1:
                q_feats = self.uniform(coordinates)
            elif choice == 2:
                q_feats = self.empty(coordinates)
            elif choice == 3:
                q_feats = self.variance(coordinates, features, i)


            # Replace in q_map
            q_map.F[mask, i] = q_feats


    def gradient(self, coordinates):
        """
        Builds a gradient Q_map based on the geometry
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """
        offset = random.uniform(-0.5, 0.5)
        direction = random.randint(1,3)
        min, max = torch.min(coordinates[:, direction]), torch.max(coordinates[:, direction])

        q_feats = torch.clamp((coordinates[:, direction] - min) / (max - min + 1e-10) + offset, 0, 1) 
        return q_feats

    def uniform(self, coordinates):
        """
        Builds a uniform Q_map based on the geometry
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """
        N = coordinates.shape[0]
        q_feats = torch.ones((N, 1), device=coordinates.device) * random.uniform(0, 1)
        return q_feats

    def empty(self, coordinates):
        """
        Builds a uniform Q_map based on the geometry
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """
        N = coordinates.shape[0]
        q_feats = torch.zeros((N, 1), device=coordinates.device)
        return q_feats

    def variance(self, coordinates, features, mode):
        """
        Builds a variance Q_map based on the features/coordinates complexity
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """

            


class Q_Map(object):
    """
    Generator for Quality maps
    """
    def __init__(self, config):
        self.mode = config["mode"]
        if self.mode == "exponential":
            self.a_A = math.log2(config["lambda_A_max"] + config["lambda_A_min"])
            self.b_A = config["lambda_A_min"] - 1
            self.a_G = math.log2(config["lambda_G_max"] + config["lambda_G_min"])
            self.b_G = config["lambda_G_min"] - 1
        elif self.mode == "quadratic":
            self.a_A = config["lambda_A_max"] - config["lambda_A_min"]
            self.b_A = config["lambda_A_min"]
            self.a_G = config["lambda_G_max"] - config["lambda_G_min"]
            self.b_G = config["lambda_G_min"]


    def __call__(self, geometry):
        """
        Documentation
        
        Parameters
        ----------
        geometry: ME.SparseTensor
            Description
        
        returns
        -------
        q_map: ME.SparseTensor
            Q_Map of the data
        """
        batch_indices = torch.unique(geometry.C[:, 0])
        N = geometry.C.shape[0]

        q_map = ME.SparseTensor(coordinates=geometry.C, features=torch.zeros(N, 2), device=geometry.device)
        for batch_idx in batch_indices:
            mask = geometry.C[:,0]==batch_idx
            self.random_q_map(geometry, q_map, mask)

        # Scale 
        lambda_map = self.scale_q_map(q_map)
        return q_map, lambda_map


    def scale_q_map(self, q_map):
        """
        Scales the q_map to receive a Lambda map for loss computation
        """
        lambda_map_features = q_map.F.clone()
        if self.mode == "exponential":
            lambda_map_features[:, 0] = 2**(lambda_map_features[:, 0] * self.a_G) + self.b_G
            lambda_map_features[:, 1] = 2**(lambda_map_features[:, 1] * self.a_A) + self.b_A
        elif self.mode == "quadratic":
            lambda_map_features[:, 0] = lambda_map_features[:, 0]**2 * self.a_G + self.b_G
            lambda_map_features[:, 1] = lambda_map_features[:, 1]**2 * self.a_A + self.b_A
        else:
            raise ValueError("Unknown Q_map mode")

        lambda_map = ME.SparseTensor(coordinates=q_map.C, 
                                features=lambda_map_features, 
                                coordinate_manager=q_map.coordinate_manager)
        return lambda_map

        
    def random_q_map(self, geometry, q_map, mask):
        """
        Builds a random Q_map for training
        
        Parameters
        ----------
        args: Datatype
            Description
        """
        coordinates = geometry.C[mask]
        features = geometry.F[mask]
        choice = random.choice(range(2))

        if choice == 0:
            q_feats = self.gradient(coordinates)
        elif choice == 1:
            q_feats = self.uniform(coordinates)
        elif choice == 3:
            q_feats = self.variance(coordinates, features, i)


        # Replace in q_map
        q_map.F[mask] = q_feats


    def gradient(self, coordinates):
        """
        Builds a gradient Q_map based on the geometry
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """
        direction = random.randint(1,3)
        min, max = torch.min(coordinates[:, direction]), torch.max(coordinates[:, direction])

        q_feats = torch.clamp((coordinates[:, direction] - min) / (max - min + 1e-10), 0, 1)
        q_feats = q_feats.unsqueeze(1).repeat(1, 2)
        return q_feats

    def uniform(self, coordinates):
        """
        Builds a uniform Q_map based on the geometry
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """
        N = coordinates.shape[0]
        scale_geometry = random.uniform(0,1)
        scale_attribute = random.uniform(0,1)
        q_feats = torch.ones((N, 2), device=coordinates.device)
        q_feats[:, 0] *= scale_geometry
        q_feats[:, 1] *= scale_attribute
        return q_feats

    def empty(self, coordinates):
        """
        Builds a uniform Q_map based on the geometry
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """
        N = coordinates.shape[0]
        q_feats = torch.zeros((N, 1), device=coordinates.device)
        return q_feats

    def variance(self, coordinates, features, mode):
        """
        Builds a variance Q_map based on the features/coordinates complexity
        
        Parameters
        ----------
        args: Datatype
            Description
        
        """

            

