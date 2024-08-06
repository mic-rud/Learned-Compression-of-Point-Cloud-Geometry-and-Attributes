import numpy as np
import torch
import MinkowskiEngine as ME
import torchvision
import random
import os


def build_transforms(config):
    """
    Wrapper class for building transforms
    """
    transforms = []
    if not config:
        return transforms
    
    config = {k: config[k] for k in sorted(config)}

    for name, setting in config.items():
        key = setting["key"]
        print(name)
        if key == "Normalize":
            block_size = setting["block_size"]
            transforms.append(Normalize(block_size))

        elif key == "Denormalize":
            block_size = setting["block_size"]
            transforms.append(Denormalize(block_size))

        elif key == "ColorShift":
            transforms.append(ColorShift())

        elif key == "RandomRotate":
            block_size = setting["block_size"]
            transforms.append(RandomRotate(block_size))

        elif key == "RandomNoise":
            probability = setting["probability"]
            std = setting["std"]
            transforms.append(RandomNoise(probability, std))

        elif key == "RGBtoYUV":
            transforms.append(RGBtoYUV())

        elif key == "ColorJitter":
            transforms.append(ColorJitter())

        elif key == "YUVtoRGB":
            transforms.append(YUVtoRGB())

        elif key == "Voxelise":
            block_size = setting["block_size"]
            transforms.append(Voxelize(block_size))

        elif key == "Devoxelise":
            block_size = setting["block_size"]
            transforms.append(Devoxelize(block_size))

        elif key == "ProjectTexture":
            block_size = setting["block_size"]
            dataset_dir = setting["dataset_dir"]
            probability = setting["probability"]
            transforms.append(ProjectTexture(dataset_dir, block_size, probability))

        elif key == "Minkowski":
            transforms.append(Minkowski())
        else:
            raise ValueError("Transform {key} not defined.")
    
    print(transforms)
    return transforms

class ColorShift(object):
    """
    Randomly shift the colors 
    Each channel is shifted independently
    """
    def __init__(self):
        pass

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)
        
    def transform(self, sample):
        with torch.no_grad():
            shifts = torch.rand(1) * torch.ones(1,3)

            # Calculate the potential new min and max after applying the shifts
            potential_mins = torch.min(sample["colors"] + shifts, dim=0).values
            potential_maxs = torch.max(sample["colors"] + shifts, dim=0).values

            # Adjust the shifts for each channel
            lower_bound_adjustments = torch.clamp(potential_mins, max=0)
            upper_bound_adjustments = torch.clamp(potential_maxs - 1, min=0)

            adjusted_shifts = shifts - lower_bound_adjustments + upper_bound_adjustments

            sample["colors"] = (sample["colors"] + adjusted_shifts) % 1

        return sample

class ColorJitter(object):
    """
    Randomly shift the colors 
    Each channel is shifted independently
    """
    def __init__(self):
        self.jitter = torchvision.transforms.ColorJitter(brightness=0.3,
                                                         contrast=0.3,
                                                         saturation=0.3,
                                                         hue=0.3)

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)
        
    def transform(self, sample):
        sample["colors"] = self.jitter(sample["colors"].T.unsqueeze(-1))
        sample["colors"] = sample["colors"].squeeze(-1).T

        return sample

class RGBtoYUV(object):
    '''
    Transforms the colors of the cube from RGB to YUV (BT.709)
    (We do not shift U and V to -0.5:0.5 but leave it in range 0:1)
    '''
    def __init__(self):
        #BT 709 RGB to YUV
        self.color_matrix = torch.tensor([[0.2126   , 0.7152   , 0.00722   ],
                                          [-0.1146, -0.3854, 0.5   ],
                                          [0.5   , -0.4542, 0.0458]])
        pass

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)
        
    def transform(self, sample):
        with torch.no_grad():
            sample['colors'] = torch.matmul(sample["colors"], self.color_matrix)
        
        return sample


class YUVtoRGB(object):
    '''
    Transforms the colors of the cube from YUV to RGB
    '''
    def __init__(self):
        #BT 709 YUV to RGB
        self.color_matrix = torch.tensor([[1.1643835616, 0.0   , 1.17927410714],
                                          [1.1643835616, -0.2132486143, -0.5329093286],
                                          [1.1643835616, 2.1124017857, 0.0]])
        self.zero_offset = torch.tensor([-0.972945075, 0.301482665, -1.133402218])
        pass

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)
        
    def transform(self, sample):
        with torch.no_grad():
            sample["colors"][:, 1:] = sample["colors"][:, 1:] - 0.5 # Offset to -0.5:0.5

            sample['colors'] = torch.matmul(sample["colors"], self.color_matrix) + self.zero_offset
            sample["colors"] = torch.clamp(sample["colors"], 0.0, 1.0)
        
        return sample



class Voxelize(object):
    '''
    Transform cube from pointcloud-structure to voxel-structure.
    '''
    def __init__(self, block_size):
        self.block_size = block_size

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self,sample):
        with torch.no_grad():
            voxel = torch.zeros((4, self.block_size, self.block_size, self.block_size))

            points = torch.clamp(sample["points"], 0, self.block_size - 1)
            voxel_indices = points.round().int()

            # Convert 3D voxel indices to unique 1D indices
            indices_1d = voxel_indices[:, 0] * self.block_size**2 + voxel_indices[:, 1] * self.block_size + voxel_indices[:, 2]
       
            voxel[1:, indices_1d // (self.block_size**2), 
                    (indices_1d // self.block_size) % self.block_size, 
                    indices_1d % self.block_size] = sample["colors"].T

            # Set occupancy mask
            voxel[0, indices_1d // (self.block_size**2), 
                    (indices_1d // self.block_size) % self.block_size, 
                    indices_1d % self.block_size] = 1

            sample["voxels"] = voxel

            # Pop points and colors?
            sample.pop("points")
            sample.pop("colors")
        return sample


class Devoxelize(object):
    '''
    Transform voxel-structure back to pointcloud-structure from a cube.
    '''
    def __init__(self, block_size):
        self.block_size = block_size
        pass

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        with torch.no_grad():
            voxel = sample["voxels"]
        
            # Extract the occupancy mask and color grid
            occupancy_mask = voxel[0]
        
            # Get the indices of the occupied voxels
            occupied_indices = torch.nonzero(occupancy_mask == 1).int()
        
            # Convert 3D voxel indices back to the center of voxel (add 0.5 for centering)
            retrieved_points = occupied_indices.float()
        
            # Fetch the colors for the occupied voxels
            i, j, k = occupied_indices[:, 0], occupied_indices[:, 1], occupied_indices[:, 2]
            retrieved_colors = voxel[1:, i, j, k].T  # Transpose to make it (N, 3)

            # Update the sample dictionary
            sample["points"] = retrieved_points
            sample["colors"] = retrieved_colors

            # Remove voxel representation?
            sample.pop("voxels")

        return sample


class Normalize(object):
    """
    Min-max normalization for a pointcloud / cube onto the intervall [-1,1]
    """
    def __init__(self, block_size):
        self.block_size = block_size


    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        sample['points'] = 2*sample['points'] / self.block_size - 1
        return sample


class Denormalize(object):
    """ 
    Inverse of Normalize() class
    """
    def __init__(self, block_size):
        self.block_size = block_size


    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        sample['points'] = (sample['points'] + 1) * self.block_size / 2
        return sample

class RandomNoise(object):
    """ 
    Inverse of Normalize() class
    """
    def __init__(self, probability, std):
        self.probability = probability
        self.std = std


    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        if torch.rand(1) < self.probability:
            noise = torch.randn(sample["colors"].shape[0]) * self.std
            sample["colors"] += noise.unsqueeze(1)
            sample["colors"] = torch.clip(sample["colors"], 0.0, 1.0)
        else:
            return sample
        return sample


class ProjectTexture(object):
    def __init__(self, dataset_dir, block_size, probability):
        self.block_size = block_size
        self.dataset_dir = dataset_dir

        self.dataset = torchvision.datasets.DTD(self.dataset_dir, download=True)
        self.crop = torchvision.transforms.RandomCrop(self.block_size, pad_if_needed=True)
        self.num_images = len(self.dataset)
        self.probability = probability


    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        if torch.rand(1) < self.probability:
            return sample
            
        idx = random.randint(0, self.num_images - 1)
        image, _ = self.dataset[idx]

        # Convert image to tensor and normalize
        image_tensor = torchvision.transforms.functional.to_tensor(image).to(sample["points"].device)  # [C, H, W]
        image_tensor = image_tensor.clone()
        image_tensor = self.crop(image_tensor)

        x = sample["points"][:, 0].long()
        y = sample["points"][:, 1].long()

        # Filter invalid dimensions
        H, W = image_tensor.shape[1], image_tensor.shape[2]
        valid_mask = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        x = x[valid_mask]
        y = y[valid_mask]

        # Use advanced indexing to gather colors
        sample["colors"][valid_mask] = image_tensor[:, y, x].permute(1, 0)

        """
        render_point_cloud(sample["points"], sample["colors"], "augment.png")
        import time
        time.sleep(1)
        """
        return sample


def render_point_cloud(points, colors, save_path, width=800, height=600, bg_color=(1.0, 1.0, 1.0)):
    """
    Render a point cloud off-screen and save to an image.

    Parameters:
        - points: numpy array of shape (N, 3) representing the point cloud coordinates.
        - colors: numpy array of shape (N, 3) representing RGB colors.
        - save_path: path to save the rendered image.
        - width: width of the output image.
        - height: height of the output image.
        - bg_color: tuple specifying the RGB background color of the rendering.
    """
    import open3d as o3d
    
    # Create a point cloud object
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pcd)
    # Adjust the point size
    render_options = vis.get_render_option()
    render_options.point_size = 10  # adjust the size as required

    vis.update_renderer()
    vis.capture_screen_image(save_path, do_render=True)
    
    vis.destroy_window()

class RandomRotate(object):
    """ 
    Randomly rotate the point cloud in 3D using PyTorch, 
    filter out-of-bounds points, round them to integers, and remove duplicates.
    """
    def __init__(self, block_size, crop=False):
        self.block_size = block_size
        self.crop = crop

    def __call__(self, sample):
        if "cubes" in sample:
            for idx, cube in enumerate(sample["cubes"]):
                sample["cubes"][idx] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, sample):
        phi = torch.rand(1) * 2 * 3.141592653589793  # Random roll
        theta = torch.rand(1) * 2 * 3.141592653589793  # Random pitch

        rotation_matrix = self.rotation_matrix_3d(phi, theta)
        points = sample["points"].clone()
        colors = sample["colors"].clone()

        # Applying rotation to the points
        rotated_points = torch.mm(points - self.block_size/2, rotation_matrix.T)
        rotated_points = rotated_points + self.block_size/2

        if self.crop:
            # Filter points that are within [0, block_size)
            valid_indices = ((rotated_points >= 0) & (rotated_points < self.block_size)).all(dim=1)
            valid_points = rotated_points[valid_indices]
            valid_colors = colors[valid_indices]

        else:
            valid_points = rotated_points
            valid_colors = colors

        # Round the coordinates to integers
        rounded_points = torch.round(valid_points)

        # Remove duplicate points
        unique_points, indices = torch.unique(rounded_points, dim=0, return_inverse=True)
        _, first_occurrence_indices = indices.unique(return_inverse=True)
        if indices.shape[0] > 1000:
            # Apply rotation only if points are still present
            sample['points'] = rounded_points[first_occurrence_indices]
            sample['colors'] = valid_colors[first_occurrence_indices]

        return sample

    @staticmethod
    def rotation_matrix_3d(phi, theta):
        """Generate a random 3D rotation matrix using PyTorch for given Euler angles."""
        
        # Roll
        R_x = torch.tensor([[1, 0, 0],
                            [0, torch.cos(phi), -torch.sin(phi)],
                            [0, torch.sin(phi), torch.cos(phi)]])
        
        # Pitch
        R_y = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                            [0, 1, 0],
                            [-torch.sin(theta), 0, torch.cos(theta)]])
        
        # Compose rotations
        R = torch.mm(R_y, R_x)
        
        return R


class Minkowski(object):
    def __init__(self):
        pass

    def __call__(self, sample):
        if "cubes" in sample:
            for cube in sample["cubes"]:
                sample["cubes"] = self.transform(cube)
            return sample
        else:
            return self.transform(sample)

    def transform(self, cube):
        coords = cube['points'].t().detach().contiguous().int()
        print(coords.shape)
        print(coords.device)
        feats = cube['colors'].t().detach().contiguous().float()
        print(feats.shape)

        coords = torch.randint(1024, (3, 250000))
        feats = torch.randn(3, 250000)
        sparse = ME.SparseTensor(features=feats, coordinates=coords)

        print(sparse.shape)
        cube["sparsetensor"] = sparse
        return cube
