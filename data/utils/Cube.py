import os
import open3d as o3d
import numpy as np
import copy

class Cube:
    """
    Abstraction holding cubes in the point cloud
    """
    def __init__(self) -> None:
        self.pointcloud = None
        self.position = (None, None, None)
        self.dimension = (None, None, None)
        self.num_points = None


    def write(self, path):
        """
        Write the cube to the path defined in self.path
        """
        if path is not None:
            o3d.io.write_point_cloud(path, self.pointcloud)
        else:
            ValueError("Cube path not set for storing.")


    def read(self, path):
        """
        Write the cube to the path defined in self.path
        """
        if not os.path.exists(path):
            ValueError("Path {} is not existant")

        self.pointcloud = o3d.io.read_point_cloud(path)
        print(np.min(np.array(self.pointcloud.points)))
        print(np.max(np.array(self.pointcloud.points)))


    def create_cube(self, pointcloud, position, dimension):
        """
        Set the pointcloud for the cube at position and dimensions

        args:
            pointcloud : open3d.geometry.PointCloud
                The point cloud containing all points in the cube
            position : tuple
                The position of the cube in the voxelized space
            dimensions : tuple
                The dimensions of the cube in the voxelized space
        """
        if not (isinstance(position, list) and len(position) == 3):
            raise AssertionError("position is not a list of size 3.")
        if not (isinstance(dimension, list) and len(dimension) == 3):
            raise AssertionError("dimensions is not a list of size 3.")
        if not isinstance(pointcloud, o3d.geometry.PointCloud):
            raise AssertionError("dimensions is not a tuple of size 3.")

        self.pointcloud = copy.deepcopy(pointcloud) #Deep copy required
        self.dimension = dimension
        self.position = position
        self.num_points = len(pointcloud.points)
        return

    
    def get_info(self):
        """
        Get the info as dict
        returns:
            info : dict
                Dictionary containing position, dimension and num_points
        """
        info = {}
        info["position"] = self.position
        info["dimension"] = self.dimension
        info["num_points"] = self.num_points
        return info

    def set_info(self, info):
        self.position = info["position"]
        self.dimension = info["dimension"]
        self.num_points = info["num_points"]

    def get_orig_pointcloud(self):
        """
        Get the point cloud shifted by position
        """
        shifted_pointcloud = copy.deepcopy(self.pointcloud)
        points = np.asarray(shifted_pointcloud.points)
        points = points + self.position
        shifted_pointcloud.points = o3d.utility.Vector3dVector(points)
        return shifted_pointcloud