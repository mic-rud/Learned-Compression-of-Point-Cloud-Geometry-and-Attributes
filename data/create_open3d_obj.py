import numpy as np
import open3d as o3d
from dataloader import PointCloudDataset
from transforms import *

def open_cube(cube):
    obj = o3d.geometry.PointCloud()
    obj.points = o3d.utility.Vector3dVector(cube['points'])
    obj.colors = o3d.utility.Vector3dVector(cube['colors'])
    if 'normals' in cube:
        obj.normals = o3d.utility.Vector3dVector(cube['normals'])
    return obj
    
if __name__ == "__main__":
    trainset2 = PointCloudDataset("./data/dataset_dev", split="train")
    testset2 = PointCloudDataset("./data/dataset_dev", split="test")

    tsr = CubetoTensor()
    norm = Normalize()
    crop = Crop([1,1,1,1,1,1])
    rot = Rotate('x',90)
    scl = Scale([1,1,1])
    voxelise = Voxelise(64)
    voxeliseINV = DeVoxelise()
    cube2 = voxeliseINV((voxelise(trainset2[0])))
    cube = trainset2[0]
    obj1 = open_cube(cube)
    obj2 = open_cube(cube2)
    error = obj1.compute_point_cloud_distance(obj2)
    error2 = obj2.compute_point_cloud_distance(obj1)
    print(np.mean(error2))
    print(np.asarray(error).mean())
    o3d.visualization.draw_geometries([obj1])