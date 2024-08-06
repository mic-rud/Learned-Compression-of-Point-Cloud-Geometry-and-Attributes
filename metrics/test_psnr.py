import numpy as np
import open3d as o3d
import time

from metric import PointCloudMetric2, PointCloudMetric

def test_psnr(pcd1, pcd2):
    ## Our implementation
    start_1 = time.time()
    metric = PointCloudMetric2(pcd1, pcd2, normalize=False)
    metric.compute_geometry_metrics(peak_signal=1023)
    metric.compute_color_psnr()
    result_1 = metric.get_result()
    print(result_1)
    time_1 = time.time() - start_1
    print(time_1)

    start_2 = time.time()
    metric = PointCloudMetric(pcd1, pcd2, resolution=1023)
    result2  = metric.compute_pointcloud_metrics(drop_duplicates=False)
    print(result2)
    time_2 = time.time() - start_2
    print(time_2)


if __name__ == "__main__":
    pcd1 = o3d.io.read_point_cloud("./test_clouds/original.ply")
    pcd2 = o3d.io.read_point_cloud("./test_clouds/reconstruct.ply")
    result = test_psnr(pcd1, pcd2)
    """
    for i in range(100):
        num_points1 = np.random.randint(100000, 1000000)
        num_points2 = num_points1 + np.random.randint(np.round(num_points1 * 0.9), np.round(num_points1 * 1.1)) 

        # Generate random points and colors for each point cloud
        points1 = np.random.rand(num_points1, 3)
        colors1 = np.random.rand(num_points1, 3)

        points2 = np.random.rand(num_points2, 3)
        colors2 = np.random.rand(num_points2, 3)

        # Create two point clouds
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(points1)
        pcd1.colors = o3d.utility.Vector3dVector(colors1)

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(points2)
        pcd2.colors = o3d.utility.Vector3dVector(colors2)

        # Call and time your two implementations
        result = test_psnr(pcd1, pcd2)
    """

