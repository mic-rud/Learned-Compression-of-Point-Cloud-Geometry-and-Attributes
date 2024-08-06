import os
import yaml
import open3d as o3d
import numpy as np

from .Cube import Cube



class CubeHandler:
    """
    Handling all point cloud cubes as well as writing and reading
    """
    def __init__(self, sequence, frameIdx) -> None:
        self.sequence = sequence
        self.frameIdx = frameIdx
        self.cubes = []


    def add_cube(self, 
                 pointcloud, 
                 position, 
                 dimension):
        """
        Adds a cube to the list of cubes
        """
        cube = Cube()
        cube.create_cube(pointcloud, position, dimension)
        self.cubes.append(cube)



    def read(self, path):
        """
        Read point cloud cubes for a full point cloud
        args:
            path : str
                Root path to write to. Will generate the path if existant
        """
        sequence_path = os.path.join(path, self.sequence)
        frame_path = os.path.join(sequence_path, str(self.frameIdx))
        side_info_path = os.path.join(frame_path, "side_info.yaml")
        
        with open(side_info_path, "r") as file:
            cubes_info = yaml.safe_load(file)

        for key, cube_info in cubes_info.items():
            idx = int(key)
            cube_path = os.path.join(frame_path, "cube_{:05d}.ply".format(idx))

            cube = Cube()
            cube.set_info(cube_info)
            cube.read(cube_path)
            self.cubes.append(cube)


    def get_cube_paths(self):
        """
        Returns the relative path to all cubes
        """
        base_path = os.path.join(self.sequence, str(self.frameIdx))
        cube_paths = []
        for idx, _ in enumerate(self.cubes):
            cube_paths.append(os.path.join(base_path, "cube_{:05d}.ply".format(idx)))

        return cube_paths
            

    def get_num_points(self):
        """
        Returns the number of points for all cubes
        """
        num_points = []
        for idx, cube in enumerate(self.cubes):
            num_points.append(cube.num_points)

        return num_points

        
    def get_pointcloud(self):
        """
        Get the point cloud from cubes
        """
        colors = True
        normals = True
        for cube in self.cubes:
            if not cube.pointcloud.has_colors():
                colors = False
            if not cube.pointcloud.has_normals():
                normals = False

        all_points = np.empty(((0,3)))
        if colors:
            all_colors = np.empty(((0,3)))
        if normals:
            all_normals = np.empty(((0,3)))

        for cube in self.cubes:
            cube_points = cube.get_orig_pointcloud()
            all_points = np.append(all_points, cube_points.points, axis=0)
            if colors:
                all_colors = np.append(all_colors, cube_points.colors, axis=0)
            if normals:
                all_normals = np.append(all_normals, cube_points.normals, axis=0)

        pointcloud = o3d.geometry.PointCloud()
        pointcloud.points = o3d.utility.Vector3dVector(all_points)
        if colors:
            pointcloud.colors = o3d.utility.Vector3dVector(all_colors)
        if normals:
            pointcloud.normals = o3d.utility.Vector3dVector(all_normals)
        return pointcloud

    
    def slice(self, pointcloud, block_size=(64, 64, 64)):
        """
        Slice a point_cloud object into cubes of defined dimensions
        args:
            point_cloud : open3d point cloud object
            dimensions : Tuple, dimensions of the cubes in x, y, z direction
        """
        # Bounding box of the point cloud object
        bbox = pointcloud.get_axis_aligned_bounding_box()

        # Assert if negative coordinates are present
        assert bbox.min_bound[0] >= 0, "Negative x coordinates"
        assert bbox.min_bound[1] >= 0, "Negative y coordinates"
        assert bbox.min_bound[2] >= 0, "Negative z coordinates"

        # Compute indices of cube offsets
        x_cube_num = np.ceil(bbox.max_bound[0] / block_size[0]).astype(int)
        y_cube_num = np.ceil(bbox.max_bound[1] / block_size[1]).astype(int)
        z_cube_num = np.ceil(bbox.max_bound[2] / block_size[2]).astype(int)

        # Iterate through all cubes
        for x in range(x_cube_num):
            for y in range(y_cube_num):
                for z in range(z_cube_num):
                    # Compute bounding boxes for the crop
                    min_bound = np.array([
                        x * block_size[0],
                        y * block_size[1],
                        z * block_size[2]
                    ])
                    max_bound = np.array([
                        min_bound[0] + block_size[0] - 1,
                        min_bound[1] + block_size[1] - 1,
                        min_bound[2] + block_size[2] - 1
                    ])

                    # Crop a cube out of point_cloud
                    crop_box = o3d.geometry.AxisAlignedBoundingBox(min_bound, max_bound)
                    cube_pointcloud = pointcloud.crop(crop_box)

                    # Skip empty cubes
                    if len(cube_pointcloud.points) == 0:
                        continue

                    # Scale
                    cube_points = np.asarray(cube_pointcloud.points)
                    cube_points = cube_points - min_bound
                    cube_pointcloud.points = o3d.utility.Vector3dVector(cube_points)

                    # Create a cube with the points
                    position = min_bound.tolist()
                    dimension = [block_size[0], block_size[1], block_size[2]]
                    self.add_cube(cube_pointcloud, position, dimension)


    def write(self, path):
        """
        Write  all data into a path of structure
        sequence
         |- frameIdx
            |-cube0.ply
            |-cube1.ply
            |-....
            |-side_info.txt

        args:
            path : str
                Root path to write to. Will generate the path if existant
        """
        sequence_path = os.path.join(path, self.sequence)
        frame_path = os.path.join(sequence_path, str(self.frameIdx))
        self._check_and_make_path(path)
        self._check_and_make_path(sequence_path)
        self._check_and_make_path(frame_path, warn=True)

        # Write all cubes to directory
        cubes_info = {}
        for idx, cube in enumerate(self.cubes):
            cube_info = cube.get_info()
            cube_info["sequence"] = self.sequence
            cube_info["frameIdx"] = self.frameIdx
            cube_path = os.path.join(frame_path, "cube_{:05d}.ply".format(idx))

            cubes_info[idx] = cube_info

            cube.path = cube_path
            cube.write(cube_path)
        
        # Write yaml
        yaml_path = os.path.join(frame_path, "side_info.yaml")
        with open(yaml_path, 'w') as file:
            yaml.dump(cubes_info, file, default_flow_style=False)


    def _check_and_make_path(self, path, warn=False):
        """
        Check if a path exists and generate it.
        args:
            path : str
                The path to be checked
            warn : bool (default=False)
                Warn the user that the directory exists and overwrite
        """
        head, tail = os.path.split(path)
        if not os.path.isdir(head):
             raise ValueError("{} is not existant. Not creating.".format(head))

        if not os.path.isdir(path):
            os.mkdir(path)
        elif warn:
            sub_files = os.listdir(path)
            print("Path {} exists with {} files - overwriting....".format(path, len(sub_files)))
            # TODO Overwrite the path

        return



if __name__== "__main__":
    import numpy as np
    import random
    from RawLoader import RawLoader
    # Create a cube handler
    raw_loader = RawLoader("./data/raw", "./data/configs/raw_loading.yaml")
    sequence = "BlueBackpack"
    frameIdx = 100
    pc = raw_loader.get_pointcloud(sequence, frameIdx)
    handler = CubeHandler(sequence, frameIdx)

    handler.slice(pc)
    """
    # Point Clouds for testing
    point_clouds = []
    for i in range(10):
        pc = o3d.geometry.PointCloud()
        pc.points = o3d.utility.Vector3dVector(np.random.randn(random.randint(500, 1000) ,3))
        point_clouds.append(pc)

    print("ready")
    # Add to handler
    for pc in point_clouds:
        handler.add_cube(pc, 
                         [random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)],
                         [random.randint(1, 100), random.randint(1, 100), random.randint(1, 100)])
    """

    # Test writing
    handler.write("./data/dataset_dev")

    handlerR = CubeHandler(sequence, frameIdx)
    handlerR.read("./data/dataset_dev")

    pc = handlerR.get_pointcloud()
    print(pc.has_colors())
    o3d.visualization.draw_geometries([pc])
