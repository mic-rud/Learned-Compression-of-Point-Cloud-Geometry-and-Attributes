import time
import yaml
import os
import open3d as o3d
import numpy as np


class RawLoader():
    """
    RawLoader class to handle loading of raw point cloud data
    """
    def __init__(self, data_dir, config_path):
        self.data_dir = data_dir

        with open(config_path) as f:
            self.config = yaml.safe_load(f)



    def get_pointcloud(self, sequence, frame_idx):
        """
        Load an open3d point cloud from file
        args:
            sequence : name of the sequence
            frame : frameIdx
        returns
            pointcloud : open3d.pointcloud object
        """
        dataset_key = self._find_dataset_from_sequence(sequence)
        sequences_dict = self.config["sequences"]
        start_frame = sequences_dict[dataset_key][sequence]["start"]
        end_frame = sequences_dict[dataset_key][sequence]["end"]

        if not frame_idx in range(start_frame, end_frame + 1):
            print("frame_idx {} is out of range [{}:{}]".format(frame_idx, start_frame, end_frame))
            return None

        ply_path = self.config["relative_paths"][dataset_key]
        ply_path = ply_path.format(sequence=sequence, frame_idx=frame_idx)
        ply_path = os.path.join(self.data_dir, ply_path)

        if not os.path.exists(ply_path):
            print("Path {} is not existant.".format(ply_path))
            return None


        point_cloud = o3d.io.read_point_cloud(ply_path)
        # Downcale for QA
        if dataset_key == "QA":
            downsampling_rates = {0: 1.0, 1: 2.0, 2: 4.0, 3: 8.0}
            factor = downsampling_rates[frame_idx]

            point_cloud = point_cloud.voxel_down_sample(factor)
            points = np.asarray(point_cloud.points)
            points /= factor
            print(np.max(points))
            point_cloud.points = o3d.utility.Vector3dVector(np.round(points))

        return point_cloud




    def view_pointcloud(self, sequence, frame):
        """
        Load an open3d point cloud from file
        args:
            sequence : name of the sequence
            frame : frameIdx
        returns
            pointcloud : open3d.pointcloud object
        """
        pointcloud = self.get_pointcloud(sequence, frame)
        o3d.visualization.draw_geometries([pointcloud])



    def view_sequence(self, sequence, target_frame_rate=1/10):
        """
        View the Point Cloud sequence in total.  
        args:
            sequence : str
                name of the sequence
            target_frame_rate : float (default: 0.1)
                maximum frame rate targeted during rendering

        returns
            pointcloud : open3d.pointcloud object
        """
        dataset_key = self._find_dataset_from_sequence(sequence)
        sequences_dict = self.config["sequences"]
        start_frame = sequences_dict[dataset_key][sequence]["start"]
        end_frame = sequences_dict[dataset_key][sequence]["end"]

        vis = o3d.visualization.Visualizer()
        vis.create_window(height=480, width=640)
        opt = vis.get_render_option()
        opt.point_size = 1.5

        t_end = time.time()
        for frame_idx in range(start_frame, end_frame+1):
            t_start = time.time()
            pointcloud = self.get_pointcloud(sequence, frame_idx)

            vis.clear_geometries()
            vis.add_geometry(pointcloud)
            vis.poll_events()
            vis.update_renderer()

            t_render = time.time() - t_start
            if t_render < target_frame_rate:
                time.sleep(target_frame_rate - t_render)
                t_render = time.time() - t_start
            print("{} : {:3.2f} fps".format(frame_idx, 1/t_render))



    def _find_dataset_from_sequence(self, sequence) -> str:
        """
        Finds the dataset of a given sequence name in the config
        args:
            sequence : str
                The name of the searched sequence
        returns:
            key : str
                The key, denoting the dataset it is in
        """
        all_sequences = self.config["sequences"]

        for key, sub_dict in all_sequences.items():
            if sequence in sub_dict.keys():
                return key

        print("Could not find key")
        return None



if __name__ == "__main__":
    print("Testing in utils")
    raw_loader = RawLoader("./data/raw", "./data/configs/raw_loading.yaml")
    pc = raw_loader.view_pointcloud("Gymnast", 100)
    pc = raw_loader.view_sequence("loot")