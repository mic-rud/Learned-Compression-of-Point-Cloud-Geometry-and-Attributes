import os
import open3d as o3d
import torch
import subprocess
import MinkowskiEngine as ME 
import numpy as np

class AverageMeter(object):
    """
    Computes and stores the average and current value
    Copied from: https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def count_bits(strings):
    """
    Computes the bpp for a nested array of strings
    
    Parameters
    ----------
    strings: list
        Nested list of strings
    
    returns
    -------
    total_bits: int
        Total bits required to save the nested list
    """
    total_bits = 0
    for string in strings:
        if not isinstance(string, list):
            total_bits += len(string) * 8
        else:
            total_bits += count_bits(string)
        
    return total_bits



def get_o3d_pointcloud(pc):
    """
    Generates a o3d point cloud on cpu from a torch tensor.

    Parameters
    ----------
    pc: torch.tensor, Nx6 or Nx3
        Tensor representing the point cloud

    returns
    -------
    o3d_pc: o3d.geometry.PointCloud
        Point Cloud object in o3d
    """
    o3d_pc = o3d.geometry.PointCloud()
    
    o3d_pc.points = o3d.utility.Vector3dVector(pc[:, :3].cpu().numpy())
    o3d_pc.colors = o3d.utility.Vector3dVector(pc[:, 3:].cpu().numpy())

    return o3d_pc


def render_pointcloud(pc, path, point_size=1.0):
    """
    Render the point cloud from 6 views along x,y,z axis

    Parameters
    ----------
    pc: o3d.geometry.PointCloud
        Point Cloud to be rendered
    path: str
        Format String with a open key field for formatting
    """
    settings = {
        "front":  [[0, -1, 0], [0, 0, 1]],
        "back":   [[0, 1, 0], [0, 0, 1]],
        "left":   [[-1, 0, 0], [0, 0, 1]],
        "right":  [[1, 0, 0], [0, 0, 1]],
        "top":    [[0, 0, 1], [0, 1, 0]],
        "bottom": [[0, 0, -1], [0, 1, 0]]
    }

    # Path
    dir, file = os.path.split(path)
    if not os.path.exists(dir):
        os.mkdir(dir)

    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=True)
    vis.add_geometry(pc)
    # Adjust the point size
    render_options = vis.get_render_option()
    render_options.point_size = 1.5  * point_size # adjust the size as required

    for key, view in settings.items():
        # Get view control
        view_control = vis.get_view_control()

        # Fit the object into the scene
        view_control.set_front(view[0])
        view_control.set_up(view[1])
        view_control.set_zoom(0.8)

        vis.update_renderer()
        
        image_path = path.format(key)
        vis.capture_screen_image(image_path, do_render=True)

    vis.destroy_window()


def downsampled_coordinates(coordinates, factor, batched=False):
    """
    Compute the remaining coordinates after downsampling by a factor

    Parameters
    ----------
    coordinates: ME.SparseTensor
        Tensor containing the orignal coordinates
    factor: int
        Downsampling factor (mutliple of 2)

    returns
    -------
    coords: torch.tensor
        Unique coordinates of the tensor
    """
    # Handle torch tensors and ME.SparseTensors
    coords = coordinates if torch.is_tensor(coordinates) else coordinates.C

    if coords.shape[1] == 3:
        coords = torch.floor(coords / factor) * factor
    else:
        # Exclude batch id
        coords[:, 1:4] = torch.floor((coords[:, 1:4]) / factor) * factor

    coords = torch.unique(coords, dim=0) 
    return coords


def sort_tensor(sparse_tensor):
    """
    Sort the coordinates of a tensor

    Parameters
    ----------
    sparse_tensor: ME.SparseTensor
        Tensor containing the orignal coordinates

    returns
    -------
    sparse_tensor: ME.SparseTensor
        Tensor containing the sorted coordinates
    """
    # Sort the coordinates
    weights = torch.tensor([1e15, 1e10, 1e5, 1], dtype=torch.int64, device=sparse_tensor.device) 
    sortable_vals = (sparse_tensor.C * weights).sum(dim=1)
    sorted_coords_indices = sortable_vals.argsort()

    sparse_tensor = ME.SparseTensor(
        features=sparse_tensor.F[sorted_coords_indices],
        coordinates=sparse_tensor.C[sorted_coords_indices],
        tensor_stride=sparse_tensor.tensor_stride,
        device=sparse_tensor.device
    )
    return sparse_tensor



def sort_points(points):
    """
    Sort the coordinates of torch list sized Nx4

    Parameters
    ----------
    points: torch.tensor
        Tensor containing the orignal coordinates Nx4

    returns
    -------
    points: torch.tensor
        Tensor containing the sorted coordinates Nx4
    """
    # Sort the coordinates
    weights = torch.tensor([1e15, 1e10, 1e5, 1], dtype=torch.int64, device=points.device) 
    sortable_vals = (points * weights).sum(dim=1)
    sorted_coords_indices = sortable_vals.argsort()

    points = points[sorted_coords_indices]
    return points


def pcqm(reference, distorted, pcqm_path, settings=None):
    """
    Compute PCQM with 

    Parameters
    ----------
    reference: o3d.geometry.PointCloud | string
        Reference Point Cloud or path to it
    distorted: o3d.geometry.PointCloud
        Distorted Point Cloud
    pcqm_path: str
        Path to the PCQM binary
    settings: dictionary (default=None)
        Extra Settings for PCQM

    returns
    -------
    pcqm: float
        PCQM value
    """
    cwd = os.getcwd()
    pcqm_path = os.path.join(cwd, pcqm_path)
    os.chdir(pcqm_path)

    # Save reference if o3d
    if isinstance(reference, o3d.geometry.PointCloud):
        ref_path = os.path.join(pcqm_path, "ref.ply") 
        save_ply(ref_path, reference)
    else:
        ref_path = os.path.join(cwd, reference)

    # Save distorted
    if isinstance(distorted, o3d.geometry.PointCloud):
        distorted_path = os.path.join(pcqm_path, "distorted.ply") 
        save_ply(distorted_path, distorted)
    else:
        distorted_path = os.path.join(cwd, distorted)

    # Call PCQM
    command = [pcqm_path + "/PCQM", ref_path, distorted_path, "-fq", "-r 0.004", "-knn 20", "-rx 2.0"]
    result = subprocess.run(command, stdout=subprocess.PIPE)

    # read output
    string = result.stdout
    lines = string.decode().split('\n')
    penultimate_line = lines[-3]  # Get the penultimate line
    pcqm_value_str = penultimate_line.split(':')[-1].strip()  # Extract the value after ':'
    pcqm_value = float(pcqm_value_str)  # Convert the value to float

    os.chdir(cwd)

    return pcqm_value

def save_ply(path, ply):
    o3d.io.write_point_cloud(path, ply, write_ascii=True)

    with open(path, "r") as ply_file:
        lines = ply_file.readlines()

    header = []
    for i, line in enumerate(lines):
        header.append(line)
        if line.strip() == "end_header":
            break

    # Update the property data type from double to float
    new_header = []
    for line in header:
        if "property double" in line:
            new_header.append(line.replace("double", "float"))
    
        else:
            new_header.append(line)

    # Convert the data values from double to float
    data_lines = lines[i + 1:]
    data = np.genfromtxt(data_lines, dtype=np.int32)

    # Save the modified PLY file
    with open(path, "w") as ply_file:
        for line in new_header:
            ply_file.write(line)
        for row in data:
            ply_file.write(" ".join(map(str, row)) + "\n")
        

def remove_gpcc_header(path):
    with open(path, "r") as ply_file:
        lines = ply_file.readlines()

    header = []
    for i, line in enumerate(lines):
        header.append(line)
        if line.strip() == "end_header":
            break

    # Update the property data type from double to float
    new_header = []
    for line in header:
        if "face" in line or "list" in line:
            continue
        elif "green" in line:
            new_header.append(line.replace("green", "red"))
        elif "blue" in line:
            new_header.append(line.replace("blue", "green"))
        elif "red" in line:
            new_header.append(line.replace("red", "blue"))
        else:
            new_header.append(line)

    # Convert the data values from double to float
    data_lines = lines[i + 1:]
    
    # Save the modified PLY file
    with open(path, "w") as ply_file:
        for line in new_header:
            ply_file.write(line)
        for row in data_lines:
            ply_file.write(row)