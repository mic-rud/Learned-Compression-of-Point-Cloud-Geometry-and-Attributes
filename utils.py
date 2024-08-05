import os
import open3d as o3d
import torch
import subprocess
import MinkowskiEngine as ME 
import numpy as np
import time

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

def pc_metrics(reference, distorted, metric_path, data_path, resolution):
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
        Extra Settings for metrics

    returns
    -------
    pcqm: float
        PCQM value
    """
    data_path = os.path.join(data_path, "tmp")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Save reference if o3d
    if isinstance(reference, o3d.geometry.PointCloud):
        ref_path = os.path.join(data_path, "ref.ply") 
        save_ply(ref_path, reference)
    else:
        ref_path = os.path.join(reference)

    # Save distorted
    if isinstance(distorted, o3d.geometry.PointCloud):
        distorted_path = os.path.join(data_path, "distorted.ply") 
        save_ply(distorted_path, distorted)
    else:
        distorted_path = os.path.join(distorted)

    # Call PCQM
    command = [metric_path,
               '--uncompressedDataPath={}'.format(ref_path),
               '--reconstructedDataPath={}'.format(distorted_path),
               '--resolution={}'.format(resolution),
               '--frameCount=1'
                ]
    result = subprocess.run(command, stdout=subprocess.PIPE)


    # read output
    string = result.stdout
    lines = string.decode().split('\n')
    start = 0
    for i, line in enumerate(lines):
        if "infile1 (A)" in line:
            start = i
            break

    prefix = ["AB_", "BA_", "sym_"]
    metrics = {}
    for j in range(3):
        metrics[prefix[j] + "p2p_mse"] = float(lines[start+1].split(':')[-1].strip())
        metrics[prefix[j] + "p2p_psnr"] = float(lines[start+2].split(':')[-1].strip())

        # only symmetric
        if j == 2:
            metrics[prefix[j] + "d2_mse"] = float(lines[start+3].split(':')[-1].strip())
            metrics[prefix[j] + "d2_psnr"] = float(lines[start+4].split(':')[-1].strip())
            start += 2

        metrics[prefix[j] + "y_mse"] = float(lines[start+3].split(':')[-1].strip())
        metrics[prefix[j] + "u_mse"] = float(lines[start+4].split(':')[-1].strip())
        metrics[prefix[j] + "v_mse"] = float(lines[start+5].split(':')[-1].strip())
        metrics[prefix[j] + "y_psnr"] = float(lines[start+6].split(':')[-1].strip())
        metrics[prefix[j] + "u_psnr"] = float(lines[start+7].split(':')[-1].strip())
        metrics[prefix[j] + "v_psnr"] = float(lines[start+8].split(':')[-1].strip())

        # Compute YUV
        metrics[prefix[j] + "yuv_psnr"] = (1/8) * (6 * metrics[prefix[j] + "y_psnr"] + metrics[prefix[j] + "u_psnr"] + metrics[prefix[j] + "v_psnr"])
        metrics[prefix[j] + "yuv_mse"] = (1/8) * (6 * metrics[prefix[j] + "y_mse"] + metrics[prefix[j] + "u_mse"] + metrics[prefix[j] + "v_mse"])

        start+=9

    return metrics

def pcqm(reference, distorted, pcqm_path, data_path, settings=None):
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
    data_path = os.path.join(cwd, data_path, "tmp")
    if not os.path.exists(data_path):
        os.mkdir(data_path)

    # Save reference if o3d
    if isinstance(reference, o3d.geometry.PointCloud):
        ref_path = os.path.join(data_path, "ref.ply") 
        save_ply(ref_path, reference)
    else:
        ref_path = os.path.join(cwd, reference)

    # Save distorted
    if isinstance(distorted, o3d.geometry.PointCloud):
        distorted_path = os.path.join(data_path, "distorted.ply") 
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
        

def remove_gpcc_header(path, gpcc=True):
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
        if gpcc:
            if "green" in line:
                new_header.append(line.replace("green", "red"))
            elif "blue" in line:
                new_header.append(line.replace("blue", "green"))
            elif "red" in line:
                new_header.append(line.replace("red", "blue"))
            else:
                new_header.append(line)
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



def compress_model_ours(experiment, model, data, q_a, q_g, device, base_path):
    """
    Compress a point cloud using our model
    """
    points = data["src"]["points"].to(device, dtype=torch.float)
    colors = data["src"]["colors"].to(device, dtype=torch.float)
    source = torch.concat([points, colors], dim=2)[0]
    N = source.shape[0]

    # Bin path
    bin_path = os.path.join(base_path,
                            experiment,
                            "tmp")
    if not os.path.exists(bin_path): 
        os.mkdir(bin_path)
    bin_path = os.path.join(bin_path, "bitstream.bin")

    # Q Map
    if type(q_a) is float or type(q_a) is np.float64:  
        # Uniform map
        Q_map = ME.SparseTensor(coordinates=torch.cat([torch.zeros((N, 1), device=device), points[0]], dim=1), 
                                features=torch.cat([torch.ones((N,1), device=device) * q_g, torch.ones((N,1), device=device) * q_a], dim=1),
                                device=source.device)
    else: 
        feats = torch.cat([torch.tensor(q_g), torch.tensor(q_a)], dim=1).type(torch.float32)
        Q_map = ME.SparseTensor(coordinates=torch.cat([torch.zeros((N, 1), device=device), points[0]], dim=1), 
                                features=feats,
                                device=source.device)

    # Compression
    torch.cuda.synchronize()
    t0 = time.time()

    #strings, shapes, k, coordinates = model.compress(source, Q_map)
    model.compress(source, Q_map, path=bin_path)

    torch.cuda.synchronize()
    t_compress = time.time() - t0

    # Decompress all rates
    # Run decompression
    torch.cuda.synchronize()
    t0 = time.time()
    reconstruction = model.decompress(path=bin_path)
    #reconstruction = model.decompress(coordinates=coordinates, strings=strings, shape=shapes, k=k)
    torch.cuda.synchronize()
    t_decompress = time.time() - t0
                    
    # Rebuild point clouds
    source_pc = get_o3d_pointcloud(source)
    rec_pc = get_o3d_pointcloud(reconstruction)

    bpp = os.path.getsize(bin_path) * 8 / N

    return source_pc, rec_pc, bpp, t_compress, t_decompress



def compress_related(experiment, data, q_a, q_g, base_path):
    """
    Compress a point cloud using V-PCC/G-PCC
    """
    path = os.path.join(base_path,
                        experiment,
                        "tmp")
    if not os.path.exists(path):
        os.mkdir(path)
    # Directories
    src_dir = os.path.join(path, "points_enc.ply")
    rec_dir = os.path.join(path, "points_dec.ply")
    bin_dir = os.path.join(path, "points_enc.bin")

    N = data["src"]["points"].shape[1]
    sequence = data["cubes"][0]["sequence"][0]

    # Data processing
    dtype = o3d.core.float32
    c_dtype = o3d.core.uint8
    points = data["src"]["points"].to(dtype=torch.float)
    colors = torch.clamp(data["src"]["colors"].to(dtype=torch.float) * 255, 0, 255)
    p_tensor = o3d.core.Tensor(points.detach().cpu().numpy()[0, :, :], dtype=dtype)
    p_colors = o3d.core.Tensor(colors.detach().cpu().numpy()[0, :, :], dtype=c_dtype)
    source = o3d.t.geometry.PointCloud(p_tensor)
    source.point.colors = p_colors
    o3d.t.io.write_point_cloud(src_dir, source, write_ascii=True)

    if experiment == "G-PCC":
        # Compress the point cloud using G-PCC
        command = ['./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3',
                '--mode=0',
                '--trisoupNodeSizeLog2=0',
                '--mergeDuplicatedPoints=1',
                '--neighbourAvailBoundaryLog2=8',
                '--intra_pred_max_node_size_log2=6',
                '--positionQuantizationScale={}'.format(q_g),
                '--maxNumQtBtBeforeOt=4',
                '--minQtbtSizeLog2=0',
                '--planarEnabled=1',
                '--planarModeIdcmUse=0',
                '--convertPlyColourspace=1',

                '--transformType=0',
                '--qp={}'.format(q_a),
                '--qpChromaOffset=-2',
                '--bitdepth=8',
                '--attrScale=1',
                '--attrOffset=0',
                '--attribute=color',

                '--uncompressedDataPath={}'.format(src_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user)" in line:
                processing_time_line = line
        t_compress = float(processing_time_line.split()[-2])

        bpp = os.path.getsize(bin_dir) * 8 / N
        # Decode
        command = ['./dependencies/mpeg-pcc-tmc13/build/tmc3/tmc3',
                '--mode=1',
                '--convertPlyColourspace=1',
                '--outputBinaryPly=0',
                '--reconstructedDataPath={}'.format(rec_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user)" in line:
                processing_time_line = line
        t_decompress = float(processing_time_line.split()[-2])

        # Read ply (o3d struggles with GBR order)
        remove_gpcc_header(rec_dir, gpcc=True)
        rec_pc = o3d.io.read_point_cloud(rec_dir)
        colors = np.asarray(rec_pc.colors)
        colors = colors[:, [2,0,1]]
        rec_pc.colors=o3d.utility.Vector3dVector(colors)

        # Clean up
        os.remove(rec_dir)
        os.remove(src_dir)
        os.remove(bin_dir)
    else: 
        # Compress the point cloud using V-PCC
        occPrecision = 4 if q_g > 16 else 2
        command = ['./dependencies/mpeg-pcc-tmc2/bin/PccAppEncoder',
                '--configurationFolder=./dependencies/mpeg-pcc-tmc2/cfg/',
                '--config=./dependencies/mpeg-pcc-tmc2/cfg/common/ctc-common.cfg',
                '--config=./dependencies/mpeg-pcc-tmc2/cfg/condition/ctc-all-intra.cfg',
                '--config=./dependencies/mpeg-pcc-tmc2/cfg/sequence/{}_vox10.cfg'.format(sequence), # Overwrite per sequence later
                '--frameCount=1',
                '--geometryQP={}'.format(q_g),
                '--attributeQP={}'.format(q_a),
                '--occupancyPrecision={}'.format(occPrecision),
                '--compressedStreamPath={}'.format(bin_dir),
                '--uncompressedDataPath={}'.format(src_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user.self)" in line:
                processing_time_line = line
        t_compress = float(processing_time_line.split()[-2])

        bpp = os.path.getsize(bin_dir) * 8 / N
        # Decode
        command = ['./dependencies/mpeg-pcc-tmc2/bin/PccAppDecoder',
                '--inverseColorSpaceConversionConfig=./dependencies/mpeg-pcc-tmc2/cfg/hdrconvert/yuv420torgb444.cfg',
                '--reconstructedDataPath={}'.format(rec_dir),
                '--compressedStreamPath={}'.format(bin_dir)]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        out = result.stdout.decode()
        output_lines = out.split("\n")
        processing_time_line = None
        for line in output_lines:
            if "Processing time (user.self)" in line:
                processing_time_line = line
        t_decompress = float(processing_time_line.split()[-2])

        # Read ply (o3d struggles with GBR order)
        remove_gpcc_header(rec_dir, gpcc=False)
        rec_pc = o3d.io.read_point_cloud(rec_dir)
        colors = np.asarray(rec_pc.colors)
        rec_pc.colors=o3d.utility.Vector3dVector(colors)

    # Reconstruct source
    points = data["src"]["points"]
    colors = data["src"]["colors"]
    source = torch.concat([points, colors], dim=2)[0]
    source_pc = get_o3d_pointcloud(source)
    return source_pc, rec_pc, bpp, t_compress, t_decompress