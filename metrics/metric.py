import open3d as o3d
import numpy as np
import copy


class PointCloudMetric():
    """
    Wrapper to compute metric on o3d point clouds
    """
    def __init__(self, source, reconstruction, resolution=1023, drop_duplicates=True):
        self.report = {}
        self.resolution = resolution

        # Load plys/pcds
        source_pcd = self.load_ply(source)
        recons_pcd = self.load_ply(reconstruction)

        # Drop duplicates
        if drop_duplicates:
            source_pcd = source_pcd.remove_duplicated_points()
            recons_pcd = recons_pcd.remove_duplicated_points()

        # Source arrays
        self.source_points = np.asarray(source_pcd.points)
        self.source_colors = np.asarray(source_pcd.colors)
        if source_pcd.has_normals():
            self.source_normals = np.asarray(source_pcd.normals)

        # Recon arrays
        self.recons_points = np.asarray(recons_pcd.points)
        self.recons_colors = np.asarray(recons_pcd.colors)
        if recons_pcd.has_normals():
            self.recons_normals = np.asarray(recons_pcd.normals)

        # NN associations
        self.source_tree = o3d.geometry.KDTreeFlann(source_pcd)
        self.recons_tree = o3d.geometry.KDTreeFlann(recons_pcd)
        self.source_2_recons = np.asarray(
            [self.recons_tree.search_knn_vector_3d(self.source_points[i], 2)[1] for i in range(len(self.source_points))]
        ).T
        self.recons_2_source = np.asarray(
            [self.source_tree.search_knn_vector_3d(self.recons_points[i], 2)[1] for i in range(len(self.recons_points))]
        ).T


    def load_ply(self, input):
        if isinstance(input, o3d.geometry.PointCloud):
            pcd = copy.deepcopy(input)
        elif (input, str): 
            # Load source from file
            pcd = o3d.io.read_point_cloud(input)
        else:
            raise ValueError("Source: {} is not a valid input")

        return pcd


    def get_result(self):
        return self.report

    def compute_pointcloud_metrics(self, drop_duplicates=False):
        result = {}
        error_vectors = {}
        resultAB, error_vector_color_AB = self.compute_metrics(mirror=False, drop_duplicates=drop_duplicates)
        resultBA, error_vector_color_BA = self.compute_metrics(mirror=True, drop_duplicates=drop_duplicates)
        result.update(resultAB)
        result.update(resultBA)

        error_vectors["colorAB"] = error_vector_color_AB
        error_vectors["colorAB"] = error_vector_color_BA

        result["sym_mse"] = np.min([result["AB_mse"], result["BA_mse"]])
        result["sym_hausdorff"] = np.min([result["AB_hausdorff"], result["BA_hausdorff"]])
        result["sym_psnr_mse"] = np.min([result["AB_psnr_mse"], result["BA_psnr_mse"]])
        result["sym_psnr_hausdorff"] = np.min([result["AB_psnr_hausdorff"], result["BA_psnr_hausdorff"]])

        result["sym_y_mse"] = np.min([result["AB_y_mse"], result["BA_y_mse"]])
        result["sym_u_mse"] = np.min([result["AB_u_mse"], result["BA_u_mse"]])
        result["sym_v_mse"] = np.min([result["AB_v_mse"], result["BA_v_mse"]])
        result["sym_y_psnr"] = np.min([result["AB_y_psnr"], result["BA_y_psnr"]])
        result["sym_u_psnr"] = np.min([result["AB_u_psnr"], result["BA_u_psnr"]])
        result["sym_v_psnr"] = np.min([result["AB_v_psnr"], result["BA_v_psnr"]])


        return result, error_vectors


    def compute_metrics(self, mirror=False, drop_duplicates=False):
        result = {}

        if not mirror:
            # A to B
            a_points = self.source_points
            b_points = self.recons_points
            a_colors = self.source_colors
            b_colors = self.recons_colors
            a2b = self.source_2_recons
            tree = self.recons_tree
            prefix = "AB_"
        else:
            # B to A
            a_points = self.recons_points
            b_points = self.source_points
            a_colors = self.recons_colors
            b_colors = self.source_colors
            a2b = self.recons_2_source
            tree = self.source_tree
            prefix = "BA_"

        b_points_ordered = b_points[a2b[0]]
        b_colors_ordered = b_colors[a2b[0]]
        ## Geometry metrics
        l2_distance = ((a_points -b_points_ordered)**2).mean(axis=1)

        result[prefix + "mse"] = l2_distance.mean()
        result[prefix + "hausdorff"] = np.max(l2_distance)

        result[prefix + "psnr_mse"] = 10 * np.log10(self.resolution ** 2 / result[prefix + "mse"])
        result[prefix + "psnr_hausdorff"] = 10 * np.log10(self.resolution ** 2 / result[prefix + "hausdorff"])

        ## Color metrics
        # points that are near the orignal content
        if not drop_duplicates:
            b_points_ordered_2 = b_points[a2b[1]]
            next_l2_distance = ((a_points - b_points_ordered_2)**2).mean(axis=1)

            # Find points with multiple neighbors at same distance and add colors
            num_points = l2_distance.shape[0]
            duplicate_indices = np.zeros(num_points)
            l2_distance_color = np.zeros(a_colors.shape)
            for i in range(num_points):
                if abs(l2_distance[i] - next_l2_distance[i]) < 1e-8:
                    duplicate_indices[i] = 1
                    # Find the 30 nearest neighbor points in steps of 5
                    for max_search in range(5, 30, 5):
                        all_nns = tree.search_knn_vector_3d(a_points[i], max_search)[1]

                        distances = [((a_points[i] - b_points[nn])**2).mean() for nn in all_nns]
                        same_dist_indices = [nn for idx, nn in enumerate(all_nns) if distances[idx] < 1e-8]
                        if len(same_dist_indices) < max_search:
                            break

                    same_dist_indices = [nn for nn in all_nns if ((a_points[i] - b_points[nn])**2).mean() == l2_distance[i]]
                    for nn in same_dist_indices:
                        b_colors_ordered[i] += b_colors[nn]
                    b_colors_ordered[i] = b_colors_ordered[i] / (len(same_dist_indices) + 1)

        # Round to rgb and convert to yuv 
        a_colors_yuv = self.convert_rgb_to_yuv(np.clip(np.round(a_colors * 255.0) / 255.0, 0.0, 1.0))
        b_colors_yuv = self.convert_rgb_to_yuv(np.clip(np.round(b_colors_ordered * 255.0) / 255.0, 0.0, 1.0))

        # Compute the mse for yuv on points with no duplicates
        l2_distance_color = ((a_colors_yuv - b_colors_yuv)**2)
        error_vector_color = l2_distance_color.copy()
        l2_distance_color = l2_distance_color.mean(axis=0)
        #print(l2_distance_color.mean(axis=0))

        result[prefix + "y_mse"] = l2_distance_color[0]
        result[prefix + "u_mse"] = l2_distance_color[1]
        result[prefix + "v_mse"] = l2_distance_color[2]
        result[prefix + "yuv_mse"] = np.mean(l2_distance_color, axis=0)
        result[prefix + "y_psnr"] = 10 * np.log10(1/result[prefix + "y_mse"])
        result[prefix + "u_psnr"] = 10 * np.log10(1/result[prefix + "u_mse"])
        result[prefix + "v_psnr"] = 10 * np.log10(1/result[prefix + "v_mse"])
        result[prefix + "yuv_psnr"] = 10 * np.log10(1/result[prefix + "yuv_mse"])
        return result, error_vector_color
        



    def convert_rgb_to_yuv(self, rgb):
        # BT 709
        scale = rgb.max() <= 1.0

        if scale:
            rgb = (rgb * 255).astype(np.uint8)

        # Perform the color transformation
        yuv = np.empty_like(rgb, dtype=np.float32)
        yuv[..., 0] = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]  # Y
        yuv[..., 1] = -0.1146 * rgb[..., 0] - 0.3854 * rgb[..., 1] + 0.5 * rgb[..., 2]  # U
        yuv[..., 2] = 0.5 * rgb[..., 0] - 0.4542 * rgb[..., 1] - 0.0458 * rgb[..., 2]  # V

        if scale:
            yuv = yuv / 255.0
            yuv[..., 1] += 0.5
            yuv[..., 2] += 0.5

        return yuv

class PointCloudMetric2():
    """
    Wrapper to compute metric on o3d point clouds
    """
    def __init__(self, source, reconstruction, normalize=True, drop_duplicates=True):
        self.report = {}
        self.source_2_reconstruction = None
        self.reconstruction_2_source = None

        # Load source
        if isinstance(source, o3d.geometry.PointCloud):
            self.source = copy.deepcopy(source)
        elif (source, str): 
            # Load source from file
            self.source = o3d.io.read_point_cloud(source)
        else:
            raise ValueError("Source: {} is not a valid input")

        # Load reconstruction
        if isinstance(reconstruction, o3d.geometry.PointCloud):
            self.reconstruction = copy.deepcopy(reconstruction)
        elif (source, str): 
            # Load source from file
            self.reconstruction = o3d.io.read_point_cloud(reconstruction)
        else:
            raise ValueError("Source: {} is not a valid input")

        if drop_duplicates:
            self.source = self.source.remove_duplicated_points()
            self.reconstruction = self.reconstruction.remove_duplicated_points()
        # Normalize
        if normalize:
            source_points = np.asarray(self.source.points)
            reconstruction_points = np.asarray(self.reconstruction.points)
            min_vals = source_points.min(0)
            max_vals = source_points.max(0)

            ranges = max_vals - min_vals
            norm_points_source = (source_points - min_vals) / ranges
            norm_points_reconstruction = (reconstruction_points - min_vals) / ranges

            self.source.points = o3d.utility.Vector3dVector(norm_points_source)
            self.reconstruction.points = o3d.utility.Vector3dVector(norm_points_reconstruction)


    def get_result(self):
        return self.report


    def compute_geometry_metrics(self, peak_signal=None):
        """
        Compute the geometry metric's:
        - PSNR
        - 
        """
        # Compute kd_tree and NN indices?
        if not self.source_2_reconstruction:
            self.compute_nns()

        if not peak_signal:
            peak_signal = 3*np.max([np.max(self.source.points), np.max(self.reconstruction.points)])

        # Source to reconstruction
        source_points = np.asarray(self.source.points)
        reconstruction_points_ordered = np.asarray(self.reconstruction.points)[self.source_2_reconstruction[0]]

        L2_AB = ((source_points - reconstruction_points_ordered)**2).mean(axis=1)

        mse_AB = L2_AB.mean()
        hausdorff_AB = np.max(L2_AB)
        psnr_mse_AB = 10 * np.log10(peak_signal**2 / mse_AB)

        # Reconstruction to source
        reconstruction_points = np.asarray(self.reconstruction.points)
        source_points_ordered = np.asarray(self.source.points)[self.reconstruction_2_source[0]]

        L2_BA = ((reconstruction_points - source_points_ordered)**2).mean(axis=1)

        mse_BA = L2_BA.mean()
        hausdorff_BA = np.max(L2_BA)
        psnr_mse_BA = 10 * np.log10(peak_signal**2 / mse_BA)


        # Symmetric
        mse_sym = np.max([mse_AB, mse_BA])
        haussdorff_sym = np.max([hausdorff_AB, hausdorff_BA])
        psnr_sym = np.min([psnr_mse_BA, psnr_mse_AB])

        # Report
        self.report["mse_AB"] = mse_AB
        self.report["mse_BA"] = mse_BA
        self.report["mse_symmetric"] = mse_sym
        self.report["hausdorff_AB"] = hausdorff_AB
        self.report["hausdorff_BA"] = hausdorff_BA
        self.report["hausdorff_symmetric"] = haussdorff_sym
        self.report["psnr_mse_AB"] = psnr_mse_AB
        self.report["psnr_mse_BA"] = psnr_mse_BA
        self.report["psnr_mse_symmetric"] = psnr_sym

    def compute_color_psnr(self):
        source_colors = np.asarray(self.source.colors)
        reconstruction_colors = np.asarray(self.reconstruction.colors)

        source_colors_yuv = self.convert_rgb_to_yuv(source_colors)
        reconstruction_colors_yuv = self.convert_rgb_to_yuv(reconstruction_colors)

        reconstruction_colors_ordered = reconstruction_colors_yuv[self.source_2_reconstruction[0]]
        source_colors_ordered = source_colors_yuv[self.reconstruction_2_source[0]]

        L2_AB = ((source_colors_yuv - reconstruction_colors_ordered)**2).mean(axis=0)
        L2_BA = ((reconstruction_colors_yuv - source_colors_ordered)**2).mean(axis=0)

        Y_MSE_AB = L2_AB[0]
        Y_MSE_BA = L2_BA[0]
        Y_MSE_BA = L2_BA[0]
        U_MSE_AB = L2_AB[1]
        U_MSE_BA = L2_BA[1]
        U_MSE_BA = L2_BA[1]
        V_MSE_AB = L2_AB[2]
        V_MSE_BA = L2_BA[2]
        V_MSE_BA = L2_BA[2]

        Y_PSNR_AB = 10 * np.log10(1**2/Y_MSE_AB)
        Y_PSNR_BA = 10 * np.log10(1**2/Y_MSE_BA)
        Y_PSNR_sym = np.min([Y_PSNR_BA, Y_PSNR_AB])
        U_PSNR_AB = 10 * np.log10(1**2/U_MSE_AB)
        U_PSNR_BA = 10 * np.log10(1**2/U_MSE_BA)
        U_PSNR_sym = np.min([U_PSNR_BA, U_PSNR_AB])
        V_PSNR_AB = 10 * np.log10(1**2/V_MSE_AB)
        V_PSNR_BA = 10 * np.log10(1**2/V_MSE_BA)
        V_PSNR_sym = np.min([V_PSNR_BA, V_PSNR_AB])

        self.report["psnr_y_AB"] = Y_PSNR_AB
        self.report["psnr_y_BA"] = Y_PSNR_BA
        self.report["psnr_y_symmetric"] = Y_PSNR_sym
        self.report["psnr_u_AB"] = U_PSNR_AB
        self.report["psnr_u_BA"] = U_PSNR_BA
        self.report["psnr_u_symmetric"] = U_PSNR_sym
        self.report["psnr_v_AB"] = V_PSNR_AB
        self.report["psnr_v_BA"] = V_PSNR_BA
        self.report["psnr_v_symmetric"] = V_PSNR_sym


    def compute_nns(self):
        """
        Computes the nearest neighbors from source to reconstruction (and vice versa) based on kdTreeFlann
        """
        source_tree = o3d.geometry.KDTreeFlann(self.source)
        reconstruction_tree = o3d.geometry.KDTreeFlann(self.reconstruction)
        self.source_2_reconstruction = [reconstruction_tree.search_knn_vector_3d(np.asarray(self.source.points[i]), 5)[1]
                                        for i in range(len(self.source.points))]
        self.reconstruction_2_source = [source_tree.search_knn_vector_3d(np.asarray(self.reconstruction.points[i]), 5)[1] 
                                        for i in range(len(self.reconstruction.points))]

        self.source_2_reconstruction = np.asarray(self.source_2_reconstruction).T
        self.reconstruction_2_source = np.asarray(self.reconstruction_2_source).T


    def convert_rgb_to_yuv(self, rgb):
        # BT 709
        scale = rgb.max() > 1.0

        if scale:
            rgb = (rgb * 255).astype(np.uint8)

        # Perform the color transformation
        yuv = np.empty_like(rgb, dtype=np.float32)
        yuv[..., 0] = 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]  # Y
        yuv[..., 1] = -0.1146 * rgb[..., 0] - 0.3854 * rgb[..., 1] + 0.5 * rgb[..., 2]  # U
        yuv[..., 2] = 0.5 * rgb[..., 0] - 0.4542 * rgb[..., 1] - 0.0458 * rgb[..., 2]  # V

        if scale:
            yuv = yuv / 255.0
            yuv[..., 1] -= 0.5
            yuv[..., 2] -= 0.5

        return yuv