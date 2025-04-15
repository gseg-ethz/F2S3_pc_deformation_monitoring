"""
Deep learning based deformation monitoring of landslides and rockfalls.

This script is used to estimate the displacement vectors between point cloud of landslides or rockfalls acquired in two temporal epochs.
It combines the following steps:
1) voxel grid downsampling and tiling of the point clouds
2) local feature estimation based on a deep neural network
3) supervoxel computation
3) efficient correspondence search (approximate nearest neighbor) in high dimensional (64) feature space
4) filtering of the putative correspondences based on local consistency inside the estimated super voxels

The pipeline is described in more detail in [1].

[1] Gojcic, Z., et al.: F2S3: Robustified determination of 3D displacement vector fields using deep learning. Journal of Applied Geodesy, 2020.

Author: Zan Gojcic
"""

import os
from importlib import resources
import glob
import numpy as np
import logging
import coloredlogs
import open3d as o3d
import torch
import time
import hnswlib
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re

from plotly.graph_objs.icicle import Pathbar
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from .descriptor_model import PointNetFeature
from .filtering_model import FilteringNetwork
from .data import FeatureExtractionDataset
from .utils import transform_point_cloud, compute_c2c

from pc_tiling import pc_tiling
from supervoxel import supervoxel

@dataclass
class F2S3RunSettings:
    results_dir: Path
    source_cloud: Optional[Path] = None
    target_cloud: Optional[Path] = None
    start_from_tiled_data: bool = False
    tiled_data: Optional[Path] = None
    max_points_per_tile: int = 1000000
    batch_size: int = 2000
    voxel_grid_size: float = 0.0
    max_disp_magnitude: float = 0.0
    save_interim: bool = False
    verbose: bool = False
    filter_median_magnitude: bool = False
    refine_results: bool = False
    fill_gaps_c2c: bool = False

    def __post_init__(self):

        # Check correct path settings for input data
        if not self.start_from_tiled_data and (self.source_cloud is None or not self.source_cloud.is_file()):
            raise FileNotFoundError(f"Source cloud could not be found at {self.source_cloud}!")
        if not self.start_from_tiled_data and (self.target_cloud is None or not self.target_cloud.is_file()):
            raise FileNotFoundError(f"Target cloud could not be found at {self.target_cloud}!")

        if self.start_from_tiled_data and (self.tiled_data is None or not self.tiled_data.is_dir()):
            raise NotADirectoryError(f"Tiled data path incorrect: {self.tiled_data}!")

        # Check and set results path
        if self.results_dir == "":
            if self.start_from_tiled_data:
                self.results_dir = self.tiled_data.parent
            else:
                self.results_dir = self.source_cloud.parent
        else:
            self.results_dir = Path(self.results_dir)

        if not self.results_dir.exists():
            self.results_dir.mkdir(parents=True)

        # Set tiled data path in case not set (and not starting_from_tiled_data
        if self.tiled_data is None:
            self.tiled_data = self.results_dir / "tiled_data"

        if not self.tiled_data.exists():
            self.tiled_data.mkdir(parents=True)

        # Check validity of run arguments
        if self.max_points_per_tile < 1:
            raise ValueError("Maximum number of point per tile needs to be a positive integer!")
        if self.batch_size < 1:
            raise ValueError("Batch size needs to be a positive integer!")
        if self.voxel_grid_size < 0.0:
            raise ValueError("Voxel grid size can't be negative!")
        if self.max_disp_magnitude < 0.0:
            raise ValueError("Maximum displacement magnitude can't be negative!")


        # Prepare the logger
        logger = logging.getLogger()
        coloredlogs.install(level='INFO', logger=logger)
        log_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s - %(message)s')

        args_save_path = self.results_dir / 'command_line_args.txt'

        # Save arguments to the dictionary
        with open(args_save_path, 'w') as f:
            print(self, file=f)

    def __repr__(self):
        return '\n'.join([f'{field}: {getattr(self, field)}' for field in self.__dataclass_fields__])


class PointCloudTile:
    """
    Base point cloud tile class. It implements all the functions needed for the robust estimation of the displacement vectors.
    """

    def __init__(self, path_s: Path, path_t: Path, tile_id: str, feature_extractor, filtering_network, args: F2S3RunSettings):
        """
        Class constructor:

        Args:
            path_s (str): path to the source point cloud
            path_t (str): path to the target point cloud
            tile_id (str): number of the current point cloud tile
            feature_extractor (pytorch model): feature extractor model with loaded pretrained weights
            filtering_network (pytorch model): outlier filtering model with loaded pretrained weights
            args (dict): selected command line arguments

        """

        # Initialize the class settings and variables
        self.path_s = path_s
        self.path_t = path_t
        self.tile_id = tile_id

        self.pcd_s = o3d.io.read_point_cloud(str(self.path_s))
        self.pcd_t = o3d.io.read_point_cloud(str(self.path_t))

        self.pcd_s_overlap = o3d.io.read_point_cloud(str(
            self.path_s.parent / "overlap" / (self.path_s.stem + "_overlap.ply")
        ))
        self.pcd_t_overlap = o3d.io.read_point_cloud(str(
            self.path_t.parent / "overlap" / (self.path_t.stem + "_overlap.ply")
        ))

        # overlap_path = os.sep.join(self.path_s.split(os.sep)[:-1])
        # self.pcd_s_overlap = o3d.io.read_point_cloud(
        #     os.path.join(overlap_path, "overlap/source_tile_{}_overlap.ply".format(self.tile_id)))
        # self.pcd_t_overlap = o3d.io.read_point_cloud(
        #     os.path.join(overlap_path, "overlap/target_tile_{}_overlap.ply".format(self.tile_id)))

        self.feature_extractor = feature_extractor
        self.filtering_network = filtering_network

        self.refine_results = args.refine_results
        self.max_disp_magnitude = args.max_disp_magnitude
        self.filter_median_magnitude = args.filter_median_magnitude
        self.fill_gaps_c2c = args.fill_gaps_c2c
        self.verbose = args.verbose
        self.save_interim = args.save_interim
        self.base_save_path = args.results_dir
        self.voxel_grid_size = args.voxel_grid_size

        # Compute median resolution of the point clouds
        self.median_resolution = self._compute_resolution()

        # Numbers of points used to compute the normal vectors in the supervoxel computation
        self.n_normals = 30

    def compute_supervoxels(self):
        """
        Computes boundary perserving supervoxels (i.e. local patches of geometrically coherent points) based on the method proposed in [2].

        [2] Lin, Y., et al.: Toward better boundary preserved supervoxel segmentation for 3D point clouds. ISPRS journal of photogrammetry and remote sensing, 2018.
        """

        # Approximate supervoxel radius is defined for each tile independently based on the median point cloud resolution
        # I changes please consider that this has to be a reasonable value (i.e. size of the patch that moves as rigid body)

        supervoxel_radius = np.max((np.sqrt(3) * (10 * self.median_resolution), self.voxel_grid_size,))

        if self.verbose:
            logging.info('Starting the supervoxel ectraction.')
            logging.info('Supervoxel radius equals {:.2f} m.'.format(supervoxel_radius))

        start_time = time.time()

        # If the interim results should not be saved the path has to be set to "None"
        if not self.save_interim:
            supervoxel_save_path = "None"
        else:
            base_folder = self.base_save_path  / 'supervoxels'
            if not base_folder.exists():
                base_folder.mkdir()

            file_name = 'supervoxel_tile_{}_{:.1f}.txt'.format(self.tile_id, supervoxel_radius)
            supervoxel_save_path = base_folder / file_name

        supervoxel_idx = supervoxel.computeSupervoxel(str(self.path_s),
                                                      self.n_normals,
                                                      supervoxel_radius,
                                                      str(supervoxel_save_path))

        supervoxel_idx = np.asarray(supervoxel_idx).reshape(-1, 1)

        # Extract the indices of individual supervoxels
        supervoxels = np.unique(supervoxel_idx)

        self.supervoxels = []
        for idx in supervoxels:
            sv_idx = np.where(supervoxel_idx == idx)[0]
            if sv_idx.shape[0] > 10:
                self.supervoxels.append(sv_idx)

        end_time = time.time()

        # Only print out the required time if the verbose mode is selected
        if self.verbose:
            logging.info('Supervoxel extraction completed')
            logging.info('{} supervoxels computed in: {:.2f} s'.format(supervoxels.shape[0], end_time - start_time))

        gc.collect()

    def compute_local_features(self):
        """
        Computes local feature descriptors for each point in both source and target point cloud using DIP method proposed in [3]-

        [3] Poiesi, F., & Boscaini, D.: Distinctive 3D local deep descriptors. International Conference on Pattern Recognition (ICPR), 2020.
        """

        neighborhood_radius = np.sqrt(3) * (10 * self.median_resolution)

        if self.verbose:
            logging.info(
                f"Starting the computation of local feature descriptors. Neighborhood radius: {neighborhood_radius:.3f}")

        # Prepare source and target data loader.
        # If running into the GPU memory problems reduce the number of points in a batch (default is 2000).
        dataset_s = FeatureExtractionDataset(self.pcd_s, self.pcd_s_overlap, 1000, neighborhood_radius)
        dataset_t = FeatureExtractionDataset(self.pcd_t, self.pcd_t_overlap, 1000, neighborhood_radius)

        dataloader_s = torch.utils.data.DataLoader(dataset_s, batch_size=1, shuffle=False, num_workers=6,
                                                   drop_last=False)
        dataloader_t = torch.utils.data.DataLoader(dataset_t, batch_size=1, shuffle=False, num_workers=6,
                                                   drop_last=False)

        start_time = time.time()

        local_features_s = []
        local_features_t = []

        start_time = time.time()

        # If verbose mode is selected show the progress bar of the feature computation
        if self.verbose:
            dataloader_s = tqdm(dataloader_s, ncols=90)
        # Compute the features for the source point cloud
        for batch in dataloader_s:
            batch_feat = batch.squeeze(0).cuda()
            batch_feat, _, _ = self.feature_extractor(batch_feat)

            local_features_s.append(batch_feat)

        # Compute the features for the target point cloud
        if self.verbose:
            dataloader_t = tqdm(dataloader_t, ncols=90)

        for batch in dataloader_t:
            batch_feat = batch.squeeze(0).cuda()
            batch_feat, _, _ = self.feature_extractor(batch_feat)

            local_features_t.append(batch_feat)

        # Save the features for the subsequent correspondence estimation
        self.local_features_s = torch.cat(local_features_s, 0)
        self.local_features_t = torch.cat(local_features_t, 0)

        torch.cuda.empty_cache()
        gc.collect()
        end_time = time.time()

        if self.verbose:
            logging.info('Extraction of the local feature descriptors completed.')
            logging.info('Local features were computed in {} s'.format(end_time - start_time))

        # Save the feature descriptors if the interim save result mode is selected
        if self.save_interim:
            save_path = os.path.join(self.base_save_path, 'features')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.savez(os.path.join(save_path, 'features_tile_{}_{:.1f}.npz'.format(self.tile_id, np.sqrt(3) * (
                        10 * self.median_resolution))),
                     feat_s=self.local_features_s.cpu(), feat_t=self.local_features_t.cpu())

    def compute_correspondences(self):
        """
        Approximate correspondence search in the feature space using the hierarchical navigable small world graph method proposed in [4].

        [4] Malkov, Y. A. et al.: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI, 2018.
        """

        # Paramters defining how close to the exact NN search the approximate search is. Should only be changed if there is a very good reason!
        # Increasing this values "sharpens" the NN search but also results in the longer processing time. Current values result in above 99% NN recall.
        M = 12
        efC = 300
        efS = 300

        # Set the number of CPU threads
        num_threads = 16

        if self.verbose:
            logging.info('Starting the computation of the correspondences in feature space.')

        start_time = time.time()

        # Intitialize the library, specify the space, the type of the vector and add data points
        p = hnswlib.Index(space='l2', dim=64)  # possible options are l2, cosine or ip
        p.init_index(max_elements=self.local_features_t.shape[0], ef_construction=efC, M=M)
        p.set_ef(efS)
        p.set_num_threads(num_threads)
        p.add_items(self.local_features_t.cpu().numpy())

        # Query the elements for themselves and measure recall:
        start_time = time.time()
        labels, distances = p.knn_query(self.local_features_s.cpu().numpy(), k=1)
        end_time = time.time()

        # Save the correspondences for subsequent steps
        self.correspondences = np.concatenate(
            (np.array(self.pcd_s.points), np.array(self.pcd_t.points)[labels.reshape(-1), :]), axis=1)

        if self.verbose:
            logging.info(
                'Correspondence estimation in the feature space completed in {:.2f} s'.format(end_time - start_time))

        # Save the correspondences ad NX2 matrix if the interim save result mode is selected
        if self.save_interim:
            save_path = os.path.join(self.base_save_path, 'correspondences')

            if not os.path.exists(save_path):
                os.makedirs(save_path)

            np.savez(os.path.join(save_path, 'correspondences_tile_{}.npz'.format(self.tile_id)),
                     corr=self.correspondences)

        gc.collect()

    def filter_correspondences(self):

        data = {}

        start_time = time.time()
        inlier_idx = []
        save_coords = []

        if self.verbose:
            logging.info('Starting the outlier detection step')

        if self.verbose:
            supervoxel_iter = tqdm(self.supervoxels, ncols=90)
        else:
            supervoxel_iter = self.supervoxels

        for supervoxel in supervoxel_iter:
            supervoxel_data = self.correspondences[supervoxel, :]
            supervoxel_data_scaled = np.divide(supervoxel_data, np.max(np.abs(supervoxel_data)))

            filtering_output = self.filtering_network.filter_input(
                torch.from_numpy(supervoxel_data_scaled).cuda().unsqueeze(0).unsqueeze(0).float(),
                torch.from_numpy(supervoxel_data).cuda().unsqueeze(0).float())

            supervoxel_coords = supervoxel_data

            if filtering_output['robust_estimate'] and self.refine_results:
                supervoxel_coords[:, 3:6] = transform_point_cloud(
                    torch.from_numpy(supervoxel_data[:, 0:3]).cuda().float(),
                    filtering_output['rot_est'],
                    filtering_output['trans_est']).cpu().numpy()
                idx = np.ones(supervoxel_coords.shape[0])

            else:
                idx = (filtering_output['scores'].reshape(-1) > 0.99999).cpu().numpy()

            inlier_idx.append(idx)
            save_coords.append(supervoxel_data)

        torch.cuda.empty_cache()
        gc.collect()

        if inlier_idx:
            inlier_idx = np.concatenate(inlier_idx, axis=0)
            inlier_idx = np.where(inlier_idx > 0.5)[0].reshape(-1)

            save_coords = np.concatenate(save_coords, axis=0)

        # Filter the outliers based on the predicted scores
        filtered_results = save_coords[inlier_idx, :]
        filtered_magnitudes = np.linalg.norm(filtered_results[:, 3:6] - filtered_results[:, 0:3], axis=1)

        if self.verbose:
            logging.info(
                '{} points out of {} were classified as inlier'.format(filtered_results.shape[0], save_coords.shape[0]))

        save_path = os.path.join(self.base_save_path, 'output')

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        if self.refine_results:
            save_path_output = os.path.join(save_path, 'refined_results')
        else:
            save_path_output = os.path.join(save_path, 'results')

        if not os.path.exists(save_path_output):
            os.makedirs(save_path_output)

        np.savetxt(os.path.join(save_path_output, 'displacement_magnitude_tile_{}.txt'.format(self.tile_id)),
                   np.concatenate((filtered_results, filtered_magnitudes.reshape(-1, 1)), axis=1))

        # If maximum magnitude parameter is set, filter all points with larger magnitude estimates
        if self.max_disp_magnitude > 0:
            max_magnitude_idx = np.where(filtered_magnitudes < self.max_disp_magnitude)[0].reshape(-1)

            filtered_results = filtered_results[max_magnitude_idx, :]
            filtered_magnitudes = filtered_magnitudes[max_magnitude_idx]
            inlier_idx = inlier_idx[max_magnitude_idx].reshape(-1)

        # If filtered by magnitude is selected also filter very large motion inside a tile
        if self.filter_median_magnitude:
            if self.verbose:
                logging.info('Filtering the displacement vectors based on the mean magnitude of the displacement')

            # Compute the median magnitude
            median_mag = np.median(filtered_magnitudes)

            if self.verbose:
                logging.info('Median magnitude {}, all displacementes above {} m will be removed'.format(median_mag,
                                                                                                         30 * median_mag))

            mag_inlier = np.where(filtered_magnitudes < 30 * median_mag)[0]
            filtered_results = filtered_results[mag_inlier, :]
            filtered_magnitudes = filtered_magnitudes[mag_inlier]

            save_path_filtered_mag = os.path.join(save_path, 'filtered_by_magnitude')

            if not os.path.exists(save_path_filtered_mag):
                os.makedirs(save_path_filtered_mag)

            np.savetxt(os.path.join(save_path_filtered_mag, 'displacement_magnitude_tile_{}.txt'.format(self.tile_id)),
                       np.concatenate((filtered_results, filtered_magnitudes.reshape(-1, 1)), axis=1))

            # If selected combine the inliers estimated by out method with the C2C estimates for the outliers
            if self.fill_gaps_c2c:
                c2c_displacements = compute_c2c(save_coords[:, 0:3], np.asarray(self.pcd_t.points)).reshape(-1)

                save_path_c2c = os.path.join(save_path, 'combined_with_c2c')

                if not os.path.exists(save_path_c2c):
                    os.makedirs(save_path_c2c)

                c2c_displacements[inlier_idx[mag_inlier]] = filtered_magnitudes

                np.savetxt(os.path.join(save_path_c2c, 'displacement_magnitude_tile_{}.txt'.format(self.tile_id)),
                           np.concatenate((save_coords[:, 0:3], c2c_displacements.reshape(-1, 1)), axis=1))


        # If selected combine the inliers estimated by out method with the C2C estimates for the outliers
        elif self.fill_gaps_c2c:
            c2c_displacements = compute_c2c(save_coords[:, 0:3], np.asarray(self.pcd_t.points)).reshape(-1)

            save_path_c2c = os.path.join(save_path, 'combined_with_c2c')
            if not os.path.exists(save_path_c2c):
                os.makedirs(save_path_c2c)

            c2c_displacements[inlier_idx] = filtered_magnitudes

            np.savetxt(os.path.join(save_path_c2c, 'displacement_magnitude_tile_{}.txt'.format(self.tile_id)),
                       np.concatenate((save_coords[:, 0:3], c2c_displacements.reshape(-1, 1)), axis=1))

        end_time = time.time()
        logging.info('Outlier detection step completed in {:.2f} s'.format(end_time - start_time))

    def _compute_resolution(self):
        """
        Computes the median point cloud resolution of the tiles, where point cloud resolution is defined as the distance to the closes point
        in teh same point cloud.

        Returns_
            pc_resolution (float): min resolution across the source and target point cloud (median distance to the closet point)

        """

        start_time = time.time()
        # Compute the point cloud resolution of the source point cloud (k=2 as the closest point is the point itself)
        neigh = NearestNeighbors(n_neighbors=2, algorithm='kd_tree')
        neigh.fit(np.array(self.pcd_s.points))
        dist01, _ = neigh.kneighbors(np.array(self.pcd_s.points), return_distance=True)

        # Resolution of the source point cloud
        resolution_s = np.median(dist01[:, -1])

        # Compute the point cloud resolution of the target point cloud (k=2 as the closest point is the point itself)
        neigh.fit(np.array(self.pcd_t.points))
        dist01, _ = neigh.kneighbors(np.array(self.pcd_t.points), return_distance=True)

        # Resolution of the target point cloud
        resolution_t = np.median(dist01[:, -1])

        end_time = time.time()

        if self.verbose:
            logging.info(
                'Median point cloud resolution of the tiles computed in: {:.2f} s'.format(end_time - start_time))

        # Lower of the both point cloud resolutions (i.e. larger median distance to the closest points)

        pc_resolution = max(resolution_s, resolution_t)

        return pc_resolution


def feature_based_deformation_analysis(args: F2S3RunSettings):
    """
    Main function of this scripts. Starts by tiling the point clouds and then loops over the individual tiles and performs the deformation analysis.

    Args:
        args (F2S3RunSettings): command line parameters

    """

    start_time_whole_analysis = time.time()

    with resources.path('f2s3.pretrained_models.feature_descriptor', 'model_best.pth') as f_m:
        f_m = Path(f_m)
    with resources.path('f2s3.pretrained_models.outlier_filtering', 'model_best.pt') as o_m:
        o_m = Path(o_m)

    # Load the feature descriptor model
    feature_descriptor = PointNetFeature()
    feature_descriptor.load_state_dict(torch.load(f_m))
    feature_descriptor.cuda()
    feature_descriptor.eval()

    filtering_network = FilteringNetwork()
    filtering_network.load_state_dict(torch.load(o_m))
    filtering_network.cuda()
    filtering_network.eval()

    if not args.start_from_tiled_data:
        # Resave point clouds with PCL (this makes subsequent steps much faster)
        logging.info('Starting the resave and tiling of the original point clouds.')
        # pc_tiling.resave_point_cloud(args.source_cloud,
        #                              args.target_cloud,
        #                              args.verbose)

        # Tile the point cloud into smaller tiles that can be processed on a standalone computer
        pc_tiling.tile_point_clouds(str(args.source_cloud),
                                    str(args.target_cloud),
                                    str(args.tiled_data),
                                    args.max_points_per_tile,
                                    10000,
                                    bool(args.voxel_grid_size),
                                    args.voxel_grid_size,
                                    0.0,
                                    -1,
                                    args.verbose)

    # Loop over the tiles and perform the deformation analysis
    logging.info(f'Starting deformation analysis on tiles found at {args.tiled_data}.')

    tile_list = sorted(list(args.tiled_data.glob("source_tile_*")))
        # glob.glob(os.path.join(os.sep.join(args.source_cloud.split(os.sep)[:-2]), 'tiled_data/source_tile_*')))

    if args.verbose:
        logging.info(f'{len(tile_list)} tiles in the first epoch. Tiles with less than 5000 points will be removed.')

    for idx, tile_s in enumerate(tile_list):
        logging.info('----------------------------------------------------------------------')
        logging.info('Processing tile {}/{}'.format(idx + 1, len(tile_list)))

        tile_nr = tile_s.stem.split("_")[-1]

        tile_t = args.tiled_data / f"target_tile_{tile_nr}.ply"

        if tile_t.exists():
            deformation_data = PointCloudTile(tile_s, tile_t, tile_nr, feature_descriptor, filtering_network, args)

            start_time = time.time()
            deformation_data.compute_local_features()
            deformation_data.compute_supervoxels()
            deformation_data.compute_correspondences()
            deformation_data.filter_correspondences()

            end_time = time.time()
            logging.info('Whole processing of tile {} was finished in: {:.2f} s'.format(tile_nr, end_time - start_time))

        else:
            logging.warning('Target tile {} does not exist, skiping computation for tile {}!'.format(tile_t, tile_nr))

    logging.info('----------------------------------------------------------------------')
    end_time_whole_analysis = time.time()
    logging.info('Deformation analysis completed. {} point cloud tiles were analysed in : {:.2f} hours.'.format(
        len(tile_list), (end_time_whole_analysis - start_time_whole_analysis) / 3600))
    logging.info('----------------------------------------------------------------------')
