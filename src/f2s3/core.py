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
from scipy.spatial import KDTree

from torch.utils.data import DataLoader

from pchandler import PointCloudData
from pchandler.scalar_fields import ScalarFieldManager
from pchandler.data_io.core import SUPPORTED_TYPES as SUPPORTED_FILE_TYPES
from pchandler.data_io import las, csv, e57, ply

from .descriptor_model import PointNetFeature
from .filtering_model import FilteringNetwork
from .data import FeatureExtractionDataset
from .utils import transform_point_cloud, compute_c2c

from pc_tiling import pc_tiling
from supervoxel import supervoxel


def load_file(file_path: Path) -> tuple[PointCloudData, bool]:
    ply_flag = False
    data_loaders = {
        '.las': las.LasHandler.load,
        '.laz': las.LasHandler.load,
        '.txt': csv.CsvHandler.load,
        '.asc': csv.CsvHandler.load,
        '.csv': csv.CsvHandler.load,
        '.pts': csv.CsvHandler.load,
        '.e57': e57.E57Handler.load,
        '.ply': ply.PlyHandler.load,
    }

    if not file_path.is_file():
        raise FileNotFoundError(f"File {file_path} not found")

    if file_path.suffix not in SUPPORTED_FILE_TYPES:
        raise ValueError(f"File suffix {file_path.suffix} is not supported. It should be in {SUPPORTED_FILE_TYPES}")

    if file_path.suffix != '.ply':
        ply_flag = False

    return data_loaders[file_path.suffix](file_path), ply_flag

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
        coloredlogs.install(level='INFO' if self.verbose else 'VERBOSE', logger=logger)
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
    def __init__(self,
                 path_s: Path,
                 path_t: Path,
                 tile_id: str,
                 feature_extractor,
                 filtering_network,
                 args: F2S3RunSettings):
        """
        Class constructor:

        Args:
            path_s (str): path to the source point cloud
            path_t (str): path to the target point cloud
            tile_id (str): number of the current point cloud tile
            feature_extractor (torch.Model): feature extractor model with loaded pretrained weights
            filtering_network (torch.Model): outlier filtering model with loaded pretrained weights
            args (dict): selected command line arguments

        """

        # Initialize the class settings and variables
        self.path_s = Path(path_s)
        self.path_t = Path(path_t)
        self.tile_id = tile_id

        self.pcd_s, flag_s_ply = load_file(self.path_s)
        self.pcd_t, flag_t_ply = load_file(self.path_t)


        self.pcd_s_overlap = o3d.io.read_point_cloud(
            self.path_s.parent / "overlap" / (self.path_s.stem + "_overlap.ply")
        )
        self.pcd_t_overlap = o3d.io.read_point_cloud(
            self.path_t.parent / "overlap" / (self.path_t.stem + "_overlap.ply")
        )

        self.feature_extractor = feature_extractor
        self.filtering_network = filtering_network

        self.refine_results: bool = args.refine_results
        self.max_disp_magnitude: float = args.max_disp_magnitude
        self.filter_median_magnitude: bool = args.filter_median_magnitude
        self.fill_gaps_c2c: bool = args.fill_gaps_c2c
        self.verbose: bool = args.verbose
        self.save_interim: bool = args.save_interim
        self.base_save_path: Path = Path(args.results_dir)
        self.voxel_grid_size: float = args.voxel_grid_size

        # Compute median resolution of the point clouds
        self.median_resolution = self._compute_resolution()

        # Numbers of points used to compute the normal vectors in the supervoxel computation
        self.n_normals: int = 30

    def compute_supervoxels(self):
        """
        Computes boundary perserving supervoxels (i.e. local patches of geometrically coherent points) based on the method proposed in [2].

        [2] Lin, Y., et al.: Toward better boundary preserved supervoxel segmentation for 3D point clouds. ISPRS journal of photogrammetry and remote sensing, 2018.
        """

        # Approximate supervoxel radius is defined for each tile independently based on the median point cloud resolution
        # I changes please consider that this has to be a reasonable value (i.e. size of the patch that moves as rigid body)

        supervoxel_radius = np.max((np.sqrt(3) * (10 * self.median_resolution), self.voxel_grid_size,))

        logging.debug('Starting the supervoxel extraction.')
        logging.debug(f'Supervoxel radius equals {supervoxel_radius:.2f} m.')

        start_time = time.time()

        # If the interim results should not be saved the path has to be set to "None"
        if not self.save_interim:
            supervoxel_save_path = "None"
        else:
            base_folder = self.base_save_path  / 'supervoxels'
            if not base_folder.exists():
                base_folder.mkdir()

            file_name = f'supervoxel_tile_{self.tile_id}_{supervoxel_radius:.1f}.txt'
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

        # Only print out the required time if the verbose mode is selected
        logging.debug('Supervoxel extraction completed')
        logging.debug(f'{supervoxels.shape[0]} supervoxels computed in: {time.time() - start_time:.2f} s')

        gc.collect()

    def compute_local_features(self):
        """
        Computes local feature descriptors for each point in both source and target point cloud using DIP method proposed in [3]-

        [3] Poiesi, F., & Boscaini, D.: Distinctive 3D local deep descriptors. International Conference on Pattern Recognition (ICPR), 2020.
        """

        neighborhood_radius = np.sqrt(3) * (10 * self.median_resolution)

        logging.debug(
            f"Starting the computation of local feature descriptors. Neighborhood radius: {neighborhood_radius:.3f}"
        )

        # Prepare source and target data loader.
        # If running into the GPU memory problems reduce the number of points in a batch (default is 2000).
        dataset_s = FeatureExtractionDataset(self.pcd_s, self.pcd_s_overlap, 1000, neighborhood_radius)
        dataset_t = FeatureExtractionDataset(self.pcd_t, self.pcd_t_overlap, 1000, neighborhood_radius)

        dataloader_s = DataLoader(dataset_s, batch_size=1, shuffle=False, num_workers=6, drop_last=False)
        dataloader_t = DataLoader(dataset_t, batch_size=1, shuffle=False, num_workers=6, drop_last=False)

        local_features_s = []
        local_features_t = []

        start_time = time.time()

        # If verbose mode is selected, show the progress bar of the feature computation
        if self.verbose:
            dataloader_s = tqdm(dataloader_s, ncols=90)
            dataloader_t = tqdm(dataloader_t, ncols=90)

        # Compute the features for the source point cloud
        for batch in dataloader_s:
            batch_feat = batch.squeeze(0).cuda()
            batch_feat, _, _ = self.feature_extractor(batch_feat)
            local_features_s.append(batch_feat)

        # Compute the features for the target point cloud
        for batch in dataloader_t:
            batch_feat = batch.squeeze(0).cuda()
            batch_feat, _, _ = self.feature_extractor(batch_feat)
            local_features_t.append(batch_feat)

        # Save the features for subsequent correspondence estimation
        self.local_features_s = torch.cat(local_features_s, 0)
        self.local_features_t = torch.cat(local_features_t, 0)

        torch.cuda.empty_cache()
        gc.collect()

        logging.debug('Extraction of the local feature descriptors completed.')
        logging.debug(f'Local features were computed in {time.time() - start_time} s')

        # Save the feature descriptors if the interim save result mode is selected
        if self.save_interim:
            save_path = self.base_save_path / 'features'

            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)

            np.savez(
                save_path / f'features_tile_{self.tile_id}_{np.sqrt(3) * (10 * self.median_resolution):.1f}.npz',
                feat_s=self.local_features_s.cpu(),
                feat_t=self.local_features_t.cpu()
            )

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

        logging.debug('Starting the computation of the correspondences in feature space.')

        start_time = time.time()

        # Intitialize the library, specify the space, the type of the vector and add data points
        p = hnswlib.Index(space='l2', dim=64)  # possible options are l2, cosine or ip
        p.init_index(max_elements=self.local_features_t.shape[0], ef_construction=efC, M=M)
        p.set_ef(efS)
        p.set_num_threads(num_threads)
        p.add_items(self.local_features_t.cpu().numpy())

        # Query the elements for themselves and measure recall:
        labels, distances = p.knn_query(self.local_features_s.cpu().numpy(), k=1)

        # Save the correspondences for subsequent steps
        self.correspondences = np.concatenate((
            np.array(self.pcd_s.xyz),
            np.array(self.pcd_t.xyz)[labels.reshape(-1), :]
        ), axis=1)

        logging.debug(f'Correspondence estimation in the feature space completed in {time.time() - start_time:.2f} s')

        # Save the correspondences ad NX2 matrix if the interim save result mode is selected
        if self.save_interim:
            save_path = self.base_save_path / 'correspondences'

            if not save_path.exists():
                save_path.mkdir(parents=True, exist_ok=True)

            np.savez(save_path / f'correspondences_tile_{self.tile_id}.npz', corr=self.correspondences)

        gc.collect()

    def filter_correspondences(self):
        start_time = time.time()

        inlier_idx = []
        save_coords = []
        segment_ids = []

        logging.debug('Starting the outlier detection step')

        supervoxel_iter = tqdm(self.supervoxels, ncols=90) if self.verbose else self.supervoxels

        # Super voxels are a vector of indices
        for i, supervoxel in enumerate(supervoxel_iter):
            segment = np.full_like(supervoxel, i)

            # Correspondences is an Nx6 matrix pairing coordinates from the source (0:3) to the target (3:)
            supervoxel_data = self.correspondences[supervoxel, :]
            supervoxel_data_scaled = np.divide(supervoxel_data, np.max(np.abs(supervoxel_data)))

            filtering_output = self.filtering_network.filter_input(
                torch.from_numpy(supervoxel_data_scaled).cuda().unsqueeze(0).unsqueeze(0).float(),
                torch.from_numpy(supervoxel_data).cuda().unsqueeze(0).float())

            supervoxel_coords = supervoxel_data

            if filtering_output['robust_estimate'] and self.refine_results:
                # TODO - Discuss if this is even needed, it is not used anywhere
                # supervoxel_coords[:, 3:6] = transform_point_cloud(
                #     torch.from_numpy(supervoxel_data[:, 0:3]).cuda().float(),
                #     filtering_output['rot_est'],
                #     filtering_output['trans_est']).cpu().numpy()
                idx = np.ones(supervoxel_coords.shape[0])

            else:
                idx = (filtering_output['scores'].reshape(-1) > 0.99999).cpu().numpy()

            inlier_idx.append(idx)
            save_coords.append(supervoxel_data)
            segment_ids.append(segment)

        torch.cuda.empty_cache()
        gc.collect()

        if inlier_idx:
            inlier_idx = np.concatenate(inlier_idx, axis=0)
            inlier_idx = np.where(inlier_idx > 0.5)[0].reshape(-1)

            save_coords = np.concatenate(save_coords, axis=0)
            segment_ids = np.concatenate(segment_ids, axis=0, dtype=np.int32)


        # Filter the outliers based on the predicted scores
        filtered_results = save_coords[inlier_idx, :]

        tile_pcd = PointCloudData(xyz=filtered_results[:, 0:3])
        tile_pcd.scalar_fields.create_field('target_x', filtered_results[:, 3])
        tile_pcd.scalar_fields.create_field('target_y', filtered_results[:, 4])
        tile_pcd.scalar_fields.create_field('target_z', filtered_results[:, 5])

        deformation_vector = filtered_results[:, 3:6] - filtered_results[:, 0:3]    # Target minus source
        filtered_magnitudes = np.linalg.norm(deformation_vector, axis=1)

        tile_pcd.scalar_fields.create_field('deformation_nx', deformation_vector[:, 0])
        tile_pcd.scalar_fields.create_field('deformation_ny', deformation_vector[:, 1])
        tile_pcd.scalar_fields.create_field('deformation_nz', deformation_vector[:, 2])
        tile_pcd.scalar_fields.create_field('magnitude', np.linalg.norm(deformation_vector, axis=1))
        tile_pcd.scalar_fields.create_field('supervoxel_ids', segment_ids[inlier_idx])

        logging.debug(f'{filtered_results.shape[0]} points out of {save_coords.shape[0]} were classified as inlier')

        save_path: Path = self.base_save_path / 'output'

        if not save_path.is_dir():
            save_path.mkdir(parents=True, exist_ok=True)

        if self.refine_results:
            save_path_output = save_path / 'refined_results'
        else:
            save_path_output = save_path / 'results'

        if not save_path_output.is_dir():
            save_path_output.mkdir(parents=True, exist_ok=True)

        # If maximum magnitude parameter is set, filter all points with larger magnitude estimates
        if self.max_disp_magnitude > 0:
            max_magnitude_idx = np.where(tile_pcd.scalar_fields['magnitude'] < self.max_disp_magnitude)[0].reshape(-1)

            tile_pcd.reduce(max_magnitude_idx)
            inlier_idx = inlier_idx[max_magnitude_idx].reshape(-1)

        # If filtered by magnitude is selected also filter very large motion inside a tile
        if self.filter_median_magnitude:
            logging.debug('Filtering the displacement vectors based on the mean magnitude of the displacement')

            # Compute the median magnitude
            median_mag = float(np.median(tile_pcd.scalar_fields['magnitude']))

            logging.debug(f'Median magnitude {median_mag}, all displacements above {30 * median_mag} m will be removed')

            mag_inlier = np.where(tile_pcd.scalar_fields['magnitude'] < 30 * median_mag)[0]
            tile_pcd.reduce(mag_inlier)

            save_path_filtered_mag = save_path / 'filtered_by_magnitude'

            if not save_path_filtered_mag.exists():
                save_path_filtered_mag.mkdir(parents=True, exist_ok=True)

            # ply.PlyHandler.save(tile_pcd, save_path_filtered_mag / f"displacement_tile_{self.tile_id}.ply")

            np.savetxt(
                save_path_filtered_mag / f'displacement_median_magnitude_tile_{self.tile_id}.txt',
                np.concatenate((filtered_results, filtered_magnitudes.reshape(-1, 1)), axis=1)
            )

            # If selected combine the inliers estimated by out method with the C2C estimates for the outliers
            if self.fill_gaps_c2c:
                c2c_displacements = compute_c2c(save_coords[:, 0:3], np.asarray(self.pcd_t.points)).reshape(-1)

                save_path_c2c = save_path / 'combined_with_c2c'

                if not save_path_c2c.exists():
                    save_path_c2c.mkdir(parents=True, exist_ok=True)

                c2c_displacements[inlier_idx[mag_inlier]] = filtered_magnitudes

                np.savetxt(
                    save_path_c2c / f'displacement_magnitude_tile_{self.tile_id}.txt',
                    np.concatenate((save_coords[:, 0:3], c2c_displacements.reshape(-1, 1)), axis=1)
                )


        # If selected combine the inliers estimated by out method with the C2C estimates for the outliers
        elif self.fill_gaps_c2c:
            c2c_displacements = compute_c2c(save_coords[:, 0:3], np.asarray(self.pcd_t.points)).reshape(-1)

            save_path_c2c = save_path / 'combined_with_c2c'
            if not save_path_c2c.exists():
                save_path_c2c.mkdir(parents=True, exist_ok=True)

            c2c_displacements[inlier_idx] = filtered_magnitudes

            np.savetxt(
                save_path_c2c / f'displacement_magnitude_tile_{self.tile_id}.txt',
                np.concatenate((save_coords[:, 0:3], c2c_displacements.reshape(-1, 1)), axis=1)
            )

        logging.info(f'Outlier detection step completed in {time.time() - start_time:.2f} s')

        ply.PlyHandler.save(tile_pcd, save_path_output / f"displacement_tile_{self.tile_id}.ply")

        return tile_pcd

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
        neigh.fit(np.array(self.pcd_s.xyz))
        dist01, _ = neigh.kneighbors(np.array(self.pcd_s.xyz), return_distance=True)

        # Resolution of the source point cloud
        resolution_s = np.median(dist01[:, -1])

        # Compute the point cloud resolution of the target point cloud (k=2 as the closest point is the point itself)
        neigh.fit(np.array(self.pcd_t.xyz))
        dist01, _ = neigh.kneighbors(np.array(self.pcd_t.xyz), return_distance=True)

        # Resolution of the target point cloud
        resolution_t = np.median(dist01[:, -1])

        logging.debug(f'Median point cloud resolution of the tiles computed in: {time.time() - start_time:.2f} s')

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

    src_path = Path(args.source_cloud)
    trg_path = Path(args.target_cloud)

    # TODO use tempfile library
    if src_path.suffix != '.ply':
        logging.debug("Creating a copy of the original SOURCE point cloud as PLY")
        pcd_s, src_ply_flag = load_file(src_path)
        pcd_s.scalar_fields = ScalarFieldManager()
        temp_src_path = src_path.parent / f"temp_{src_path.stem}.ply"
        ply.PlyHandler.save(pcd_s, temp_src_path)
    else:
        src_ply_flag = True
        temp_src_path = ""

    if trg_path.suffix != '.ply':
        logging.debug("Creating a copy of the original TARGET point cloud as PLY")
        pcd_t, trg_ply_flag = load_file(trg_path)
        temp_trg_path = trg_path.parent / f"temp_{trg_path.stem}.ply"
        pcd_t.scalar_fields = ScalarFieldManager()
        ply.PlyHandler.save(pcd_t, temp_trg_path)
    else:
        trg_ply_flag = True
        temp_trg_path = ""

    # Tiles are always created as PLY - Non PLY files should be caught and converted here
    if not args.start_from_tiled_data:
        # Resave point clouds with PCL (this makes subsequent steps much faster)
        logging.info('Starting the resave and tiling of the original point clouds.')
        # pc_tiling.resave_point_cloud(args.source_cloud,
        #                              args.target_cloud,
        #                              args.verbose)

        # Tile the point cloud into smaller tiles that can be processed on a standalone computer
        pc_tiling.tile_point_clouds(str(src_path if src_ply_flag else temp_src_path),
                                    str(trg_path if trg_ply_flag else temp_trg_path),
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

    logging.debug(f'{len(tile_list)} tiles in the first epoch. Tiles with less than 5000 points will be removed.')

    pcd_tiles: list[PointCloudData] = []
    tile_index_vector = []

    for idx, tile_s in enumerate(tile_list):
        logging.info('----------------------------------------------------------------------')
        logging.info(f'Processing tile {idx + 1}/{len(tile_list)}')

        tile_nr = tile_s.stem.split("_")[-1]

        tile_t = args.tiled_data / f"target_tile_{tile_nr}.ply"

        if tile_t.exists():
            deformation_data = PointCloudTile(tile_s, tile_t, tile_nr, feature_descriptor, filtering_network, args)

            start_time = time.time()

            deformation_data.compute_local_features()

            if not src_ply_flag:
                deformation_data.path_s = temp_src_path

            deformation_data.compute_supervoxels()

            if not src_ply_flag:
                deformation_data.path_s = src_path

            deformation_data.compute_correspondences()

            pcd_tiles.append(deformation_data.filter_correspondences())
            tile_index_vector.append(np.full(len(pcd_tiles[-1]), idx))

            end_time = time.time()
            logging.info(f"Whole processing of tile {tile_nr} was finished in: {end_time - start_time:.2f}s")

        else:
            logging.warning(f'Target tile {tile_t} does not exist, skipping computation for tile {tile_nr}!')

    tile_index_vector = np.concatenate(tile_index_vector, dtype=np.int32)

    merged_pcd = PointCloudData.merge(*pcd_tiles)
    merged_pcd.scalar_fields.create_field("tile_idx", tile_index_vector)

    pcd_source = ply.PlyHandler.load(src_path)
    kdt = KDTree(pcd_source.xyz)
    _, indices = kdt.query(merged_pcd.xyz, k=1, distance_upper_bound=0.001)

    for name, value in pcd_source.scalar_fields.fields.items():
        merged_pcd.scalar_fields.create_field(name, value[indices])

    ply.PlyHandler.save(merged_pcd, deformation_data.base_save_path / "output" / "merged_tiles.ply")

    logging.info('----------------------------------------------------------------------')
    logging.info(f"Deformation analysis completed. {len(tile_list)} point cloud tiles were analysed in : "
                 f"{(time.time() - start_time_whole_analysis) / 3600:.2f} hours.")
    logging.info('----------------------------------------------------------------------')

    # Clear the temporary point cloud file after tiling
    if not trg_ply_flag:
        os.remove(str(temp_trg_path))
        logging.debug("Temporary TARGET point cloud removed.")

    if not src_ply_flag:
        os.remove(str(temp_src_path))
        logging.debug("Temporary SOURCE point cloud removed.")

    return merged_pcd
