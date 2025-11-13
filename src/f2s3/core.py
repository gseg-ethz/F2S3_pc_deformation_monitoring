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


from importlib import resources
from tempfile import TemporaryDirectory
import numpy as np
import logging
import open3d as o3d
import torch
import time
import hnswlib
import gc
from pathlib import Path

from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

from torch.utils.data import DataLoader

from pchandler import PointCloudData
from pchandler.scalar_fields import ScalarFieldManager
from pchandler.scalar_fields.scalar_fields import ScalarFieldBoolean
from pchandler.data_io import ply, load_file
from pchandler.constants import COMMON_FIELD_BASES

from .config import F2S3Config
from .descriptor_model import PointNetFeature
from .filtering_model import FilteringNetwork
from .data import FeatureExtractionDataset
from .utils import transform_point_cloud, compute_c2c, get_original_point_indexes

from pc_tiling import pc_tiling
from supervoxel import supervoxel


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
                 args: F2S3Config):
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
        self.args = args
        self.path_s = Path(path_s)
        self.path_t = Path(path_t)
        self.tile_id = tile_id

        self.feature_extractor = feature_extractor
        self.filtering_network = filtering_network

        self.supervoxels: list = []
        self.correspondences: np.ndarray = np.empty(0)

        # Load the original source and target point clouds
        self.pcd_s = load_file(self.path_s)
        self.pcd_t = load_file(self.path_t)

        self.pcd_s_overlap = o3d.io.read_point_cloud(
            self.path_s.parent / "overlap" / (self.path_s.stem + "_overlap.ply")
        )
        self.pcd_t_overlap = o3d.io.read_point_cloud(
            self.path_t.parent / "overlap" / (self.path_t.stem + "_overlap.ply")
        )

        # Compute median resolution of the point clouds
        self.median_resolution = self._compute_resolution()


    def compute_supervoxels(self):
        """
        Computes boundary preserving supervoxels (i.e. local patches of geometrically coherent points) based on the method proposed in [2].

        [2] Lin, Y., et al.: Toward better boundary preserved supervoxel segmentation for 3D point clouds. ISPRS journal of photogrammetry and remote sensing, 2018.
        """
        start_time = time.time()
        self.supervoxels = []
        supervoxel_save_path = "None"

        # Approximate supervoxel radius is defined for each tile independently based on the median
        # point cloud resolution. When looking to change self.minimum_points, ("I"), please consider that this has
        # to be a reasonable value (i.e. number of points in the supervoxel that move as rigid body)
        radius = self.args.supervoxel_radius(self.median_resolution)

        logging.debug('Starting the supervoxel extraction.')
        logging.debug(f'Supervoxel radius equals {radius:.2f} m.')

        if self.args.save_interim:
            supervoxel_save_path = self.args.supervoxel_dir / f'supervoxel_tile_{self.tile_id}_{radius:.1f}.txt'

        supervoxel_idx = supervoxel.computeSupervoxel(input_file=str(self.path_s),
                                                      k_neighbors=self.args.n_normals,
                                                      resolution=radius,
                                                      save_file=str(supervoxel_save_path))

        supervoxel_idx = np.asarray(supervoxel_idx).reshape(-1, 1)

        # Extract the indices of individual supervoxels
        supervoxels = np.unique(supervoxel_idx)

        for voxel_id in supervoxels:
            matching_voxels_indexes = np.where(supervoxel_idx == voxel_id)[0]
            if matching_voxels_indexes.shape[0] > self.args.minimum_points:
                self.supervoxels.append(matching_voxels_indexes)

        # Only print out the required time if the verbose mode is selected
        logging.debug('Supervoxel extraction completed')
        logging.debug(f'{supervoxels.shape[0]} supervoxels computed in: {time.time() - start_time:.2f} s')

        gc.collect()

    def compute_local_features(self):
        """
        Computes local feature descriptors for each point in both source and target point cloud using DIP method proposed in [3]-

        [3] Poiesi, F., & Boscaini, D.: Distinctive 3D local deep descriptors. International Conference on Pattern Recognition (ICPR), 2020.
        """
        start_time = time.time()

        local_features_s = []
        local_features_t = []
        neighborhood_radius = self.args.feature_radius(self.median_resolution)

        logging.debug(f"Computing local feature descriptors with neighborhood radius: {neighborhood_radius:.3f} ...")

        # Prepare source and target data loader.
        # If running into the GPU memory problems reduce the number of points in a batch (default is 2000).
        dataset_s = FeatureExtractionDataset(self.pcd_s, self.pcd_s_overlap, self.args.batch_size, neighborhood_radius)
        dataset_t = FeatureExtractionDataset(self.pcd_t, self.pcd_t_overlap, self.args.batch_size, neighborhood_radius)

        n = self.args.num_workers
        dataloader_s = DataLoader(dataset_s, batch_size=1, shuffle=False, num_workers=n, drop_last=False)
        dataloader_t = DataLoader(dataset_t, batch_size=1, shuffle=False, num_workers=n, drop_last=False)

        # If verbose mode is selected, show the progress bar of the feature computation
        if self.args.verbose:
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
        if self.args.save_interim:
            save_path = (self.args.features_dir / f'features_tile_{self.tile_id}_{neighborhood_radius:.1f}.npz')
            np.savez(save_path, feat_s=self.local_features_s.cpu(), feat_t=self.local_features_t.cpu())

    def compute_correspondences(self):
        """
        Approximate correspondence search in the feature space using the hierarchical navigable small world graph method proposed in [4].

        [4] Malkov, Y. A. et al.: Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs. IEEE TPAMI, 2018.
        """

        # Paramters defining how close to the exact NN search the approximate search is. Should only be changed if there is a very good reason!
        # Increasing this values "sharpens" the NN search but also results in the longer processing time. Current values result in above 99% NN recall.
        M = self.args.correspondences.M
        efC = self.args.correspondences.efC
        efS = self.args.correspondences.efS
        space = self.args.correspondences.space
        dimensions = self.args.correspondences.dimensions

        # Set the number of CPU threads
        num_threads = self.args.correspondences.num_threads

        logging.debug('Starting the computation of the correspondences in feature space.')

        start_time = time.time()

        # Intitialize the library, specify the space, the type of the vector and add data points
        p = hnswlib.Index(space=space, dim=dimensions)  # possible options are l2, cosine or ip
        p.init_index(max_elements=self.local_features_t.shape[0], ef_construction=efC, M=M)
        p.set_ef(efS)
        p.set_num_threads(num_threads)
        p.add_items(self.local_features_t.cpu().numpy())

        # Query the elements for themselves and measure recall:
        labels, distances = p.knn_query(self.local_features_s.cpu().numpy(), k=1)

        # Save the correspondences for future steps
        self.correspondences = np.concatenate((
            np.array(self.pcd_s.xyz),
            np.array(self.pcd_t.xyz)[labels.reshape(-1), :]
        ), axis=1)

        logging.debug(f'Correspondence estimation in the feature space completed in {time.time() - start_time:.2f} s')

        # Save the correspondences and NX2 matrix if the interim save result mode is selected
        if self.args.save_interim:
            save_path = self.args.correspondences_dir / f'correspondences_tile_{self.tile_id}.npz'
            np.savez(save_path, corr=self.correspondences)

        gc.collect()

    def filter_correspondences(self):
        start_time = time.time()

        inlier_idx = []
        save_coords = []
        segment_ids = []
        supervoxel_iter = tqdm(self.supervoxels, ncols=90) if self.args.verbose else self.supervoxels

        logging.debug('Starting the outlier detection step')

        # Super voxels are a vector of indices
        for i, supervoxel in enumerate(supervoxel_iter):
            segment = np.full_like(supervoxel, i)

            # Correspondences is an Nx6 matrix pairing coordinates from the source (0:3) to the target (3:)
            supervoxel_data = self.correspondences[supervoxel, :]
            supervoxel_data_scaled = np.divide(supervoxel_data, np.max(np.abs(supervoxel_data)))

            filtering_output = self.filtering_network.filter_input(
                torch.from_numpy(supervoxel_data_scaled).cuda().unsqueeze(0).unsqueeze(0).float(),
                torch.from_numpy(supervoxel_data).cuda().unsqueeze(0).float())

            if filtering_output['robust_estimate'] and self.args.refine_results:
                # Transform the source coordinates to the target coordinate system using the estimated transformation
                supervoxel_data[:, 3:6] = transform_point_cloud(
                    torch.from_numpy(supervoxel_data[:, 0:3]).cuda().float(),
                    filtering_output['rot_est'],
                    filtering_output['trans_est']).cpu().numpy()
                idx = np.ones(supervoxel_data.shape[0])

            else:
                # Filter out supervoxel points that are not classified as inliers
                idx = (filtering_output['scores'].reshape(-1) > 0.99999).cpu().numpy()

            inlier_idx.append(idx)
            save_coords.append(supervoxel_data)
            segment_ids.append(segment)

        torch.cuda.empty_cache()
        gc.collect()

        if inlier_idx:
            inlier_idx = np.concatenate(inlier_idx, axis=0).astype(np.bool_)
            save_coords = np.concatenate(save_coords, axis=0)
            segment_ids = np.concatenate(segment_ids, axis=0, dtype=np.int32)

        # Filter the outliers based on the predicted scores
        filtered_results = save_coords[inlier_idx, :]

        tile_pcd = PointCloudData(xyz=filtered_results[:, 0:3])
        fields = tile_pcd.scalar_fields    # Reference to the scalar field manager for easier code management

        # Compute the displacement vector and magnitude of the displacement vector
        displacement_vector = filtered_results[:, 3:6] - filtered_results[:, 0:3]    # Target minus source
        filtered_magnitudes = np.linalg.norm(displacement_vector, axis=1)

        for i, c in enumerate('xyz', 0):    # Add the scalar fields
            fields.create_field(f'target_{c}', filtered_results[:, i+3])
            fields.create_field(f'deformation_n{c}', filtered_results[:, i])    # Vector from src to trg

        fields.create_field('magnitude', np.linalg.norm(displacement_vector, axis=1))
        fields.create_field('supervoxel_ids', segment_ids[inlier_idx])

        logging.debug(f'{filtered_results.shape[0]} points out of {save_coords.shape[0]} were classified as inlier')

        # Compute a max displacement filter
        self.apply_max_displacement_filter(tile_pcd, fields)

        # If filtered by magnitude is selected also filter very large motion inside a tile
        self.apply_median_magnitude_filter(fields)

        # If selected combine the inliers estimated by our method with the C2C estimates for the outliers
        self.fill_gaps_w_c2c(tile_pcd,
                             save_coords,
                             inlier_idx,
                             filtered_magnitudes,
                             fields)

        logging.info(f'Outlier detection step completed in {time.time() - start_time:.2f} s')

        if self.args.save_tiles:
            ply.PlyHandler.save(tile_pcd, self.args.output_tiles_dir / f"displacement_tile_{self.tile_id}.ply")

        return tile_pcd

    def apply_max_displacement_filter(self, tile_pcd: PointCloudData, fields: ScalarFieldManager):
        # If maximum magnitude parameter is set, filter all points with larger magnitude estimates
        if self.args.max_disp_magnitude > 0:
            max_magnitude_idx = fields['magnitude'] < self.args.max_disp_magnitude
            fields.add_field(ScalarFieldBoolean(max_magnitude_idx, name='max_displacement_filter'))

            logging.debug(f'Filtered {np.sum(~max_magnitude_idx)} points with a magnitude larger than {self.args.max_disp_magnitude}m')
            logging.debug(f'{len(tile_pcd) - np.sum(~max_magnitude_idx)} points remaining')

    def apply_median_magnitude_filter(self, fields: ScalarFieldManager):
        if self.args.filter_median_magnitude:
            logging.debug('Filtering the displacement vectors based on the mean magnitude of the displacement')

            median_magnitude = np.median(fields['magnitude'])
            threshold = float(median_magnitude) * self.args.magnitude_multiplier

            logging.debug(f'Median magnitude:{median_magnitude}, all displacements above {threshold}m will be removed')

            mag_inlier_sf = fields['magnitude'] < threshold
            fields.add_field(
                ScalarFieldBoolean(mag_inlier_sf, name='median_magnitude_inliers')
            )

    def fill_gaps_w_c2c(self,
                        tile_pcd: PointCloudData,
                        save_coords: np.ndarray,
                        inlier_idx: np.ndarray,
                        filtered_magnitudes: np.ndarray,
                        fields: ScalarFieldManager):
        if (self.args.fill_gaps_c2c
                and (self.args.filter_median_magnitude
                     or self.args.max_disp_magnitude > 0)):
            filter_index = np.ones(len(tile_pcd), dtype=np.bool_)

            if self.args.filter_median_magnitude:
                filter_index *= fields['median_magnitude_inliers']

            if self.args.max_disp_magnitude > 0:
                filter_index *= fields['max_displacement_filter']

            c2c_displacements = compute_c2c(
                save_coords[:, 0:3], np.asarray(self.pcd_t.xyz)
            ).reshape(-1)[inlier_idx]
            c2c_displacements[filter_index] = filtered_magnitudes[filter_index]
            fields.create_field(f'c2c_filled_gaps', c2c_displacements)

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

class F2S3:
    FEATURE_MODEL_NAME = 'f2s3.pretrained_models.feature_descriptor'
    FEATURE_MODEL_WEIGHTS = 'model_best.pth'

    OUTLIER_MODEL_NAME = 'f2s3.pretrained_models.outlier_filtering'
    OUTLIER_MODEL_WEIGHTS = 'model_best.pt'

    # TODO implement unpacking method
    def __init__(self, args: F2S3Config):
        self.args = args
        self.feature_descriptor = PointNetFeature()
        self.filtering_network = FilteringNetwork()

        self.load_feature_descriptor_weights()
        self.load_filter_network_weights()

        self.temp_dir = TemporaryDirectory()

        self.raw_path = None

        self.src_ply_path = None
        self.trg_ply_path = None

    def load_feature_descriptor_weights(self):
        with resources.path(F2S3.FEATURE_MODEL_NAME, F2S3.FEATURE_MODEL_WEIGHTS) as f_m:
            f_m = Path(f_m)

        self.feature_descriptor.load_state_dict(torch.load(f_m))
        self.feature_descriptor.cuda()
        self.feature_descriptor.eval()

    def load_filter_network_weights(self):
        with resources.path(F2S3.OUTLIER_MODEL_NAME, F2S3.OUTLIER_MODEL_WEIGHTS) as o_m:
            o_m = Path(o_m)

        self.filtering_network.load_state_dict(torch.load(o_m))
        self.filtering_network.cuda()
        self.filtering_network.eval()

    def compare_pcds(self, source: PointCloudData, target: PointCloudData) -> PointCloudData:
        self.src_ply_path = self.create_temp_files(source, 'source')
        self.trg_ply_path = self.create_temp_files(target, 'target')
        gc.collect()

        return self.feature_based_deformation_analysis()

    def compare_files(self, source: Path, target: Path) -> PointCloudData:
        self.args.source_cloud = source
        self.args.target_cloud = target

        return self.run()

    def run(self):
        self.src_ply_path = self.create_ply_file_copy(self.args.source_cloud)
        self.trg_ply_path = self.create_ply_file_copy(self.args.target_cloud)
        gc.collect()

        return self.feature_based_deformation_analysis()

    def create_input_pcd_tiles(self):
        if not self.args.start_from_tiled_data:
            self.args.tiled_data.mkdir(parents=True, exist_ok=True)
            # Resave point clouds with PCL (this makes subsequent steps much faster)
            logging.info('Starting the resave and tiling of the original point clouds.')
            # pc_tiling.resave_point_cloud(args.source_cloud,
            #                              args.target_cloud,
            #                              args.verbose)

            # Tile the point cloud into smaller tiles that can be processed on a standalone computer
            pc_tiling.tile_point_clouds(str(self.src_ply_path),
                                        str(self.trg_ply_path),
                                        str(self.args.tiled_data),
                                        self.args.max_points_per_tile,
                                        self.args.min_points_per_tile,
                                        bool(self.args.voxel_grid_size),
                                        self.args.voxel_grid_size,
                                        self.args.overlap_tiles,
                                        -1,
                                        self.args.verbose)

    def feature_based_deformation_analysis(self, config_name: str = "config") -> PointCloudData|None:
        """
        Main function of this scripts. Starts by tiling the point clouds and then loops over the individual tiles and performs the deformation analysis.
        """
        with torch.no_grad():
            self.args.base_dir.mkdir(parents=True, exist_ok=True)
            self.args.save_to_json(self.args.base_dir / f"{config_name}.json")
            start_time_whole_analysis = time.time()

            # Step 0 - Tile data to enable GPU processing
            self.create_input_pcd_tiles()
            tile_list = sorted(list(self.args.tiled_data.glob("source_tile_*")))

            # Loop over the tiles and perform the deformation analysis
            logging.info(f'Starting deformation analysis on tiles found at {self.args.tiled_data}.')
            logging.debug(f'{len(tile_list)} tiles in the first epoch. Tiles with less than 5000 points will be removed.')

            pcd_tiles: list[PointCloudData] = []
            tile_index_vector = []
            merged_pcd: PointCloudData | None = None

            for idx, tile_s in enumerate(tile_list):
                logging.info('----------------------------------------------------------------------')
                logging.info(f'Processing tile {idx + 1}/{len(tile_list)}')

                tile_name = tile_s.stem.split("_")[-1]
                tile_t = self.args.tiled_data / f"target_tile_{tile_name}.ply"

                if tile_t.exists():
                    # Step 1 - Get point cloud tile data
                    deformation_data = PointCloudTile(tile_s, tile_t, tile_name, self.feature_descriptor, self.filtering_network, self.args)

                    start_time = time.time()

                    # Step 2 - Compute local features
                    deformation_data.compute_local_features()

                    # Step 3 - Compute supervoxels
                    deformation_data.compute_supervoxels()

                    # Step 4 - Compute correspondences
                    deformation_data.compute_correspondences()

                    # Step 5 - Filter correspondences
                    pcd_tiles.append( deformation_data.filter_correspondences() )

                    # Tracks the tile index as a scalar field
                    tile_index_vector.append( np.full(len(pcd_tiles[-1]), idx) )

                    logging.info(f"Whole processing of tile {tile_name} was finished in: {time.time() - start_time:.2f}s")

                else:
                    logging.warning(f'Target tile {tile_t} does not exist, skipping computation for tile {tile_name}!')

            if len(pcd_tiles) > 0:
                tile_index_vector = np.concatenate(tile_index_vector, dtype=np.int32)
                merged_pcd = self.merge_tiles(pcd_tiles, tile_index_vector)

            logging.info(f"{'-'*40}\n"
                         f"Deformation analysis completed. {len(pcd_tiles)} point cloud tiles analysed in: "
                         f"{(time.time() - start_time_whole_analysis) / 3600:.2f} hours.\n"
                         f"{'-'*40}")

            self.temp_dir.cleanup()

            return merged_pcd

    def merge_tiles(self, pcds: list[PointCloudData], tile_indexes: np.ndarray):
        merged_pcd = PointCloudData.merge(*pcds)
        merged_pcd.scalar_fields.create_field("tile_idx", tile_indexes)

        pcd_source = load_file(self.args.source_cloud)
        indices = get_original_point_indexes(pcd_source, merged_pcd)

        for name, value in pcd_source.scalar_fields.items():
            # TODO check the reason because of duplicates - overlap regions?
            if name in COMMON_FIELD_BASES:
                if value.ndim == 2:
                    setattr(merged_pcd, name, value.arr[indices, :])
                else:
                    setattr(merged_pcd, name, value.arr[indices])
            else:
                merged_pcd.scalar_fields.create_field(name, value.arr[indices])

        ply.PlyHandler.save(merged_pcd, self.args.result_dir / "deformation_result.ply")

        return merged_pcd

    def create_ply_file_copy(self, pcd_path: Path) -> Path:
        """Creates a copy of the point cloud as PLY due to limitations of the PCL library."""
        if not pcd_path.suffix == '.ply':
            logging.debug(f"Creating a copy of {pcd_path} point cloud as PLY")

            pcd = load_file(pcd_path)
            temp_path = Path(self.temp_dir.name) / f"temp_{pcd_path.stem}.ply"
            self._save_xyz_ply(pcd, temp_path)

            return temp_path
        return pcd_path

    @staticmethod
    def _save_xyz_ply(pcd: PointCloudData, path: Path):
        pcd.scalar_fields = ScalarFieldManager()
        ply.PlyHandler.save(pcd, path)

    def create_temp_files(self, pcd: PointCloudData, name: str):
        """Create temporary files for the point cloud tiling."""
        temp_path = Path(self.temp_dir.name) / f"temp_{name}.ply"
        self._save_xyz_ply(pcd, temp_path)
        return temp_path


