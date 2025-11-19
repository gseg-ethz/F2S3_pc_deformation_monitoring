import argparse
from pathlib import Path

import torch

from f2s3.core import F2S3
from f2s3.config import F2S3Config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-s', '--source_cloud', type=Path,
        help='Path to the source point cloud (ignored if ``--start_from_tiled_data``).'
    )

    parser.add_argument(
        '-t', '--target_cloud', type=Path,
        help='Path to the target point cloud (ignored if ``--start_from_tiled_data``).'
    )

    parser.add_argument(
        "--start_from_tiled_data", action="store_true",
        help="Start processing from pre-tiled data (needs to be done with ``pc_tiling``)")


    parser.add_argument(
        "--tiled_data", type=Path,
        help="Used to store (and retrieve in case of ``--start_from_tiled_data``) tiled data. If not set, then "
             "``{results_dir}/tiled_data`` (in case of ``--start_from_tiled_data`` the path has to be explicitly set)."
    )

    parser.add_argument(
        '-r', '--results_dir', type=str, default="",
        help='Path to the root results directory'
    )

    parser.add_argument(
        '--max_points_per_tile', type=int, default=argparse.SUPPRESS,
        help='Maximum number of points per each tile. Should be decreased if the program runs OOM.'
    )

    parser.add_argument(
        '--min_points_per_tile', type=int, default=argparse.SUPPRESS,
        help='Minimum number of points per each tile.'
    )

    parser.add_argument(
        '--batch_size', type=int, default=argparse.SUPPRESS,
        help='Batch size used in neural networks. Should be adapted depending on available virtual memory.'
    )

    parser.add_argument(
        '--voxel_grid_size', type=float, default=argparse.SUPPRESS,
        help='Target spacing for for voxel grid filter. Used to make the point density more uniform (No Filter if 0)'
    )

    parser.add_argument(
        '--max_disp_magnitude', type=float, default=argparse.SUPPRESS,
        help='Maximum displacement magnitude, points with higher magnitude will be filtered out (no filtering if 0)'
    )


    parser.add_argument(
        '--save_interim', action='store_true',
        help='Save interim results such as pointwise feature desriptors and supervoxels.'
    )

    parser.add_argument(
        '--save_tiles', action='store_true',
        help='Save also the tiled point cloud results.'
    )

    parser.add_argument(
        '--verbose', action='store_true',
        help='Print out more details about the process to the command line'
    )

    parser.add_argument(
        '--filter_median_magnitude', action='store_true',
        help='filter the remaining outliers by 30x median magnitude (improves visualizations where large displacements do not dominate)'
    )

    parser.add_argument(
        '--apply_filter', action='store_true',
        help='Applies the filters to the point clouds'
    )

    parser.add_argument(
        '--refine_results', action='store_true',
        help='refine correspondence based displacement vectors with the estimate transformation parameters'
    )

    parser.add_argument(
        '--fill_gaps_c2c', action='store_true',
        help='estimate the displacement with c2c for the points that were filtered by the outlier filtering algorithm'
    )

    parser.add_argument(
        '--num_workers', type=int, default=argparse.SUPPRESS,
        help='Number of workers for the dataloader'
        )

    parser.add_argument(
        '--magnitude_multiplier', type=float, default=argparse.SUPPRESS,
        help='Multiplier for the median magnitude filter'
        )

    parser.add_argument(
        '--n_normals', type=int, default=argparse.SUPPRESS,
        help='Number of points used to compute normal vectors in the supervoxels'
        )

    parser.add_argument(
        '--minimum_points', type=int, default=argparse.SUPPRESS,
        help='Number of points used to compute normal vectors in the supervoxels'
        )

    args = parser.parse_args()

    run_settings = F2S3Config(**vars(args))

    with torch.no_grad():
        obj = F2S3(run_settings)
        obj.run()


if __name__ == '__main__':
    # main()
    cfg = F2S3Config(
        results_dir=Path("/home/jonal/projects/F2S3/data/pchandler_base"),
        source=Path("/home/jonal/projects/F2S3/data/Mattertal/2019_Ground_aligned_clipped.laz"),
        target=Path("/home/jonal/projects/F2S3/data/Mattertal/2021_Ground_aligned_clipped.laz"),
        start_from_tiled_data=False
    )
    cfg.refine_results = True
    cfg.save_interim = True
    cfg.save_tiles = True
    cfg.filter_median_magnitude = True
    cfg.max_disp_magnitude = 1.0
    cfg.fill_gaps_c2c = True
    cfg.verbose = True

    algorithm = F2S3(cfg)
    algorithm.run()


