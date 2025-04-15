import argparse
from pathlib import Path

import torch

from .core import F2S3RunSettings, feature_based_deformation_analysis

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_cloud', type=Path,
                        help='Path to the source point cloud (ignored if ``--start_from_tiled_data``).')
    parser.add_argument('-t', '--target_cloud', type=Path,
                        help='Path to the target point cloud (ignored if ``--start_from_tiled_data``).')
    parser.add_argument("--start_from_tiled_data", action="store_true",
                        help="Start processing from pre-tiled data (needs to be done with ``pc_tiling``)")
    parser.add_argument("--tiled_data", type=Path,
                        help="Used to store (and retrieve in case of ``--start_from_tiled_data``) tiled data. "
                             "If not set, then ``{results_dir}/tiled_data`` (in case of ``--start_from_tiled_data`` "
                             "the path has to be explicitly set).")

    parser.add_argument('-r', '--results_dir', type=str, help='Path to the root results directory', default="")

    parser.add_argument('--max_points_per_tile', type=int, default=argparse.SUPPRESS,
        help='Maximum number of points per each tile. Should be decreased if the program runs OOM.')
    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
        help='Batch size used in neural networks. Should be adapted depending on available virtual memory.')

    parser.add_argument('--voxel_grid_size', type=float, default=argparse.SUPPRESS,
        help='Target spacing for for voxel grid filter. Used to make the point density more uniform (No Filter if 0)')
    parser.add_argument('--max_disp_magnitude', type=float, default=argparse.SUPPRESS,
        help='Maximum displacement magnitude, all points with higher magnitude will be filtered out (no filtering if 0)')
    parser.add_argument(
        '--save_interim', action='store_true',
        help='Save interim results such as pointwise feature desriptors and supervoxels.')
    parser.add_argument(
        '--verbose', action='store_true', help='Print out more details about the process to the command line')
    parser.add_argument(
        '--filter_median_magnitude',
        action='store_true',
        help='filter the remaining outliers by 30x median magnitude (improves visualizations where large displacements do not dominate)')
    parser.add_argument(
        '--refine_results',
        action='store_true',
        help='refine correspondence based displacement vectors with the estimate transformation parameters')
    parser.add_argument(
        '--fill_gaps_c2c',
        action='store_true',
        help='estimate the displacement with c2c for the points that were filtered by the outlier filtering algorithm')

    args = parser.parse_args()

    run_settings = F2S3RunSettings(**vars(args))



    with torch.no_grad():
        feature_based_deformation_analysis(run_settings)


if __name__ == '__main__':
    main()
