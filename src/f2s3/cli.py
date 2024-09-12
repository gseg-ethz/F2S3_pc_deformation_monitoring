import argparse
from pathlib import Path

from f2s3.core import F2S3RunSettings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--source_cloud', type=Path, help='path to the source point cloud', required=True)
    parser.add_argument('-t', '--target_cloud', type=Path, help='path to the target point cloud', required=True)
    parser.add_argument(
        '--voxel_grid_size', default=0.0, type=float,
        help='Target spacing for for voxel grid filter. Used to make the point density more uniform (No Filter if 0)')
    parser.add_argument(
        '--max_points_per_tile', default=1000000, type=int,
        help='Maximum number of points per each tile. Should be decreasead if the program runs OOM.')
    parser.add_argument(
        '--max_disp_magnitude', default=0, type=float,
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
        feature_based_deformation_analysis(args)


if __name__ == '__main__':
    main()
