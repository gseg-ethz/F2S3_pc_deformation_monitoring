"""
Initial check for F2S3.

Downloads two epochs of point cloud data from the public zhaoyiww/Rockfall_Simulator HuggingFace dataset and runs the F2S3 pipeline,
"""

import sys
import glob
import tempfile
import logging
import coloredlogs
from pathlib import Path


def main() -> None:
    logger = logging.getLogger()
    coloredlogs.install(level='INFO', logger=logger)

    try:
        import torch
    except ImportError:
        logging.error("Could not import torch... ")
        sys.exit(1)

    if not torch.cuda.is_available():
        logging.error(
            "torch.cuda.is_available() returned False. F2S3 requires an GPU."
        )
        sys.exit(1)

    logging.info(f"Device: {torch.cuda.get_device_name(0)}")
    logging.info('Downloading sample data from HuggingFace')

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        logging.error("Could not import huggingface_hub")
        sys.exit(1)

    repo_id = "zhaoyiww/Rockfall_Simulator"
    base_path = "02_ExportedData/02_TLS/raw_pcd"

    try:
        source_cloud = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{base_path}/epoch_1_raw.ply",
                repo_type="dataset",
            )
        )
        target_cloud = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=f"{base_path}/epoch_2_raw.ply",
                repo_type="dataset",
            )
        )
    except Exception as exc:
        logging.error(
            f"Could not download sample data from HuggingFace.\n"
            f"Error: {exc}\n"
            f"Check your internet connection or if dataset '{repo_id}' is accessible."
        )
        sys.exit(1)

    logging.info(f"Source cloud: {source_cloud}")
    logging.info(f"Target cloud: {target_cloud}")

    logging.info("Running F2S3 pipeline...")
    try:
        from f2s3.core import F2S3RunSettings, feature_based_deformation_analysis
    except ImportError as exc:
        logging.error(f"Could not import F2S3 pipeline components.\nError: {exc}")
        sys.exit(1)

    with tempfile.TemporaryDirectory() as tmp_dir:
        results_dir = Path(tmp_dir)

        try:
            settings = F2S3RunSettings(
                source_cloud=source_cloud,
                target_cloud=target_cloud,
                results_dir=results_dir,
                verbose=True,
            )

            with torch.no_grad():
                feature_based_deformation_analysis(settings)

        except Exception as exc:
            logging.error(f"Pipeline raised an exception.\nError: {exc}")
            sys.exit(1)

        output_files = glob.glob(
            str(results_dir / "output" / "results" / "displacement_magnitude_tile_*.txt")
        )

        if not output_files:
            logging.error(
                "Pipeline completed without error but no output files were found in "
                f"{results_dir / 'output' / 'results'}."
            )
            sys.exit(1)

        logging.info(f"Found {len(output_files)} output file(s): {[Path(f).name for f in output_files]}")


    logging.info("PASSED!")
    sys.exit(0)
